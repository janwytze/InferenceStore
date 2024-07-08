use std::sync::Arc;

use tokio::sync::mpsc;
use tonic::codegen::tokio_stream::wrappers::ReceiverStream;
use tonic::codegen::tokio_stream::StreamExt;
use tonic::transport::Channel;
use tonic::{Request, Response, Status, Streaming};

use crate::caching::cachable_modelconfig::CachableModelConfig;
use crate::caching::cachable_modelinfer::CachableModelInfer;
use crate::caching::cachestore::CacheStore;
use crate::parsing::input::ProcessedInput;
use crate::parsing::output::ProcessedOutput;
use crate::service::inference_protocol::{
    CudaSharedMemoryRegisterRequest, CudaSharedMemoryRegisterResponse,
    CudaSharedMemoryStatusRequest, CudaSharedMemoryStatusResponse,
    CudaSharedMemoryUnregisterRequest, CudaSharedMemoryUnregisterResponse, LogSettingsRequest,
    LogSettingsResponse, ModelConfigRequest, ModelConfigResponse, ModelStatisticsRequest,
    ModelStatisticsResponse, ModelStreamInferResponse, RepositoryIndexRequest,
    RepositoryIndexResponse, RepositoryModelLoadRequest, RepositoryModelLoadResponse,
    RepositoryModelUnloadRequest, RepositoryModelUnloadResponse, SystemSharedMemoryRegisterRequest,
    SystemSharedMemoryRegisterResponse, SystemSharedMemoryStatusRequest,
    SystemSharedMemoryStatusResponse, SystemSharedMemoryUnregisterRequest,
    SystemSharedMemoryUnregisterResponse, TraceSettingRequest, TraceSettingResponse,
};
use crate::settings::Settings;
use inference_protocol::grpc_inference_service_client::GrpcInferenceServiceClient;
use inference_protocol::grpc_inference_service_server::GrpcInferenceService;
use inference_protocol::{
    ModelInferRequest, ModelInferResponse, ModelMetadataRequest, ModelMetadataResponse,
    ModelReadyRequest, ModelReadyResponse, ServerLiveRequest, ServerLiveResponse,
    ServerMetadataRequest, ServerMetadataResponse, ServerReadyRequest, ServerReadyResponse,
};
use log::{debug, warn};

pub mod inference_protocol {
    tonic::include_proto!("inference");
}

pub struct InferenceStoreGrpcInferenceService {
    settings: Settings,
    inference_service_client: Option<GrpcInferenceServiceClient<Channel>>,
    inference_store: Arc<CacheStore<CachableModelInfer>>,
    config_store: Arc<CacheStore<CachableModelConfig>>,
}

impl InferenceStoreGrpcInferenceService {
    pub fn new(
        settings: Settings,
        inference_store: CacheStore<CachableModelInfer>,
        config_store: CacheStore<CachableModelConfig>,
        inference_service_client: Option<GrpcInferenceServiceClient<Channel>>,
    ) -> Self {
        Self {
            inference_store: Arc::new(inference_store),
            config_store: Arc::new(config_store),
            settings,
            inference_service_client,
        }
    }
}

#[tonic::async_trait]
impl GrpcInferenceService for InferenceStoreGrpcInferenceService {
    async fn server_live(
        &self,
        _request: Request<ServerLiveRequest>,
    ) -> Result<Response<ServerLiveResponse>, Status> {
        Ok(Response::new(ServerLiveResponse { live: true }))
    }

    async fn server_ready(
        &self,
        _request: Request<ServerReadyRequest>,
    ) -> Result<Response<ServerReadyResponse>, Status> {
        Ok(Response::new(ServerReadyResponse { ready: true }))
    }

    async fn model_ready(
        &self,
        _request: Request<ModelReadyRequest>,
    ) -> Result<Response<ModelReadyResponse>, Status> {
        Ok(Response::new(ModelReadyResponse { ready: true }))
    }

    async fn server_metadata(
        &self,
        _request: Request<ServerMetadataRequest>,
    ) -> Result<Response<ServerMetadataResponse>, Status> {
        Ok(Response::new(ServerMetadataResponse {
            name: String::from("Inference Store Server"),
            version: String::from("0.0.0"),
            extensions: Vec::new(),
        }))
    }
    async fn model_metadata(
        &self,
        _request: Request<ModelMetadataRequest>,
    ) -> Result<Response<ModelMetadataResponse>, Status> {
        Ok(Response::new(ModelMetadataResponse {
            name: String::from("test"),
            platform: String::from("test"),
            inputs: Vec::new(),
            outputs: Vec::new(),
            versions: Vec::new(),
        }))
    }

    async fn model_infer(
        &self,
        request: Request<ModelInferRequest>,
    ) -> Result<Response<ModelInferResponse>, Status> {
        let parsed_input = ProcessedInput::from_infer_request(request.get_ref().clone());

        if let Some(cached_output) = self
            .inference_store
            .find_output(&parsed_input, &self.settings.get_match_config())
            .await
        {
            let response = cached_output.to_response(request.get_ref().clone());
            return Ok(Response::new(response));
        }

        // When self.inference_service_client is None, Serve mode is enabled.
        // In Serve mode only requests from cache will be served.
        let inference_service_client = match &self.inference_service_client {
            Some(client) => client,
            None => return Err(Status::not_found("could not match request")),
        };

        let response = inference_service_client
            .clone()
            .model_infer(request)
            .await?;

        let processed_response = ProcessedOutput::from_response(response.get_ref());

        if let Err(err) = self
            .inference_store
            .store(parsed_input, processed_response)
            .await
        {
            return Err(Status::unknown(err.to_string()));
        }

        Ok(Response::new(response.into_inner()))
    }

    type ModelStreamInferStream = ReceiverStream<Result<ModelStreamInferResponse, Status>>;

    async fn model_stream_infer(
        &self,
        request: Request<Streaming<ModelInferRequest>>,
    ) -> Result<Response<Self::ModelStreamInferStream>, Status> {
        debug!("Received model_stream_infer request");

        let mut stream = request.into_inner();
        let (tx, rx) = mpsc::channel(4);

        let inference_service_client = self.inference_service_client.clone();
        let inference_store = self.inference_store.clone();
        let settings = self.settings.clone();

        tokio::spawn(async move {
            while let Some(infer_request) = stream.next().await {
                let infer_request = match infer_request {
                    Ok(infer_request) => infer_request,
                    Err(err) => {
                        debug!("Error receiving request from stream: {err}");
                        let _ = tx
                            .send(Ok(ModelStreamInferResponse {
                                error_message: err.to_string(),
                                infer_response: None,
                            }))
                            .await;
                        return;
                    }
                };
                let parsed_input = ProcessedInput::from_infer_request(infer_request.clone());

                if let Some(cached_output) = inference_store
                    .find_output(&parsed_input, &settings.get_match_config())
                    .await
                {
                    debug!("Found input in cache, return the cached output");

                    let response = cached_output.to_stream_response(infer_request);
                    if let Err(err) = tx.send(Ok(response)).await {
                        warn!("sending cached response failed: {err}")
                    }
                    return;
                }

                // When self.inference_service_client is None, Serve mode is enabled.
                // In Serve mode only requests from cache will be served.
                let inference_service_client = match &inference_service_client {
                    Some(client) => client,
                    None => {
                        if let Err(err) = tx
                            .send(Err(Status::not_found("could not match request")))
                            .await
                        {
                            warn!("sending inference error response failed: {err}")
                        }

                        return;
                    }
                };

                debug!("Input not found in cache, calling the target grpc server");

                let response = inference_service_client
                    .clone()
                    .model_infer(infer_request)
                    .await;

                let response = match response {
                    Ok(response) => response,
                    Err(err) => {
                        debug!("Target GRPC server returned error: {err}");
                        if let Err(err) = tx
                            .send(Ok(ModelStreamInferResponse {
                                error_message: err.to_string(),
                                infer_response: None,
                            }))
                            .await
                        {
                            warn!("sending inference error response failed: {err}")
                        }
                        return;
                    }
                };

                let processed_response = ProcessedOutput::from_response(response.get_ref());

                debug!("Writing target GRPC server response to disk");

                if let Err(err) = inference_store
                    .store(parsed_input, processed_response)
                    .await
                {
                    let _ = tx
                        .send(Ok(ModelStreamInferResponse {
                            error_message: format!("{err}"),
                            infer_response: None,
                        }))
                        .await;
                    return;
                }

                if let Err(err) = tx
                    .send(Ok(ModelStreamInferResponse {
                        error_message: "".to_string(),
                        infer_response: Some(response.into_inner()),
                    }))
                    .await
                {
                    warn!("sending inference response failed: {err}")
                }
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }

    async fn model_config(
        &self,
        request: Request<ModelConfigRequest>,
    ) -> Result<Response<ModelConfigResponse>, Status> {
        if let Some(cached_output) = self
            .config_store
            .find_output(request.get_ref(), &Default::default())
            .await
        {
            return Ok(Response::new(cached_output));
        }

        let inference_service_client = match &self.inference_service_client {
            Some(client) => client,
            None => {
                return Err(Status::unavailable(
                    "uncached model config not available during serving mode",
                ))
            }
        };

        match inference_service_client
            .clone()
            .model_config(request.get_ref().clone())
            .await
        {
            Ok(res) => {
                self.config_store
                    .store(request.into_inner(), res.get_ref().clone())
                    .await
                    .unwrap();
                Ok(Response::new(res.get_ref().clone()))
            }
            Err(err) => Err(Status::unknown(err.to_string())),
        }
    }

    async fn model_statistics(
        &self,
        _request: Request<ModelStatisticsRequest>,
    ) -> Result<Response<ModelStatisticsResponse>, Status> {
        todo!()
    }

    async fn repository_index(
        &self,
        _request: Request<RepositoryIndexRequest>,
    ) -> Result<Response<RepositoryIndexResponse>, Status> {
        todo!()
    }

    async fn repository_model_load(
        &self,
        _request: Request<RepositoryModelLoadRequest>,
    ) -> Result<Response<RepositoryModelLoadResponse>, Status> {
        todo!()
    }

    async fn repository_model_unload(
        &self,
        _request: Request<RepositoryModelUnloadRequest>,
    ) -> Result<Response<RepositoryModelUnloadResponse>, Status> {
        todo!()
    }

    async fn system_shared_memory_status(
        &self,
        _request: Request<SystemSharedMemoryStatusRequest>,
    ) -> Result<Response<SystemSharedMemoryStatusResponse>, Status> {
        todo!()
    }

    async fn system_shared_memory_register(
        &self,
        _request: Request<SystemSharedMemoryRegisterRequest>,
    ) -> Result<Response<SystemSharedMemoryRegisterResponse>, Status> {
        todo!()
    }

    async fn system_shared_memory_unregister(
        &self,
        _request: Request<SystemSharedMemoryUnregisterRequest>,
    ) -> Result<Response<SystemSharedMemoryUnregisterResponse>, Status> {
        todo!()
    }

    async fn cuda_shared_memory_status(
        &self,
        _request: Request<CudaSharedMemoryStatusRequest>,
    ) -> Result<Response<CudaSharedMemoryStatusResponse>, Status> {
        todo!()
    }

    async fn cuda_shared_memory_register(
        &self,
        _request: Request<CudaSharedMemoryRegisterRequest>,
    ) -> Result<Response<CudaSharedMemoryRegisterResponse>, Status> {
        todo!()
    }

    async fn cuda_shared_memory_unregister(
        &self,
        _request: Request<CudaSharedMemoryUnregisterRequest>,
    ) -> Result<Response<CudaSharedMemoryUnregisterResponse>, Status> {
        todo!()
    }

    async fn trace_setting(
        &self,
        _request: Request<TraceSettingRequest>,
    ) -> Result<Response<TraceSettingResponse>, Status> {
        todo!()
    }

    async fn log_settings(
        &self,
        _request: Request<LogSettingsRequest>,
    ) -> Result<Response<LogSettingsResponse>, Status> {
        todo!()
    }
}
