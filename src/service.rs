use std::sync::Arc;

use tokio::sync::{mpsc, RwLock};
use tonic::codegen::tokio_stream::wrappers::ReceiverStream;
use tonic::codegen::tokio_stream::StreamExt;
use tonic::transport::Channel;
use tonic::{Request, Response, Status, Streaming};

use crate::inference_store::ProcessedInputOutputVecExt;
use inference_protocol::grpc_inference_service_client::GrpcInferenceServiceClient;
use inference_protocol::grpc_inference_service_server::GrpcInferenceService;
use inference_protocol::{
    ModelInferRequest, ModelInferResponse, ModelMetadataRequest, ModelMetadataResponse,
    ModelReadyRequest, ModelReadyResponse, ServerLiveRequest, ServerLiveResponse,
    ServerMetadataRequest, ServerMetadataResponse, ServerReadyRequest, ServerReadyResponse,
};
use log::warn;

use crate::input_parsing::ProcessedInput;
use crate::output_parsing::ProcessedOutput;
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

pub mod inference_protocol {
    tonic::include_proto!("inference");
}

pub struct MockGrpcInferenceService {
    settings: Settings,
    inference_service_client: GrpcInferenceServiceClient<Channel>,
    inference_store: Arc<RwLock<Vec<(ProcessedInput, ProcessedOutput)>>>,
}

impl MockGrpcInferenceService {
    pub fn new(
        settings: Settings,
        inference_service_client: GrpcInferenceServiceClient<Channel>,
    ) -> Self {
        Self {
            settings,
            inference_service_client,
            inference_store: Arc::new(RwLock::new(vec![])),
        }
    }
}

#[tonic::async_trait]
impl GrpcInferenceService for MockGrpcInferenceService {
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

        // Scope for read lock.
        {
            let readable_inference_store = self.inference_store.read().await;

            if let Some(cached_output) =
                readable_inference_store.find_output(&parsed_input, Default::default())
            {
                let response = cached_output.to_response(request.get_ref().clone());
                return Ok(Response::new(response));
            }
        }

        // let json_string = serde_json::to_string(&parsed_input);

        let response = self
            .inference_service_client
            .clone()
            .model_infer(request)
            .await?;

        let processed_response = ProcessedOutput::from_response(response.get_ref());

        // Scope for write lock.
        {
            let writable_inference_store = &mut self.inference_store.write().await;
            writable_inference_store.push((parsed_input, processed_response));
        }

        Ok(Response::new(response.into_inner()))
    }

    type ModelStreamInferStream = ReceiverStream<Result<ModelStreamInferResponse, Status>>;

    async fn model_stream_infer(
        &self,
        request: Request<Streaming<ModelInferRequest>>,
    ) -> Result<Response<Self::ModelStreamInferStream>, Status> {
        let mut stream = request.into_inner();
        let (tx, rx) = mpsc::channel(4);

        let inference_service_client = self.inference_service_client.clone();
        let inference_store = self.inference_store.clone();

        tokio::spawn(async move {
            while let Some(infer_request) = stream.next().await {
                let infer_request = match infer_request {
                    Ok(infer_request) => infer_request,
                    Err(err) => {
                        //
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

                // Scope for read lock.
                {
                    let readable_inference_store = inference_store.read().await;

                    if let Some(cached_output) =
                        readable_inference_store.find_output(&parsed_input, Default::default())
                    {
                        let response = cached_output.to_stream_response(infer_request);
                        if let Err(err) = tx.send(Ok(response)).await {
                            warn!("sending cached response failed: {err}")
                        }
                        return;
                    }
                }

                let response = inference_service_client
                    .clone()
                    .model_infer(infer_request)
                    .await;

                let response = match response {
                    Ok(response) => response,
                    Err(err) => {
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

                // Scope for write lock.
                {
                    let writable_inference_store = &mut inference_store.write().await;
                    writable_inference_store.push((parsed_input, processed_response));
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
        let response = self
            .inference_service_client
            .clone()
            .model_config(request)
            .await
            .expect("ohnee");
        Ok(Response::new(response.into_inner()))
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
