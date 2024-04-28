mod inference_store;
mod input_parsing;
mod output_parsing;
mod service;
mod settings;
mod utils;

use crate::service::inference_protocol::grpc_inference_service_client::GrpcInferenceServiceClient;
use crate::service::inference_protocol::grpc_inference_service_server::GrpcInferenceServiceServer;
use settings::Settings;
use tonic::transport::Server;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let settings = Settings::new()?;
    let addr = format!("{}:{}", settings.server.host, settings.server.port).parse()?;

    let inference_client = GrpcInferenceServiceClient::connect(settings.target_server.host.clone())
        .await
        .expect("unable to connect to target inference server");

    let service = service::MockGrpcInferenceService::new(settings, inference_client);
    let service_server =
        GrpcInferenceServiceServer::new(service).max_decoding_message_size(1024 * 1024 * 128);

    Server::builder()
        .add_service(service_server)
        .serve(addr)
        .await?;

    Ok(())
}
