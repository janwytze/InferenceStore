mod inference_store;
mod input_parsing;
mod output_parsing;
mod service;
mod settings;
mod utils;

use crate::inference_store::InferenceStore;
use crate::service::inference_protocol::grpc_inference_service_client::GrpcInferenceServiceClient;
use crate::service::inference_protocol::grpc_inference_service_server::GrpcInferenceServiceServer;
use log::{info, LevelFilter};
use settings::Settings;
use std::fs;
use std::io::ErrorKind::NotFound;
use std::path::PathBuf;
use tonic::transport::Server;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let settings = Settings::new()?;

    log::set_max_level(if settings.debug {
        LevelFilter::Debug
    } else {
        LevelFilter::Info
    });

    let addr = format!("{}:{}", settings.server.host, settings.server.port).parse()?;

    let inference_client = GrpcInferenceServiceClient::connect(settings.target_server.host.clone())
        .await
        .expect("unable to connect to target inference server");

    info!(
        "Connected to target grpc inference service {}",
        settings.target_server.host.clone()
    );

    let inference_store_path = PathBuf::from(settings.request_collection.path.clone());
    let inference_store = InferenceStore::new(inference_store_path.clone());

    let inference_files = match inference_store.get_inference_files() {
        Ok(res) => res,
        Err(err) if err.kind() == NotFound => {
            fs::create_dir_all(inference_store_path.clone())?;
            info!(
                "Created path {} to store inference files",
                inference_store_path
                    .clone()
                    .into_os_string()
                    .into_string()
                    .unwrap(),
            );
            vec![]
        }
        Err(err) => return Err(err.into()),
    };

    for inference_file in inference_files {
        inference_store.load(inference_file).await?;
    }

    let service =
        service::MockGrpcInferenceService::new(settings, inference_store, inference_client);
    let service_server =
        GrpcInferenceServiceServer::new(service).max_decoding_message_size(1024 * 1024 * 128);

    Server::builder()
        .add_service(service_server)
        .serve(addr)
        .await?;

    Ok(())
}
