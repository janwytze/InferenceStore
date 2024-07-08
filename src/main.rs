mod caching;
mod parsing;
mod service;
mod settings;
mod utils;

use crate::caching::cachestore::CacheStore;
use crate::service::inference_protocol::grpc_inference_service_client::GrpcInferenceServiceClient;
use crate::service::inference_protocol::grpc_inference_service_server::GrpcInferenceServiceServer;
use crate::settings::ServerMode;
use log::{error, info, LevelFilter};
use settings::Settings;
use std::io::ErrorKind::NotFound;
use std::path::PathBuf;
use std::{fs, io};
use tonic::transport::Server;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

    let settings = match Settings::new() {
        Ok(settings) => settings,
        Err(err) => {
            error!("Could not load config: {}", err.to_string());
            std::process::exit(1)
        }
    };

    log::set_max_level(if settings.debug {
        LevelFilter::Debug
    } else {
        LevelFilter::Info
    });

    let addr = format!("{}:{}", settings.server.host, settings.server.port).parse()?;

    let inference_client = match settings.mode {
        ServerMode::Collect => {
            match GrpcInferenceServiceClient::connect(settings.target_server.host.clone()).await {
                Ok(client) => {
                    info!(
                        "Connected to target grpc inference service {}",
                        settings.target_server.host.clone()
                    );
                    Some(client)
                }
                Err(err) => {
                    error!(
                        "Could not connect to grpc inference service {}: {}",
                        settings.target_server.host.clone(),
                        err.to_string()
                    );
                    std::process::exit(1)
                }
            }
        }
        ServerMode::Serve => {
            info!("Started in serving mode, not connecting");
            None
        }
    };

    let inference_store_path = PathBuf::from(&settings.request_collection.path);
    let inference_store = CacheStore::new(inference_store_path.clone());
    let config_store = CacheStore::new(inference_store_path.clone());

    match inference_store.load().await {
        Err(err)
            if err
                .downcast_ref::<io::Error>()
                .map_or(false, |e| e.kind() == NotFound) =>
        {
            fs::create_dir_all(&inference_store_path)?;
            info!(
                "Created path {} to store inference files",
                inference_store_path.display()
            );
        }
        Err(err) => return Err(err.into()),
        _ => {}
    }

    match config_store.load().await {
        Err(err)
            if err
                .downcast_ref::<io::Error>()
                .map_or(false, |e| e.kind() == NotFound) =>
        {
            fs::create_dir_all(&inference_store_path)?;
            info!(
                "Created path {} to store inference files",
                inference_store_path.display()
            );
        }
        Err(err) => return Err(err.into()),
        _ => {}
    }

    let service = service::InferenceStoreGrpcInferenceService::new(
        settings,
        inference_store,
        config_store,
        inference_client,
    );
    let service_server =
        GrpcInferenceServiceServer::new(service).max_decoding_message_size(1024 * 1024 * 128);

    info!("Starting GRPC server on {}", addr);

    Server::builder()
        .add_service(service_server)
        .serve(addr)
        .await?;

    Ok(())
}
