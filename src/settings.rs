use config::{Config, ConfigError, Environment, File};
use serde::Deserialize;

#[derive(Deserialize)]
#[allow(unused)]
pub enum ServerMode {
    #[serde(alias = "collect")]
    Collect,

    #[serde(alias = "mock")]
    Mock,
}

#[derive(Deserialize)]
#[allow(unused)]
pub struct TargetServer {
    pub host: String,
}

#[derive(Deserialize)]
#[allow(unused)]
pub struct Server {
    pub host: String,
    pub port: u16,
}

#[derive(Deserialize)]
#[allow(unused)]
pub struct RequestMatching {
    /// Should ModelInfer requests be seen as a different request when the id is different?
    pub match_id: bool,

    /// Should ModelInfer requests be seen as a different request when the paramters differ?
    pub match_parameters: bool,

    /// If match_parameters is enabled, should any parameters be skipped? e.g. request time
    pub skip_parameters: Vec<String>,

    /// When a request has a set of outputs that has already been used in another. Use the other outputs and remove the once that are not needed.
    pub match_pruned_output: bool,
}

#[derive(Deserialize)]
#[allow(unused)]
pub struct RequestCollection {
    pub path: String,
}

#[derive(Deserialize)]
#[allow(unused)]
pub struct Settings {
    pub debug: bool,
    pub mode: ServerMode,
    pub server: Server,
    pub target_server: TargetServer,
    pub request_matching: RequestMatching,
    pub request_collection: RequestCollection,
}

impl Settings {
    pub fn new() -> Result<Self, ConfigError> {
        let s = Config::builder()
            .set_default("debug", false)?
            .set_default("mode", "collect")?
            .set_default("server.host", "0.0.0.0")?
            .set_default("server.port", 50051u16)?
            .set_default("target_server.host", "http://localhost:8001")?
            .set_default("request_matching.match_id", false)?
            .set_default("request_matching.match_parameters", false)?
            .set_default("request_matching.skip_parameters", Vec::<String>::new())?
            .set_default("request_matching.match_pruned_output", false)?
            .set_default("request_collection.path", "collection")
            .unwrap()
            .add_source(File::with_name("config").required(false))
            .add_source(Environment::with_prefix("app"))
            .build()?;

        s.try_deserialize()
    }
}
