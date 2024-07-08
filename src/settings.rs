use crate::parsing::input::MatchConfig;
use config::{Config, Environment, File};
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Deserialize, PartialEq, Clone)]
#[allow(unused)]
pub enum ServerMode {
    // Collect responses to later be used.
    #[serde(alias = "collect")]
    Collect,

    // Serve cached responses.
    #[serde(alias = "serve")]
    Serve,
}

#[derive(Deserialize, Clone)]
#[allow(unused)]
pub struct TargetServer {
    pub host: String,
}

#[derive(Deserialize, Clone)]
#[allow(unused)]
pub struct Server {
    pub host: String,

    pub port: u16,
}

#[derive(Deserialize, PartialEq, Clone)]
#[allow(unused)]
pub enum ParameterMatching {
    // Do not match any parameters.
    #[serde(alias = "disable")]
    Disable,

    // Match all supplied parameters.
    #[serde(alias = "match_keys")]
    MatchKeys,

    // Match all parameters, except for the supplied once.
    #[serde(alias = "ignore_keys")]
    IgnoreKeys,
}

#[derive(Deserialize, Clone)]
#[allow(unused)]
pub struct RequestMatching {
    // When true, the requests id of an incoming request needs to be equal to the request id of a cached request to be considered a match.
    pub match_id: bool,

    // The request parameter matching config.
    pub parameter_matching: ParameterMatching,

    // The request parameter keys that should be matched according to the provided parameter matching config.
    pub parameter_keys: Vec<String>,

    // The input parameter matching config.
    pub input_parameter_matching: ParameterMatching,

    // The input parameter keys that should be matched according to the provided parameter matching config.
    pub input_parameter_keys: HashMap<String, Vec<String>>,

    // The output parameter matching config.
    pub output_parameter_matching: ParameterMatching,

    // The output parameter keys that should be matched according to the provided parameter matching config.
    pub output_parameter_keys: HashMap<String, Vec<String>>,

    // When true, an incoming request that has a subset of outputs of a cached request, is considered matched.
    pub match_pruned_output: bool,
}

#[derive(Deserialize, Clone)]
#[allow(unused)]
pub struct RequestCollection {
    pub path: String,
}

#[derive(Deserialize, Clone)]
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
    pub fn new() -> anyhow::Result<Self> {
        let s = Config::builder()
            .set_default("debug", false)?
            .set_default("mode", "collect")?
            .set_default("server.host", "0.0.0.0")?
            .set_default("server.port", 50051u16)?
            .set_default("target_server.host", "http://localhost:8001")?
            .set_default("request_matching.match_id", false)?
            .set_default("request_matching.parameter_matching", "disable")?
            .set_default("request_matching.parameter_keys", Vec::<String>::new())?
            .set_default("request_matching.input_parameter_matching", "disable")?
            .set_default(
                "request_matching.input_parameter_keys",
                HashMap::<String, Vec<String>>::new(),
            )?
            .set_default("request_matching.output_parameter_matching", "disable")?
            .set_default(
                "request_matching.output_parameter_keys",
                HashMap::<String, Vec<String>>::new(),
            )?
            .set_default("request_matching.match_pruned_output", false)?
            .set_default("request_collection.path", "inferencestore")
            .unwrap()
            .add_source(File::with_name("inferencestore").required(false))
            .add_source(Environment::with_prefix("APP").separator("__"))
            .build()?;

        let c = s.try_deserialize()?;

        Ok(c)
    }

    pub fn get_match_config(&self) -> MatchConfig {
        return MatchConfig {
            match_id: self.request_matching.match_id,
            parameter_keys: if self.request_matching.parameter_matching
                == ParameterMatching::Disable
            {
                vec![]
            } else {
                self.request_matching.parameter_keys.clone()
            },
            exclude_parameters: self.request_matching.parameter_matching
                != ParameterMatching::MatchKeys,
            input_parameter_keys: if self.request_matching.input_parameter_matching
                == ParameterMatching::Disable
            {
                HashMap::new()
            } else {
                self.request_matching.input_parameter_keys.clone()
            },
            exclude_input_parameters: self.request_matching.input_parameter_matching
                != ParameterMatching::MatchKeys,
            output_parameter_keys: if self.request_matching.output_parameter_matching
                == ParameterMatching::Disable
            {
                HashMap::new()
            } else {
                self.request_matching.output_parameter_keys.clone()
            },
            exclude_output_parameters: self.request_matching.output_parameter_matching
                != ParameterMatching::MatchKeys,
            match_pruned_output: self.request_matching.match_pruned_output,
        };
    }
}
