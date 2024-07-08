use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use urlencoding::{decode, encode};

use crate::caching::cachable::Cachable;
use crate::service::inference_protocol::{ModelConfigRequest, ModelConfigResponse};

#[derive(Clone)]
pub struct CachableModelConfig {
    input: ModelConfigRequest,
    output: ModelConfigResponse,
}

impl Cachable for CachableModelConfig {
    type Input = ModelConfigRequest;
    type Output = ModelConfigResponse;
    type Config = ();

    fn get_input(&self) -> anyhow::Result<&ModelConfigRequest> {
        Ok(&self.input)
    }

    fn get_output(&self) -> anyhow::Result<ModelConfigResponse> {
        Ok(self.output.clone())
    }

    fn from_file<P: AsRef<Path>>(path: P) -> anyhow::Result<Box<Self>> {
        let file = File::open(&path)?;
        let model_config_response: ModelConfigResponse = serde_json::from_reader(file)?;

        let file_stem = path.as_ref().file_stem().unwrap().to_str().unwrap();
        let mut parts = file_stem[7..file_stem.len()].split('#');

        let model_config_request = ModelConfigRequest {
            name: decode(parts.next().unwrap()).unwrap().to_string(),
            version: decode(parts.next().unwrap()).unwrap().to_string(),
        };

        Ok(Box::new(CachableModelConfig {
            input: model_config_request,
            output: model_config_response,
        }))
    }

    fn new<P: AsRef<Path>>(
        dir: P,
        input: ModelConfigRequest,
        output: ModelConfigResponse,
    ) -> anyhow::Result<(PathBuf, Box<Self>)> {
        let cachable = CachableModelConfig {
            input: input.clone(),
            output: output.clone(),
        };
        let ModelConfigRequest { name, version } = input;
        let file_name = format!(
            "config-{}#{}.inferstore",
            encode(name.as_str()),
            encode(version.as_str())
        );

        let path = dir.as_ref().join(file_name);
        let file = File::create_new(path.clone())?;

        let mut writer = BufWriter::new(file);
        serde_json::to_writer(&mut writer, &output)?;
        writer.flush()?;

        Ok((path, Box::new(cachable)))
    }

    fn matches(&self, input: &ModelConfigRequest, _config: &()) -> bool {
        self.input.name == input.name && self.input.version == input.version
    }

    fn matches_file_name(file_name: String) -> bool {
        file_name.starts_with("config-") && file_name.ends_with(".inferstore")
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::{BufWriter, Write};

    use once_cell::sync::Lazy;
    use tempdir::TempDir;

    use crate::service::inference_protocol::ModelConfig;

    use super::*;

    pub static BASE_CONFIG_OUTPUT: Lazy<ModelConfigResponse> = Lazy::new(|| ModelConfigResponse {
        config: Some(ModelConfig {
            name: "".to_string(),
            platform: "".to_string(),
            backend: "".to_string(),
            runtime: "".to_string(),
            version_policy: None,
            max_batch_size: 0,
            input: vec![],
            output: vec![],
            batch_input: vec![],
            batch_output: vec![],
            optimization: None,
            instance_group: vec![],
            default_model_filename: "".to_string(),
            cc_model_filenames: Default::default(),
            metric_tags: Default::default(),
            parameters: Default::default(),
            model_warmup: vec![],
            model_operations: None,
            model_transaction_policy: None,
            model_repository_agents: None,
            response_cache: None,
            scheduling_choice: None,
        }),
    });

    #[test]
    fn it_creates() {
        let tmp_dir = TempDir::new("inference_store_test").unwrap();
        let tmp_path = tmp_dir.path().to_path_buf();

        let req = ModelConfigRequest {
            name: "test".to_string(),
            version: "1".to_string(),
        };

        let (path, cachable) =
            CachableModelConfig::new(tmp_path.clone(), req.clone(), BASE_CONFIG_OUTPUT.clone())
                .expect("could not create cachable");

        let output = cachable.get_output().expect("could not get output");
        let input = cachable.get_input().expect("could not get input");

        assert_eq!(req, *input);
        assert_eq!(BASE_CONFIG_OUTPUT.clone(), output);
        assert_eq!(path, tmp_path.join("config-test#1.inferstore"));
        assert!(tmp_path.join("config-test#1.inferstore").exists());
    }

    #[test]
    fn it_loads() {
        let tmp_dir = TempDir::new("inference_store_test").unwrap();
        let tmp_path = tmp_dir.path().to_path_buf();

        let path = tmp_path.clone().join("config-test#1.inferstore");
        let file = File::create(&path).unwrap();

        let mut writer = BufWriter::new(file);
        serde_json::to_writer(&mut writer, &BASE_CONFIG_OUTPUT.clone()).unwrap();
        writer.flush().unwrap();

        let cachable =
            CachableModelConfig::from_file(path.clone()).expect("could not load cachable");

        let input = cachable.get_input().expect("could not get input");
        let output = cachable.get_output().expect("could not get output");

        assert_eq!(
            ModelConfigRequest {
                name: "test".to_string(),
                version: "1".to_string()
            },
            *input
        );
        assert_eq!(BASE_CONFIG_OUTPUT.clone(), output);
        assert_eq!(path, tmp_path.clone().join("config-test#1.inferstore"));
        assert!(tmp_path.clone().join("config-test#1.inferstore").exists());
    }

    #[test]
    fn it_saves_and_loads_special_characters() {
        let tmp_dir = TempDir::new("inference_store_test").unwrap();
        let tmp_path = tmp_dir.path().to_path_buf();

        let req = ModelConfigRequest {
            name: "_test-".to_string(),
            version: "_1-".to_string(),
        };

        let (path, cachable) =
            CachableModelConfig::new(tmp_path.clone(), req.clone(), BASE_CONFIG_OUTPUT.clone())
                .expect("could not create cachable");

        assert_eq!("_test-", cachable.input.name);
        assert_eq!("_1-", cachable.input.version);

        let cachable = CachableModelConfig::from_file(path).expect("could not load cachable");

        assert_eq!("_test-", cachable.input.name);
        assert_eq!("_1-", cachable.input.version);
    }

    #[test]
    fn it_matches_input() {
        let tmp_dir = TempDir::new("inference_store_test").unwrap();
        let tmp_path = tmp_dir.path().to_path_buf();

        let req = ModelConfigRequest {
            name: "test".to_string(),
            version: "1".to_string(),
        };

        let (_, cachable) =
            CachableModelConfig::new(tmp_path, req.clone(), BASE_CONFIG_OUTPUT.clone())
                .expect("could not create cachable");

        assert!(cachable.matches(&req, &Default::default()));
    }

    #[test]
    fn it_matches_file_name() {
        assert!(CachableModelConfig::matches_file_name(
            "config-test#1.inferstore".to_string()
        ));
        assert!(!CachableModelConfig::matches_file_name(
            "asdf.inferstore".to_string()
        ));
    }
}
