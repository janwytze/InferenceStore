use crate::caching::cachable::Cachable;
use crate::parsing::input::{MatchConfig, ProcessedInput};
use crate::parsing::output::ProcessedOutput;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(Clone)]
pub struct CachableModelInfer {
    dir: PathBuf,
    input: ProcessedInput,
    output_hash: Vec<u8>,
}

impl CachableModelInfer {
    fn get_file_name(&self, output_hash: Vec<u8>) -> String {
        let hash = self.get_hash(output_hash);

        format!(
            "infer-{}#{}#{}#{}.inferstore",
            hex::encode(&hash[0..8]),
            hex::encode(&hash[8..16]),
            hex::encode(&hash[16..24]),
            hex::encode(&hash[24..32]),
        )
    }

    fn get_hash(&self, output_hash: Vec<u8>) -> Vec<u8> {
        let mut hash = Vec::with_capacity(32);

        hash.extend_from_slice(&self.input.inputs_hash());
        hash.extend_from_slice(&self.input.outputs_hash());
        hash.extend_from_slice(&self.input.metadata_hash());
        hash.extend_from_slice(&output_hash);

        hash
    }

    fn new<P: AsRef<Path>>(
        path: P,
        input: ProcessedInput,
        output_hash: Vec<u8>,
    ) -> (PathBuf, Self) {
        let cachable_model_infer = CachableModelInfer {
            dir: path.as_ref().to_path_buf(),
            input,
            output_hash: output_hash.clone(),
        };

        let file_name = cachable_model_infer.get_file_name(output_hash);

        (path.as_ref().join(file_name), cachable_model_infer)
    }
}

#[derive(Serialize, Deserialize)]
pub struct InputOutputWrapper {
    pub input: ProcessedInput,
    pub output: ProcessedOutput,
}

#[derive(Serialize, Deserialize)]
struct OutputWrapper {
    pub output: ProcessedOutput,
}

#[derive(Serialize, Deserialize)]
struct InputWrapper {
    pub input: ProcessedInput,
}

impl Cachable for CachableModelInfer {
    type Input = ProcessedInput;
    type Output = ProcessedOutput;
    type Config = MatchConfig;

    fn get_input(&self) -> anyhow::Result<&ProcessedInput> {
        Ok(&self.input)
    }

    fn get_output(&self) -> anyhow::Result<ProcessedOutput> {
        let file_name = self.get_file_name(self.output_hash.clone());
        let file = File::open(self.dir.join(file_name))?;
        let OutputWrapper { output } = serde_json::from_reader(file)?;

        Ok(output)
    }

    fn from_file<P: AsRef<Path>>(path: P) -> anyhow::Result<Box<Self>> {
        let file = File::open(&path)?;
        let InputWrapper { input } = serde_json::from_reader(file)?;

        let output_hash =
            hex::decode(path.as_ref().file_name().unwrap().to_str().unwrap()[57..73].to_string())
                .unwrap();

        Ok(Box::new(CachableModelInfer {
            dir: path.as_ref().parent().unwrap().to_path_buf(),
            input,
            output_hash,
        }))
    }

    fn new<P: AsRef<Path>>(
        dir: P,
        input: ProcessedInput,
        output: ProcessedOutput,
    ) -> anyhow::Result<(PathBuf, Box<Self>)> {
        let (path, cachable_model_infer) =
            CachableModelInfer::new(dir, input.clone(), output.hash().into());
        let file = File::create_new(path.clone())?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer(&mut writer, &InputOutputWrapper { input, output })?;
        writer.flush()?;

        Ok((path, Box::new(cachable_model_infer)))
    }

    fn matches(&self, input: &ProcessedInput, config: &MatchConfig) -> bool {
        self.input.matches(input, config.clone())
    }

    fn matches_file_name(file_name: String) -> bool {
        file_name.starts_with("infer-")
            && file_name.ends_with(".inferstore")
            && file_name.len() == 84
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::{BufWriter, Write};

    use crate::parsing::input::tests::BASE_INFER_INPUT;
    use crate::parsing::output::tests::BASE_INFER_OUTPUT;
    use tempdir::TempDir;

    use super::*;

    #[test]
    fn it_creates() {
        let tmp_dir = TempDir::new("inference_store_test").unwrap();
        let tmp_path = tmp_dir.path().to_path_buf();

        let (path, cachable): (PathBuf, Box<CachableModelInfer>) = Cachable::new(
            tmp_path.clone(),
            BASE_INFER_INPUT.clone(),
            BASE_INFER_OUTPUT.clone(),
        )
        .expect("could not create cachable");

        let output = cachable.get_output().expect("could not get output");
        let input = cachable.get_input().expect("could not get input");

        assert_eq!(BASE_INFER_INPUT.clone(), *input);
        assert_eq!(BASE_INFER_OUTPUT.clone(), output);
        assert_eq!(path, tmp_path.join("infer-c9b7e475dd69fa72#bf645d11f6b25b6f#192d91107cec4716#111f49954e134b85.inferstore"));
        assert!(tmp_path.join("infer-c9b7e475dd69fa72#bf645d11f6b25b6f#192d91107cec4716#111f49954e134b85.inferstore").exists());
    }

    #[test]
    fn it_loads() {
        let tmp_dir = TempDir::new("inference_store_test").unwrap();
        let tmp_path = tmp_dir.path().to_path_buf();

        let path = tmp_path.clone().join(
            "infer-c9b7e475dd69fa72#bf645d11f6b25b6f#192d91107cec4716#111f49954e134b85.inferstore",
        );
        let file = File::create(&path).unwrap();

        let mut writer = BufWriter::new(file);
        serde_json::to_writer(
            &mut writer,
            &InputOutputWrapper {
                input: BASE_INFER_INPUT.clone(),
                output: BASE_INFER_OUTPUT.clone(),
            },
        )
        .unwrap();
        writer.flush().unwrap();

        let cachable =
            CachableModelInfer::from_file(path.clone()).expect("could not load cachable");

        let input = cachable.get_input().expect("could not get input");
        let output = cachable.get_output().expect("could not get output");

        assert_eq!(BASE_INFER_INPUT.clone(), *input);
        assert_eq!(BASE_INFER_OUTPUT.clone(), output);
        assert_eq!(path, tmp_path.clone().join("infer-c9b7e475dd69fa72#bf645d11f6b25b6f#192d91107cec4716#111f49954e134b85.inferstore"));
        assert!(tmp_path.clone().join("infer-c9b7e475dd69fa72#bf645d11f6b25b6f#192d91107cec4716#111f49954e134b85.inferstore").exists());
    }

    #[test]
    fn it_matches_input() {
        let tmp_dir = TempDir::new("inference_store_test").unwrap();
        let tmp_path = tmp_dir.path().to_path_buf();

        let (_, cachable): (PathBuf, Box<CachableModelInfer>) = Cachable::new(
            tmp_path.clone(),
            BASE_INFER_INPUT.clone(),
            BASE_INFER_OUTPUT.clone(),
        )
        .expect("could not create cachable");

        assert!(cachable.matches(&BASE_INFER_INPUT.clone(), &Default::default()));
    }

    #[test]
    fn it_matches_file_name() {
        assert!(CachableModelInfer::matches_file_name(
            "infer-c9b7e475dd69fa72#bf645d11f6b25b6f#192d91107cec4716#111f49954e134b85.inferstore"
                .to_string()
        ));
        assert!(!CachableModelInfer::matches_file_name(
            "infer-asdf.inferstore".to_string()
        ));
    }
}
