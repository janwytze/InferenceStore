use indexmap::IndexMap;
use std::ffi::OsStr;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::{fs, io};

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::input_parsing::{MatchConfig, ProcessedInput};
use crate::output_parsing::ProcessedOutput;

#[derive(Serialize, Deserialize)]
pub struct InputWrapper {
    pub input: ProcessedInput,
}

#[derive(Serialize, Deserialize)]
pub struct OutputWrapper {
    pub output: ProcessedOutput,
}

#[derive(Serialize, Deserialize)]
pub struct InputOutputWrapper {
    pub input: ProcessedInput,
    pub output: ProcessedOutput,
}

impl InputOutputWrapper {
    pub fn new(input: ProcessedInput, output: ProcessedOutput) -> InputOutputWrapper {
        InputOutputWrapper { input, output }
    }

    pub fn hash(&self) -> [u8; 32] {
        let mut hash: [u8; 32] = [0; 32];

        hash[0..8].copy_from_slice(&self.input.inputs_hash());
        hash[8..16].copy_from_slice(&self.input.outputs_hash());
        hash[16..24].copy_from_slice(&self.input.metadata_hash());
        hash[24..32].copy_from_slice(&self.output.hash());

        hash
    }

    pub fn file_name(&self) -> String {
        let hash = self.hash();

        format!(
            "{}_{}_{}_{}.inferstore",
            hex::encode(&hash[0..8]),
            hex::encode(&hash[8..16]),
            hex::encode(&hash[16..24]),
            hex::encode(&hash[24..32]),
        )
    }
}

pub struct InferenceStore {
    // The path where requests/responses are stored on disk.
    path: PathBuf,

    // The in-memory request store.
    inference_store: RwLock<IndexMap<[u8; 32], ProcessedInput>>,
}

impl InferenceStore {
    pub fn new(path: PathBuf) -> InferenceStore {
        InferenceStore {
            path,
            inference_store: Default::default(),
        }
    }
    // Loads all inference files from the inference store path.
    pub fn get_inference_files(&self) -> io::Result<Vec<PathBuf>> {
        let res = fs::read_dir(&self.path)?
            .filter_map(Result::ok)
            .filter(|entry| entry.path().extension() == Some(OsStr::new("inferstore")))
            .map(|r| r.path())
            .collect();

        Ok(res)
    }

    pub async fn load(&self, path: PathBuf) -> anyhow::Result<()> {
        let file = File::open(&path)?;
        let InputWrapper { input } = serde_json::from_reader(file)?;

        let mut write_store = self.inference_store.write().await;

        let mut file_name = path
            .file_name()
            .unwrap()
            .to_os_string()
            .into_string()
            .unwrap();
        file_name.truncate(file_name.len() - 11);
        file_name = file_name.replace("_", "");

        let hash: [u8; 32] = hex::decode(file_name)?.as_slice().try_into()?;

        write_store.insert(hash, input);

        Ok(())
    }

    pub async fn write(
        &self,
        input: ProcessedInput,
        output: ProcessedOutput,
    ) -> anyhow::Result<()> {
        let input_output = InputOutputWrapper::new(input.clone(), output.clone());

        // Create the file.
        fs::create_dir_all(&self.path)?;
        let file = match File::create_new(self.path.join(input_output.file_name())) {
            Ok(file) => file,
            Err(err) => {
                let _test = format!("{err}");
                return Ok(());
            } // TODO check for the type of error.
        };

        // Write the file.
        let mut writer = BufWriter::new(file);
        serde_json::to_writer(&mut writer, &input_output).unwrap();
        writer.flush()?;

        // Write to memory.
        let write_inference_store = &mut self.inference_store.write().await;
        write_inference_store.insert(input_output.hash(), input);

        Ok(())
    }

    pub async fn find_output(
        &self,
        match_input: &ProcessedInput,
        config: MatchConfig,
    ) -> Option<ProcessedOutput> {
        let readable_inference_store = self.inference_store.read().await;

        for (hash, input) in readable_inference_store.deref() {
            if match_input.matches(input, config.clone()) {
                let filename = format!(
                    "{}_{}_{}_{}.inferstore",
                    hex::encode(&hash[0..8]),
                    hex::encode(&hash[8..16]),
                    hex::encode(&hash[16..24]),
                    hex::encode(&hash[24..32]),
                );
                let file = File::open(self.path.join(filename)).ok()?;
                let OutputWrapper { output } = serde_json::from_reader(file).ok()?;

                return Some(output);
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::{BufWriter, Write};

    use tempdir::TempDir;

    use crate::input_parsing::tests::BASE_INPUT;
    use crate::output_parsing::tests::BASE_OUTPUT;

    use super::*;

    #[test]
    fn it_reads_inference_files() {
        // ARRANGE
        let tmp_dir = TempDir::new("inference_store_test").unwrap();
        let tmp_path = tmp_dir.path().to_path_buf();

        File::create(tmp_path.join("1.inferstore")).unwrap();
        File::create(tmp_path.join("2.test")).unwrap();

        let inference_store = InferenceStore {
            path: tmp_path,
            inference_store: Default::default(),
        };

        // ACT
        let files = inference_store.get_inference_files().unwrap();

        // ASSERT
        assert_eq!(files.len(), 1);
        assert_eq!(
            files[0]
                .file_name()
                .unwrap()
                .to_os_string()
                .into_string()
                .unwrap(),
            "1.inferstore"
        );
    }

    #[tokio::test]
    async fn it_loads_inference_file() {
        // ARRANGE
        let tmp_dir = TempDir::new("inference_store_test").unwrap();
        let tmp_path = tmp_dir.path().to_path_buf();

        let path = tmp_path
            .join("c9b7e475dd69fa72_bf645d11f6b25b6f_192d91107cec4716_111f49954e134b85.inferstore");
        let file = File::create(&path).unwrap();

        let mut writer = BufWriter::new(file);
        serde_json::to_writer(
            &mut writer,
            &InputOutputWrapper::new(BASE_INPUT.clone(), BASE_OUTPUT.clone()),
        )
        .unwrap();
        writer.flush().unwrap();

        let inference_store = InferenceStore {
            path: tmp_path,
            inference_store: Default::default(),
        };

        // ACT
        inference_store.load(path).await.unwrap();

        // ASSERT
        let read_inference_store = inference_store.inference_store.read().await;
        assert_eq!(read_inference_store.len(), 1);
        let (_, input) = read_inference_store.first().unwrap();
        assert_eq!(*input, BASE_INPUT.clone());
    }

    #[tokio::test]
    async fn it_writes_inference_file() {
        // ARRANGE
        let tmp_dir = TempDir::new("inference_store_test").unwrap();
        let tmp_path = tmp_dir.path().to_path_buf();

        let inference_store = InferenceStore {
            path: tmp_path.clone(),
            inference_store: Default::default(),
        };

        // ACT
        inference_store
            .write(BASE_INPUT.clone(), BASE_OUTPUT.clone())
            .await
            .unwrap();

        // ASSERT
        assert!(tmp_path
            .join("c9b7e475dd69fa72_bf645d11f6b25b6f_192d91107cec4716_111f49954e134b85.inferstore")
            .exists());
    }

    #[tokio::test]
    async fn it_does_not_write_when_exists() {
        // ARRANGE
        let tmp_dir = TempDir::new("inference_store_test").unwrap();
        let tmp_path = tmp_dir.path().to_path_buf();

        let inference_store = InferenceStore {
            path: tmp_path.clone(),
            inference_store: Default::default(),
        };
        inference_store
            .write(BASE_INPUT.clone(), BASE_OUTPUT.clone())
            .await
            .unwrap();

        // ACT
        inference_store
            .write(BASE_INPUT.clone(), BASE_OUTPUT.clone())
            .await
            .unwrap();

        // ASSERT
        assert!(tmp_path
            .join("c9b7e475dd69fa72_bf645d11f6b25b6f_192d91107cec4716_111f49954e134b85.inferstore")
            .exists());
    }
}
