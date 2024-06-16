use std::ffi::OsStr;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::ops::Deref;
use std::path::PathBuf;
use std::{fs, io};

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::input_parsing::{MatchConfig, ProcessedInput};
use crate::output_parsing::ProcessedOutput;

#[derive(Serialize, Deserialize)]
pub struct InputOutput {
    pub input: ProcessedInput,
    pub output: ProcessedOutput,
}

impl InputOutput {
    pub fn new(input: ProcessedInput, output: ProcessedOutput) -> InputOutput {
        InputOutput { input, output }
    }

    pub fn file_name(&self) -> String {
        let inputs_hash = hex::encode(self.input.inputs_hash());
        let outputs_hash = hex::encode(self.input.outputs_hash());
        let metadata_hash = hex::encode(self.input.metadata_hash());
        let result_hash = hex::encode(self.output.hash());

        format!("{inputs_hash}_{outputs_hash}_{metadata_hash}_{result_hash}.inferstore")
    }
}

pub struct InferenceStore {
    // The path where requests/responses are stored on disk.
    path: PathBuf,

    // The in-memory request/response store.
    inference_store: RwLock<Vec<(ProcessedInput, ProcessedOutput)>>,
}

impl InferenceStore {
    pub fn new(path: PathBuf) -> InferenceStore {
        InferenceStore {
            path: path,
            inference_store: Default::default(),
        }
    }
    // Loads all inference files from the inference store path.
    pub fn get_inference_files(&self) -> io::Result<Vec<PathBuf>> {
        let res = fs::read_dir(self.path.clone())?
            .filter_map(Result::ok)
            .filter(|entry| entry.path().extension() == Some(OsStr::new("inferstore")))
            .map(|r| r.path())
            .collect();

        Ok(res)
    }

    pub async fn load(&self, path: PathBuf) -> anyhow::Result<()> {
        let file = File::open(path)?;
        let InputOutput { input, output } = serde_json::from_reader(file)?;

        let mut write_store = self.inference_store.write().await;

        write_store.push((input, output));

        Ok(())
    }

    pub async fn write(
        &self,
        input: ProcessedInput,
        output: ProcessedOutput,
    ) -> anyhow::Result<()> {
        let input_output = InputOutput::new(input.clone(), output.clone());

        // Create the file.
        fs::create_dir_all(self.path.clone())?;
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
        write_inference_store.push((input, output));

        Ok(())
    }

    pub async fn find_output(
        &self,
        match_input: &ProcessedInput,
        config: MatchConfig,
    ) -> Option<ProcessedOutput> {
        let readable_inference_store = self.inference_store.read().await;

        for (input, output) in readable_inference_store.deref() {
            if match_input.matches(input, config.clone()) {
                return Some(output.clone());
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

        File::create(tmp_dir.path().join("1.inferstore")).unwrap();
        File::create(tmp_dir.path().join("2.test")).unwrap();

        let tmp_path = tmp_dir.into_path();

        let inference_store = InferenceStore {
            path: tmp_path.clone(),
            inference_store: Default::default(),
        };

        // ACT
        let files = inference_store.get_inference_files().unwrap();
        fs::remove_dir_all(tmp_path).unwrap();

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

        let path = tmp_dir.path().join("1.inferstore");
        let file = File::create(&path).unwrap();

        let mut writer = BufWriter::new(file);
        serde_json::to_writer(
            &mut writer,
            &InputOutput::new(BASE_INPUT.clone(), BASE_OUTPUT.clone()),
        )
        .unwrap();
        writer.flush().unwrap();

        let tmp_path = tmp_dir.into_path();

        let inference_store = InferenceStore {
            path: tmp_path.clone(),
            inference_store: Default::default(),
        };

        // ACT
        inference_store.load(path).await.unwrap();
        fs::remove_dir_all(tmp_path).unwrap();

        // ASSERT
        let read_inference_store = inference_store.inference_store.read().await;
        assert_eq!(read_inference_store.len(), 1);
        let (input, output) = read_inference_store.first().unwrap();
        assert_eq!(*input, BASE_INPUT.clone());
        assert_eq!(*output, BASE_OUTPUT.clone());
    }

    #[tokio::test]
    async fn it_writes_inference_file() {
        // ARRANGE
        let tmp_dir = TempDir::new("inference_store_test").unwrap();
        let tmp_path = tmp_dir.into_path();
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
        fs::remove_dir_all(tmp_path).unwrap();
    }

    #[tokio::test]
    async fn it_does_not_write_when_exists() {
        // ARRANGE
        let tmp_dir = TempDir::new("inference_store_test").unwrap();
        let tmp_path = tmp_dir.into_path();
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
            .clone()
            .join("c9b7e475dd69fa72_bf645d11f6b25b6f_192d91107cec4716_111f49954e134b85.inferstore")
            .exists());
        fs::remove_dir_all(tmp_path).unwrap();
    }
}
