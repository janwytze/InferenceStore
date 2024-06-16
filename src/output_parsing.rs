use crate::input_parsing::Parameter;
use crate::service::inference_protocol::model_infer_response::InferOutputTensor;
use crate::service::inference_protocol::{
    InferParameter, ModelInferRequest, ModelInferResponse, ModelStreamInferResponse,
};
use blake2::{Blake2b, Digest};
use digest::consts::U8;
use serde::{Deserialize, Serialize};
use serde_with::base64::Base64;
use serde_with::serde_as;
use std::collections::BTreeMap;

type Blake2b64 = Blake2b<U8>;

// Represents a parsed form of ModelInferRequest that is less heavy to process as the full request.
// It basically contains the same information, but the content has been hashed to reduce the size.
#[serde_as]
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct ProcessedOutput {
    pub parameters: BTreeMap<String, Option<Parameter>>,
    pub outputs: Vec<Output>,
    #[serde_as(as = "Vec<Base64>")]
    pub raw_output_contents: Vec<Vec<u8>>,
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct Output {
    pub parameters: BTreeMap<String, Option<Parameter>>,
    pub name: String,
    pub datatype: String,
    pub shape: Vec<i64>,
}

impl ProcessedOutput {
    pub fn hash(&self) -> [u8; 8] {
        let mut hasher = Blake2b64::new();

        for (key, value) in &self.parameters {
            blake2::Digest::update(&mut hasher, &key.as_bytes());
            if value.is_some() {
                blake2::Digest::update(&mut hasher, value.as_ref().unwrap().as_bytes());
            }
        }

        for output in &self.outputs {
            blake2::Digest::update(&mut hasher, &output.datatype.as_bytes());
            blake2::Digest::update(&mut hasher, &output.name.as_bytes());

            for shape in &output.shape {
                blake2::Digest::update(&mut hasher, &shape.to_le_bytes());
            }

            for (key, value) in &output.parameters {
                blake2::Digest::update(&mut hasher, &key.as_bytes());
                if value.is_some() {
                    blake2::Digest::update(&mut hasher, value.as_ref().unwrap().as_bytes());
                }
            }
        }

        for output_content in &self.raw_output_contents {
            blake2::Digest::update(&mut hasher, output_content);
        }

        let hash = hasher.finalize();
        let hash: &[u8; 8] = hash.as_slice().try_into().unwrap();

        return *hash;
    }
    pub fn from_response(response: &ModelInferResponse) -> ProcessedOutput {
        return ProcessedOutput {
            parameters: response
                .parameters
                .iter()
                .map(|(key, value)| {
                    (
                        key.to_string(),
                        Parameter::from_infer_parameter(value.clone()),
                    )
                })
                .collect(),
            outputs: response
                .outputs
                .iter()
                .map(
                    |InferOutputTensor {
                         parameters,
                         name,
                         shape,
                         datatype,
                         ..
                     }| {
                        Output {
                            parameters: parameters
                                .iter()
                                .map(|(key, value)| {
                                    (
                                        key.to_string(),
                                        Parameter::from_infer_parameter(value.clone()),
                                    )
                                })
                                .collect(),
                            name: name.clone(),
                            datatype: datatype.clone(),
                            shape: shape.clone(),
                        }
                    },
                )
                .collect(),
            raw_output_contents: response.raw_output_contents.clone(),
        };
    }

    /// Convert the processed output to an actual ModelInferResponse based on the request.
    pub fn to_response(&self, request: ModelInferRequest) -> ModelInferResponse {
        return ModelInferResponse {
            model_name: request.model_name,
            model_version: request.model_version,
            id: request.id,
            parameters: self
                .parameters
                .iter()
                .map(|(name, parameter)| {
                    return (
                        name.clone(),
                        match parameter {
                            None => InferParameter::default(),
                            Some(parameter) => parameter.clone().to_infer_parameter(),
                        },
                    );
                })
                .collect(),
            outputs: self
                .outputs
                .iter()
                .map(
                    |Output {
                         name,
                         datatype,
                         shape,
                         parameters,
                     }| {
                        return InferOutputTensor {
                            name: name.clone(),
                            datatype: datatype.clone(),
                            shape: shape.clone(),
                            parameters: parameters
                                .iter()
                                .map(|(name, parameter)| {
                                    return (
                                        name.clone(),
                                        match parameter {
                                            None => InferParameter::default(),
                                            Some(parameter) => {
                                                parameter.clone().to_infer_parameter()
                                            }
                                        },
                                    );
                                })
                                .collect(),
                            contents: None, // TODO add contents.
                        };
                    },
                )
                .collect(),
            raw_output_contents: self.raw_output_contents.clone(),
        };
    }

    pub fn to_stream_response(&self, request: ModelInferRequest) -> ModelStreamInferResponse {
        return ModelStreamInferResponse {
            error_message: "".to_string(),
            infer_response: Some(self.to_response(request)),
        };
    }
}

#[cfg(test)]
pub mod tests {
    use once_cell::sync::Lazy;

    use super::*;

    pub static BASE_OUTPUT: Lazy<ProcessedOutput> = Lazy::new(|| ProcessedOutput {
        parameters: BTreeMap::from([(
            "test".to_string(),
            Some(Parameter::StringParam("test".to_string())),
        )]),
        outputs: vec![Output {
            parameters: BTreeMap::from([(
                "test".to_string(),
                Some(Parameter::StringParam("test".to_string())),
            )]),
            name: "test".to_string(),
            datatype: "INT64".to_string(),
            shape: vec![1, 2, 3],
        }],
        raw_output_contents: vec![vec![69]],
    });

    #[test]
    fn it_converts_output_to_infer_response() {
        let response = BASE_OUTPUT.clone().to_response(ModelInferRequest {
            model_name: "test".to_string(),
            model_version: "1".to_string(),
            id: "asdf".to_string(),
            parameters: Default::default(),
            inputs: vec![],
            outputs: vec![],
            raw_input_contents: vec![],
        });

        assert_eq!(response.model_name, "test");
        assert_eq!(response.model_version, "1");
        assert_eq!(response.id, "asdf");
    }
}
