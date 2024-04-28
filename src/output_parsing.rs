use crate::input_parsing::Parameter;
use crate::service::inference_protocol::model_infer_response::InferOutputTensor;
use crate::service::inference_protocol::{
    InferParameter, ModelInferRequest, ModelInferResponse, ModelStreamInferResponse,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Represents a parsed form of ModelInferRequest that is less heavy to process as the full request.
// It basically contains the same information, but the content has been hashed to reduce the size.
#[derive(Serialize, Deserialize, Clone)]
pub struct ProcessedOutput {
    parameters: HashMap<String, Option<Parameter>>,
    outputs: Vec<Output>,
    raw_output_contents: Vec<Vec<u8>>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Output {
    parameters: HashMap<String, Option<Parameter>>,
    name: String,
    datatype: String,
    shape: Vec<i64>,
}

impl ProcessedOutput {
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
