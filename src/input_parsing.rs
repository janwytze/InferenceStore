use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::service::inference_protocol::infer_parameter::ParameterChoice;
use crate::service::inference_protocol::model_infer_request::{
    InferInputTensor, InferRequestedOutputTensor,
};
use crate::service::inference_protocol::{InferParameter, ModelInferRequest};
use crate::utils::hashmap_compare;

// Represents a parsed form of ModelInferRequest that is less heavy to process as the full request.
// It basically contains the same information, but the content has been hashed to reduce the size.
#[derive(Serialize, Deserialize, Clone)]
pub struct ProcessedInput {
    model_name: String,
    model_version: String,
    id: String,
    parameters: HashMap<String, Option<Parameter>>,
    inputs: Vec<Input>,
    outputs: Vec<Output>,
    hash: String,
}

#[derive(Clone)]
pub struct MatchConfig {
    match_id: bool,
    parameter_keys: Vec<String>,
    exclude_parameters: bool,
    input_parameter_keys: HashMap<String, Vec<String>>,
    exclude_input_parameters: bool,
    output_parameter_keys: HashMap<String, Vec<String>>,
    exclude_output_parameters: bool,
    match_pruned_output: bool,
}

impl Default for MatchConfig {
    fn default() -> MatchConfig {
        MatchConfig {
            match_id: false,
            parameter_keys: vec![],
            exclude_parameters: true,
            input_parameter_keys: Default::default(),
            exclude_input_parameters: true,
            output_parameter_keys: Default::default(),
            exclude_output_parameters: true,
            match_pruned_output: true,
        }
    }
}

impl ProcessedInput {
    /// Parse a ModelInfer request in a format that makes matching it with future requests easier.
    pub fn from_infer_request(req: ModelInferRequest) -> ProcessedInput {
        let mut hasher = Sha256::new();

        // TODO parse inputs if there are not raw_input_contents.
        for content in req.raw_input_contents {
            Digest::update(&mut hasher, content);
        }

        let hash = format!("{:#01x}", hasher.finalize());

        return ProcessedInput {
            model_name: req.model_name,
            model_version: req.model_version,
            id: req.id,
            parameters: req
                .parameters
                .iter()
                .map(|(key, value)| {
                    (
                        key.to_string(),
                        Parameter::from_infer_parameter(value.clone()),
                    )
                })
                .collect(),
            inputs: req
                .inputs
                .iter()
                .map(|input: &InferInputTensor| Input {
                    name: input.clone().name,
                    datatype: input.clone().datatype,
                    shape: input.clone().shape,
                    parameters: input
                        .parameters
                        .iter()
                        .map(|(key, value)| {
                            (
                                key.to_string(),
                                Parameter::from_infer_parameter(value.clone()),
                            )
                        })
                        .collect(),
                })
                .collect(),
            outputs: req
                .outputs
                .iter()
                .map(|output: &InferRequestedOutputTensor| Output {
                    name: output.clone().name,
                    parameters: output
                        .parameters
                        .iter()
                        .map(|(key, value)| {
                            (
                                key.to_string(),
                                Parameter::from_infer_parameter(value.clone()),
                            )
                        })
                        .collect(),
                })
                .collect(),
            hash,
        };
    }

    /// Check if the provided input is compatible with this input.
    ///
    /// # Arguments
    ///
    /// * `other_input` - The input to compare this input to.
    /// * `match_id` - Should the `id` be compared?
    pub fn matches(&self, other_input: &ProcessedInput, config: MatchConfig) -> bool {
        if self.model_name != other_input.model_name
            || self.model_version != other_input.model_version
            || self.hash != other_input.hash
        {
            return false;
        }

        if config.match_id && self.id != other_input.id {
            return false;
        }

        if !hashmap_compare(
            self.parameters.clone(),
            other_input.parameters.clone(),
            config.parameter_keys,
            config.exclude_parameters,
        ) {
            return false;
        }

        let self_inputs: HashMap<_, _> = self
            .inputs
            .iter()
            .map(|input| (input.name.clone(), input.clone()))
            .collect();

        let other_inputs: HashMap<_, _> = other_input
            .inputs
            .iter()
            .map(|input| (input.name.clone(), input.clone()))
            .collect();

        for (key, self_value) in self_inputs {
            if let Some(other_value) = other_inputs.get(&key) {
                if self_value.name != other_value.name
                    || self_value.datatype != other_value.datatype
                    || self_value.shape != other_value.shape
                {
                    return false;
                }

                if !hashmap_compare(
                    self_value.parameters,
                    other_value.parameters.clone(),
                    config
                        .input_parameter_keys
                        .clone()
                        .entry(key)
                        .or_insert(Vec::new())
                        .clone(),
                    config.exclude_input_parameters,
                ) {
                    return false;
                }
            } else {
                return false;
            }
        }

        let self_outputs: HashMap<_, _> = self
            .outputs
            .iter()
            .map(|output| (output.name.clone(), output.clone()))
            .collect();

        let other_outputs: HashMap<_, _> = other_input
            .outputs
            .iter()
            .map(|output| (output.name.clone(), output.clone()))
            .collect();

        for (key, self_value) in self_outputs {
            if let Some(other_value) = other_outputs.get(&key) {
                if self_value.name != other_value.name {
                    return false;
                }

                if !hashmap_compare(
                    self_value.parameters,
                    other_value.parameters.clone(),
                    config
                        .output_parameter_keys
                        .clone()
                        .entry(key)
                        .or_insert(Vec::new())
                        .clone(),
                    config.exclude_output_parameters,
                ) {
                    return false;
                }
            } else {
                return false;
            }
        }

        return true;
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Input {
    name: String,
    datatype: String,
    shape: Vec<i64>,
    parameters: HashMap<String, Option<Parameter>>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Output {
    name: String,
    parameters: HashMap<String, Option<Parameter>>,
}

#[derive(Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum Parameter {
    BoolParam(bool),
    Int64Param(i64),
    StringParam(String),
    DoubleParam(f64),
    Uint64Param(u64),
}

impl Parameter {
    pub fn from_infer_parameter(parameter: InferParameter) -> Option<Parameter> {
        match parameter.parameter_choice {
            None => None,
            Some(p) => match p {
                ParameterChoice::BoolParam(v) => Some(Parameter::BoolParam(v)),
                ParameterChoice::Int64Param(v) => Some(Parameter::Int64Param(v)),
                ParameterChoice::StringParam(v) => Some(Parameter::StringParam(v)),
                ParameterChoice::DoubleParam(v) => Some(Parameter::DoubleParam(v)),
                ParameterChoice::Uint64Param(v) => Some(Parameter::Uint64Param(v)),
            },
        }
    }

    pub fn to_infer_parameter(self) -> InferParameter {
        InferParameter {
            parameter_choice: match self {
                Parameter::BoolParam(v) => Some(ParameterChoice::BoolParam(v)),
                Parameter::Int64Param(v) => Some(ParameterChoice::Int64Param(v)),
                Parameter::StringParam(v) => Some(ParameterChoice::StringParam(v)),
                Parameter::DoubleParam(v) => Some(ParameterChoice::DoubleParam(v)),
                Parameter::Uint64Param(v) => Some(ParameterChoice::Uint64Param(v)),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use once_cell::sync::Lazy;

    use super::*;

    static BASE_INPUT: Lazy<ProcessedInput> = Lazy::new(|| ProcessedInput {
        model_name: "test".to_string(),
        model_version: "1".to_string(),
        id: "1".to_string(),
        parameters: HashMap::from([
            (
                "param1".to_string(),
                Some(Parameter::StringParam("param_value1".to_string())),
            ),
            (
                "param2".to_string(),
                Some(Parameter::StringParam("param_value2".to_string())),
            ),
        ]),
        inputs: vec![Input {
            name: "input1".to_string(),
            datatype: "INT64".to_string(),
            shape: vec![1, 2, 3],
            parameters: HashMap::from([
                (
                    "input_param1".to_string(),
                    Some(Parameter::StringParam("input_param_value1".to_string())),
                ),
                (
                    "input_param2".to_string(),
                    Some(Parameter::StringParam("input_param_value2".to_string())),
                ),
            ]),
        }],
        outputs: vec![Output {
            name: "output1".to_string(),
            parameters: HashMap::from([
                (
                    "output_param1".to_string(),
                    Some(Parameter::StringParam("output_param_value1".to_string())),
                ),
                (
                    "output_param2".to_string(),
                    Some(Parameter::StringParam("output_param_value2".to_string())),
                ),
            ]),
        }],
        hash: "9ed1515819dec61fd361d5fdabb57f41ecce1a5fe1fe263b98c0d6943b9b232e".to_string(),
    });

    #[test]
    fn it_parsed_a_model_infer_request() {
        let input = ProcessedInput::from_infer_request(ModelInferRequest {
            model_name: "test".to_string(),
            model_version: "v1".to_string(),
            id: "999".to_string(),
            parameters: HashMap::from([(
                "param1".to_string(),
                InferParameter {
                    parameter_choice: Some(ParameterChoice::StringParam("hoi".to_string())),
                },
            )]),
            inputs: vec![InferInputTensor {
                name: "img".to_string(),
                datatype: "FP32".to_string(),
                shape: vec![1, 2, 3],
                parameters: HashMap::from([(
                    "input_param1".to_string(),
                    InferParameter {
                        parameter_choice: Some(ParameterChoice::StringParam("hoi".to_string())),
                    },
                )]),
                contents: None,
            }],
            outputs: vec![InferRequestedOutputTensor {
                name: "output1".to_string(),
                parameters: HashMap::from([(
                    "output_param1".to_string(),
                    InferParameter {
                        parameter_choice: Some(ParameterChoice::StringParam("hoi".to_string())),
                    },
                )]),
            }],
            raw_input_contents: vec![vec![255, 128, 1]],
        });

        assert_eq!(input.model_name, "test");
        assert_eq!(input.model_version, "v1");
        assert_eq!(input.id, "999");

        // TODO add more asserts
    }

    #[test]
    fn it_matches_equal_inputs() {
        let input1 = BASE_INPUT.clone();
        let input2 = BASE_INPUT.clone();

        assert!(input1.matches(&input2, Default::default()));
    }

    #[test]
    fn it_not_matches_different_model_name() {
        let input1 = BASE_INPUT.clone();
        let mut input2 = BASE_INPUT.clone();

        input2.model_name = "hoi".to_string();

        assert!(!input1.matches(&input2, Default::default()));
    }

    #[test]
    fn it_not_matches_different_model_version() {
        let input1 = BASE_INPUT.clone();
        let mut input2 = BASE_INPUT.clone();

        input2.model_version = "19".to_string();

        assert!(!input1.matches(&input2, Default::default()));
    }

    #[test]
    fn it_not_matches_different_parameters() {
        let mut input1 = BASE_INPUT.clone();
        let mut input2 = BASE_INPUT.clone();

        input2.parameters.insert(
            "test".to_string(),
            Some(Parameter::StringParam("test2".to_string())),
        );

        assert!(!input1.matches(&input2, Default::default()));
    }

    #[test]
    fn it_excludes_provided_parameters() {
        let mut input1 = BASE_INPUT.clone();
        let mut input2 = BASE_INPUT.clone();

        input1.parameters.insert(
            "ignore_me".to_string(),
            Some(Parameter::StringParam("1".to_string())),
        );
        input2.parameters.insert(
            "ignore_me".to_string(),
            Some(Parameter::StringParam("2".to_string())),
        );

        assert!(input1.matches(
            &input2,
            MatchConfig {
                parameter_keys: vec!["ignore_me".to_string()],
                ..Default::default()
            }
        ));
    }

    #[test]
    fn it_includes_provided_parameters() {
        let mut input1 = BASE_INPUT.clone();
        let mut input2 = BASE_INPUT.clone();

        input1.parameters.insert(
            "ignore_me".to_string(),
            Some(Parameter::StringParam("1".to_string())),
        );
        input2.parameters.insert(
            "ignore_me".to_string(),
            Some(Parameter::StringParam("2".to_string())),
        );

        assert!(input1.matches(
            &input2,
            MatchConfig {
                parameter_keys: vec!["test".to_string()],
                exclude_parameters: false,
                ..Default::default()
            }
        ));
    }

    #[test]
    fn it_not_matches_different_input_parameters() {
        let mut input1 = BASE_INPUT.clone();
        let mut input2 = BASE_INPUT.clone();

        input2.inputs[0].parameters.insert(
            "test".to_string(),
            Some(Parameter::StringParam("test2".to_string())),
        );

        assert!(!input1.matches(&input2, Default::default()));
    }

    #[test]
    fn it_excludes_provided_input_parameters() {
        let mut input1 = BASE_INPUT.clone();
        let mut input2 = BASE_INPUT.clone();

        input1.inputs[0].parameters.insert(
            "ignore_me".to_string(),
            Some(Parameter::StringParam("1".to_string())),
        );
        input2.inputs[0].parameters.insert(
            "ignore_me".to_string(),
            Some(Parameter::StringParam("2".to_string())),
        );

        assert!(input1.matches(
            &input2,
            MatchConfig {
                input_parameter_keys: HashMap::from([(
                    "input1".to_string(),
                    vec!["ignore_me".to_string()]
                ),]),
                ..Default::default()
            }
        ));
    }

    #[test]
    fn it_includes_provided_input_parameters() {
        let mut input1 = BASE_INPUT.clone();
        let mut input2 = BASE_INPUT.clone();

        input1.inputs[0].parameters.insert(
            "ignore_me".to_string(),
            Some(Parameter::StringParam("1".to_string())),
        );
        input2.inputs[0].parameters.insert(
            "ignore_me".to_string(),
            Some(Parameter::StringParam("2".to_string())),
        );

        assert!(input1.matches(
            &input2,
            MatchConfig {
                input_parameter_keys: HashMap::from([(
                    "input1".to_string(),
                    vec!["test".to_string()]
                ),]),
                exclude_input_parameters: false,
                ..Default::default()
            }
        ));
    }

    #[test]
    fn it_not_matches_different_output_parameters() {
        let mut input1 = BASE_INPUT.clone();
        let mut input2 = BASE_INPUT.clone();

        input2.outputs[0].parameters.insert(
            "test".to_string(),
            Some(Parameter::StringParam("test2".to_string())),
        );

        assert!(!input1.matches(&input2, Default::default()));
    }

    #[test]
    fn it_excludes_provided_output_parameters() {
        let mut input1 = BASE_INPUT.clone();
        let mut input2 = BASE_INPUT.clone();

        input1.outputs[0].parameters.insert(
            "ignore_me".to_string(),
            Some(Parameter::StringParam("1".to_string())),
        );
        input2.outputs[0].parameters.insert(
            "ignore_me".to_string(),
            Some(Parameter::StringParam("2".to_string())),
        );

        assert!(input1.matches(
            &input2,
            MatchConfig {
                output_parameter_keys: HashMap::from([(
                    "output1".to_string(),
                    vec!["ignore_me".to_string()]
                ),]),
                ..Default::default()
            }
        ));
    }

    #[test]
    fn it_includes_provided_output_parameters() {
        let mut input1 = BASE_INPUT.clone();
        let mut input2 = BASE_INPUT.clone();

        input1.outputs[0].parameters.insert(
            "ignore_me".to_string(),
            Some(Parameter::StringParam("1".to_string())),
        );
        input2.outputs[0].parameters.insert(
            "ignore_me".to_string(),
            Some(Parameter::StringParam("2".to_string())),
        );

        assert!(input1.matches(
            &input2,
            MatchConfig {
                output_parameter_keys: HashMap::from([(
                    "input1".to_string(),
                    vec!["test".to_string()]
                ),]),
                exclude_output_parameters: false,
                ..Default::default()
            }
        ));
    }

    #[test]
    fn it_not_matches_different_input_name() {
        let mut input1 = BASE_INPUT.clone();
        let mut input2 = BASE_INPUT.clone();

        input2.inputs[0].name = "asdf".to_string();

        assert!(!input1.matches(
            &input2,
            MatchConfig {
                ..Default::default()
            }
        ));
    }

    #[test]
    fn it_not_matches_different_input_shape() {
        let mut input1 = BASE_INPUT.clone();
        let mut input2 = BASE_INPUT.clone();

        input2.inputs[0].shape = vec![3, 2, 1];

        assert!(!input1.matches(
            &input2,
            MatchConfig {
                ..Default::default()
            }
        ));
    }

    #[test]
    fn it_not_matches_different_input_datatype() {
        let mut input1 = BASE_INPUT.clone();
        let mut input2 = BASE_INPUT.clone();

        input2.inputs[0].datatype = "FP32".to_string();

        assert!(!input1.matches(
            &input2,
            MatchConfig {
                ..Default::default()
            }
        ));
    }

    #[test]
    fn it_not_matches_different_output_name() {
        let mut input1 = BASE_INPUT.clone();
        let mut input2 = BASE_INPUT.clone();

        input2.outputs[0].name = "asdf".to_string();

        assert!(!input1.matches(
            &input2,
            MatchConfig {
                ..Default::default()
            }
        ));
    }
}
