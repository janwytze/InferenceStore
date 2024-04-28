use crate::input_parsing::{MatchConfig, ProcessedInput};
use crate::output_parsing::ProcessedOutput;

pub trait ProcessedInputOutputVecExt {
    fn find_output(&self, input: &ProcessedInput, config: MatchConfig) -> Option<&ProcessedOutput>;
}

impl ProcessedInputOutputVecExt for Vec<(ProcessedInput, ProcessedOutput)> {
    fn find_output(
        &self,
        match_input: &ProcessedInput,
        config: MatchConfig,
    ) -> Option<&ProcessedOutput> {
        for (input, output) in self {
            if match_input.matches(input, config.clone()) {
                return Some(output);
            }
        }

        None
    }
}
