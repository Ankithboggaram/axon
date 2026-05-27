//! Validates the input tensor shape, checking for NaN and infinite values.

use pipex::error::PipelineError;
use pipex::stage::Stage;

use crate::pipeline::InferenceScratchpad;

/// Checks input tensor shape, NaN, and infinite values.
///
/// Fails if the actual shape does not match `expected_shape`, or if any
/// element is NaN or infinite. If an impute stage is present, place it
/// before this stage to replace NaN values rather than reject them.
#[derive(Debug)]
pub struct ValidateStage {
    pub expected_shape: Vec<i64>,
}

impl Stage<InferenceScratchpad> for ValidateStage {
    fn run(&mut self, ctx: &mut InferenceScratchpad) -> Result<(), PipelineError> {
        // Check shape.
        let actual_shape: Vec<i64> = ctx.input.shape().iter().map(|&d| d as i64).collect();
        if actual_shape != self.expected_shape {
            return Err(PipelineError::StageFailed(format!(
                "validate: expected shape {:?}, got {:?}",
                self.expected_shape, actual_shape
            )));
        }

        // Check for NaN and infinite values.
        for &val in ctx.input.iter() {
            if val.is_nan() {
                return Err(PipelineError::StageFailed(
                    "validate: input contains NaN".into(),
                ));
            }
            if val.is_infinite() {
                return Err(PipelineError::StageFailed(
                    "validate: input contains infinite value".into(),
                ));
            }
        }

        Ok(())
    }
}
