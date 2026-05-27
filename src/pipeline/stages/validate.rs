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
    /// Expected shape stored as usize to match ndarray's shape() directly,
    /// avoiding a cast and a Vec allocation on every request.
    pub expected_shape: Box<[usize]>,
}

impl Stage<InferenceScratchpad> for ValidateStage {
    fn run(&mut self, ctx: &mut InferenceScratchpad) -> Result<(), PipelineError> {
        // Direct slice comparison — no allocation, no cast.
        if ctx.input.shape() != self.expected_shape.as_ref() {
            return Err(PipelineError::StageFailed(format!(
                "validate: expected shape {:?}, got {:?}",
                self.expected_shape,
                ctx.input.shape(),
            )));
        }

        // is_finite() is one CPU instruction covering both NaN and infinite.
        // Only branch into the specific check on the (rare) failure path.
        for &val in ctx.input.iter() {
            if !val.is_finite() {
                return Err(PipelineError::StageFailed(
                    if val.is_nan() {
                        "validate: input contains NaN"
                    } else {
                        "validate: input contains infinite value"
                    }
                    .into(),
                ));
            }
        }

        Ok(())
    }
}
