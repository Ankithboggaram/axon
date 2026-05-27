//! Clamps input tensor values to a configured [min, max] range.

use pipex::error::PipelineError;
use pipex::stage::Stage;

use crate::pipeline::InferenceScratchpad;

/// Clamps every element of the input tensor to the range `[min, max]`.
///
/// Runs before normalisation to prevent outliers from producing extreme
/// normalised values that would corrupt model predictions.
#[derive(Debug)]
pub struct ClipStage {
    pub min: f32,
    pub max: f32,
}

impl Stage<InferenceScratchpad> for ClipStage {
    fn run(&mut self, ctx: &mut InferenceScratchpad) -> Result<(), PipelineError> {
        ctx.input.mapv_inplace(|v| v.clamp(self.min, self.max));
        Ok(())
    }
}
