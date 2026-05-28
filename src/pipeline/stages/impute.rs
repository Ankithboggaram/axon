//! Replaces NaN values in the input tensor with a configured default.

use pipex::error::PipelineError;
use pipex::stage::Stage;

use crate::pipeline::InferenceScratchpad;

/// Replaces every NaN element of the input tensor with `default_value`.
///
/// Runs before normalisation so that missing values do not propagate
/// as NaN through the rest of the pipeline.
#[derive(Debug)]
pub struct ImputeStage {
    pub default_value: f32,
}

impl Stage<InferenceScratchpad> for ImputeStage {
    #[inline]
    fn run(&mut self, ctx: &mut InferenceScratchpad) -> Result<(), PipelineError> {
        ctx.input
            .mapv_inplace(|v| if v.is_nan() { self.default_value } else { v });
        Ok(())
    }
}
