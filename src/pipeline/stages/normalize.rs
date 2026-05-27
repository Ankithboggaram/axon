//! Normalises the input tensor by subtracting the mean and dividing by std.

use pipex::error::PipelineError;
use pipex::stage::Stage;

use crate::pipeline::InferenceScratchpad;

/// Applies zero-mean unit-variance normalisation to the input tensor.
///
/// Each element is transformed as `(x - mean) / std`. Uses ndarray's
/// vectorised scalar arithmetic rather than an element-wise loop.
#[derive(Debug)]
pub struct NormalizeStage {
    pub mean: f32,
    pub std: f32,
}

impl Stage<InferenceScratchpad> for NormalizeStage {
    fn run(&mut self, ctx: &mut InferenceScratchpad) -> Result<(), PipelineError> {
        ctx.input -= self.mean;
        ctx.input /= self.std;
        Ok(())
    }
}
