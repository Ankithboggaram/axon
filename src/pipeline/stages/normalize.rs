//! Normalises the input tensor by subtracting the mean and multiplying by inv_std.

use pipex::error::PipelineError;
use pipex::stage::Stage;

use crate::pipeline::InferenceScratchpad;

/// Applies zero-mean unit-variance normalisation to the input tensor.
///
/// Each element is transformed as `(x - mean) * inv_std`. Stores the
/// reciprocal of std rather than std itself so the hot path multiplies
/// instead of divides — division is 10-30x slower than multiplication
/// on modern CPUs. The wiring code computes `1.0 / std` once at startup.
#[derive(Debug)]
pub struct NormalizeStage {
    pub mean: f32,
    /// Reciprocal of std (1.0 / std), precomputed at wiring time.
    pub inv_std: f32,
}

impl Stage<InferenceScratchpad> for NormalizeStage {
    fn run(&mut self, ctx: &mut InferenceScratchpad) -> Result<(), PipelineError> {
        ctx.input -= self.mean;
        ctx.input *= self.inv_std;
        Ok(())
    }
}
