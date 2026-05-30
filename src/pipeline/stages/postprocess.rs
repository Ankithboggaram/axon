//! Transforms raw model output into a structured prediction.

use pipex::error::PipelineError;
use pipex::stage::Stage;

use crate::config::OutputType;
use crate::pipeline::InferenceScratchpad;

/// Transforms the first output tensor based on the configured output type.
///
/// - `Binary`: thresholds the score to 1.0 or -1.0
/// - `Probability`: passes the raw score through unchanged
/// - `Raw`: passes the raw model output through unchanged
#[derive(Debug)]
pub struct PostprocessStage {
    pub threshold: f32,
    pub output_type: OutputType,
}

impl Stage<InferenceScratchpad> for PostprocessStage {
    #[inline]
    fn run(&mut self, ctx: &mut InferenceScratchpad) -> Result<(), PipelineError> {
        if ctx.outputs.is_empty() {
            return Err(PipelineError::StageFailed(
                "postprocess: no outputs from infer stage".into(),
            ));
        }

        let output = &mut ctx.outputs[0];

        match self.output_type {
            OutputType::Binary => {
                // Binary thresholding is only meaningful for scalar outputs.
                if output.data.len() != 1 {
                    return Err(PipelineError::StageFailed(format!(
                        "postprocess: binary output type requires a scalar output, got shape {:?}",
                        output.data.shape()
                    )));
                }
                output
                    .data
                    .mapv_inplace(|v| if v > self.threshold { 1.0 } else { -1.0 });
            }
            OutputType::Probability | OutputType::Raw => {
                // Pass through unchanged - caller is responsible for interpreting the tensor.
            }
        }

        Ok(())
    }
}
