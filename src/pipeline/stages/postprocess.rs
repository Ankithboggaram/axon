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
    /// Decision boundary for binary classification.
    pub threshold: f32,
    /// How to interpret and transform the raw model score.
    pub output_type: OutputType,
}

impl Stage<InferenceScratchpad> for PostprocessStage {
    #[inline]
    fn run(&mut self, ctx: &mut InferenceScratchpad) -> Result<(), PipelineError> {
        if ctx.outputs.is_empty() {
            return Err(PipelineError::StageFailed {
                stage: "PostprocessStage",
                message: "postprocess: no outputs from infer stage".into(),
            });
        }

        let output = &mut ctx.outputs[0];

        match self.output_type {
            OutputType::Binary => {
                // Binary thresholding is only meaningful for scalar outputs.
                if output.data.len() != 1 {
                    return Err(PipelineError::StageFailed {
                        stage: "PostprocessStage",
                        message: format!(
                            "postprocess: binary output type requires a scalar output, got shape {:?}",
                            output.data.shape()
                        ),
                    });
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

#[cfg(test)]
mod tests {
    use arrayvec::ArrayString;
    use ndarray::{ArrayD, IxDyn, arr1};
    use pipex::stage::Stage;

    use super::*;
    use crate::pipeline::InferenceScratchpad;
    use crate::types::OutputBuffer;

    fn ctx_with_output(data: ArrayD<f32>) -> InferenceScratchpad {
        InferenceScratchpad {
            entity_id: ArrayString::new(),
            timestamp_ms: 0,
            input: ArrayD::zeros(IxDyn(&[1])),
            outputs: vec![OutputBuffer {
                name: "output".parse().unwrap(),
                data,
            }]
            .into_boxed_slice(),
        }
    }

    fn ctx_no_outputs() -> InferenceScratchpad {
        InferenceScratchpad {
            entity_id: ArrayString::new(),
            timestamp_ms: 0,
            input: ArrayD::zeros(IxDyn(&[1])),
            outputs: Box::new([]),
        }
    }

    #[test]
    fn binary_above_threshold_gives_1() {
        let mut stage = PostprocessStage {
            threshold: 0.5,
            output_type: OutputType::Binary,
        };
        let mut ctx = ctx_with_output(ArrayD::from_elem(IxDyn(&[1]), 0.8f32));
        stage.run(&mut ctx).unwrap();
        assert_eq!(ctx.outputs[0].data[[0]], 1.0);
    }

    #[test]
    fn binary_at_threshold_gives_neg1() {
        // threshold comparison is strict (v > threshold), so equal → -1.0
        let mut stage = PostprocessStage {
            threshold: 0.5,
            output_type: OutputType::Binary,
        };
        let mut ctx = ctx_with_output(ArrayD::from_elem(IxDyn(&[1]), 0.5f32));
        stage.run(&mut ctx).unwrap();
        assert_eq!(ctx.outputs[0].data[[0]], -1.0);
    }

    #[test]
    fn binary_below_threshold_gives_neg1() {
        let mut stage = PostprocessStage {
            threshold: 0.5,
            output_type: OutputType::Binary,
        };
        let mut ctx = ctx_with_output(ArrayD::from_elem(IxDyn(&[1]), 0.2f32));
        stage.run(&mut ctx).unwrap();
        assert_eq!(ctx.outputs[0].data[[0]], -1.0);
    }

    #[test]
    fn probability_passthrough() {
        let mut stage = PostprocessStage {
            threshold: 0.5,
            output_type: OutputType::Probability,
        };
        let mut ctx = ctx_with_output(arr1(&[0.7f32, 0.3]).into_dyn());
        stage.run(&mut ctx).unwrap();
        assert_eq!(ctx.outputs[0].data.as_slice().unwrap(), &[0.7, 0.3]);
    }

    #[test]
    fn raw_passthrough() {
        let mut stage = PostprocessStage {
            threshold: 0.5,
            output_type: OutputType::Raw,
        };
        let mut ctx = ctx_with_output(arr1(&[1.5f32, -2.0]).into_dyn());
        stage.run(&mut ctx).unwrap();
        assert_eq!(ctx.outputs[0].data.as_slice().unwrap(), &[1.5, -2.0]);
    }

    #[test]
    fn empty_outputs_fails() {
        let mut stage = PostprocessStage {
            threshold: 0.5,
            output_type: OutputType::Binary,
        };
        assert!(stage.run(&mut ctx_no_outputs()).is_err());
    }

    #[test]
    fn binary_non_scalar_fails() {
        let mut stage = PostprocessStage {
            threshold: 0.5,
            output_type: OutputType::Binary,
        };
        let mut ctx = ctx_with_output(arr1(&[0.6f32, 0.4]).into_dyn());
        assert!(stage.run(&mut ctx).is_err());
    }
}
