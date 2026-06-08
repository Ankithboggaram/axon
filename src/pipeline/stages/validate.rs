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

// Separated from run() so #[cold] applies to the entire error-construction path,
// giving the branch predictor a stronger hint that the hot path never branches here.
#[cold]
#[inline(never)]
fn non_finite_err(val: f32) -> PipelineError {
    let msg = if val.is_nan() {
        "validate: input contains NaN"
    } else {
        "validate: input contains infinite value"
    };
    PipelineError::StageFailed {
        stage: "ValidateStage",
        message: msg.into(),
    }
}

impl Stage<InferenceScratchpad> for ValidateStage {
    #[inline]
    fn run(&mut self, ctx: &mut InferenceScratchpad) -> Result<(), PipelineError> {
        if ctx.input.shape() != self.expected_shape.as_ref() {
            return Err(PipelineError::StageFailed {
                stage: "ValidateStage",
                message: format!(
                    "validate: expected shape {:?}, got {:?}",
                    self.expected_shape,
                    ctx.input.shape(),
                ),
            });
        }

        // is_finite() is one CPU instruction covering both NaN and infinite.
        // Only branch into the specific check on the (rare) failure path.
        for &val in ctx.input.iter() {
            if !val.is_finite() {
                return Err(non_finite_err(val));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use arrayvec::ArrayString;
    use ndarray::arr1;
    use pipex::stage::Stage;
    use proptest::prelude::*;

    use super::*;
    use crate::pipeline::InferenceScratchpad;

    fn ctx(input: ndarray::ArrayD<f32>) -> InferenceScratchpad {
        InferenceScratchpad {
            entity_id: ArrayString::new(),
            timestamp_ms: 0,
            input,
            outputs: Box::new([]),
        }
    }

    #[test]
    fn correct_shape_passes() {
        let mut stage = ValidateStage {
            expected_shape: Box::new([3]),
        };
        let mut ctx = ctx(arr1(&[1.0f32, 2.0, 3.0]).into_dyn());
        assert!(stage.run(&mut ctx).is_ok());
    }

    #[test]
    fn wrong_shape_fails() {
        let mut stage = ValidateStage {
            expected_shape: Box::new([4]),
        };
        let mut ctx = ctx(arr1(&[1.0f32, 2.0, 3.0]).into_dyn());
        assert!(stage.run(&mut ctx).is_err());
    }

    #[test]
    fn nan_fails() {
        let mut stage = ValidateStage {
            expected_shape: Box::new([2]),
        };
        let mut ctx = ctx(arr1(&[1.0f32, f32::NAN]).into_dyn());
        assert!(stage.run(&mut ctx).is_err());
    }

    #[test]
    fn infinite_fails() {
        let mut stage = ValidateStage {
            expected_shape: Box::new([1]),
        };
        let mut ctx = ctx(arr1(&[f32::INFINITY]).into_dyn());
        assert!(stage.run(&mut ctx).is_err());
    }

    #[test]
    fn neg_infinite_fails() {
        let mut stage = ValidateStage {
            expected_shape: Box::new([1]),
        };
        let mut ctx = ctx(arr1(&[f32::NEG_INFINITY]).into_dyn());
        assert!(stage.run(&mut ctx).is_err());
    }

    #[test]
    fn all_finite_passes() {
        let mut stage = ValidateStage {
            expected_shape: Box::new([4]),
        };
        let mut ctx = ctx(arr1(&[-1.0f32, 0.0, 0.5, 1e6]).into_dyn());
        assert!(stage.run(&mut ctx).is_ok());
    }

    proptest! {
        #[test]
        fn validate_accepts_any_finite_correct_shape(
            values in proptest::collection::vec(-1e6f32..1e6f32, 1..50usize),
        ) {
            let n = values.len();
            let mut stage = ValidateStage {
                expected_shape: Box::new([n]),
            };
            let mut ctx = ctx(ndarray::arr1(&values).into_dyn());
            prop_assert!(stage.run(&mut ctx).is_ok());
        }

        #[test]
        fn validate_rejects_any_wrong_shape(
            values in proptest::collection::vec(-1e6f32..1e6f32, 1..50usize),
            extra in 1usize..10usize,
        ) {
            let n = values.len();
            let mut stage = ValidateStage {
                expected_shape: Box::new([n + extra]),
            };
            let mut ctx = ctx(ndarray::arr1(&values).into_dyn());
            prop_assert!(stage.run(&mut ctx).is_err());
        }
    }
}
