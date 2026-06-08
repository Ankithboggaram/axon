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
    /// Value substituted for each NaN element.
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
    fn non_nan_values_unchanged() {
        let mut stage = ImputeStage { default_value: 0.0 };
        let mut ctx = ctx(arr1(&[1.0f32, 2.0, 3.0]).into_dyn());
        stage.run(&mut ctx).unwrap();
        assert_eq!(ctx.input.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn nan_replaced_with_default() {
        let mut stage = ImputeStage {
            default_value: -1.0,
        };
        let mut ctx = ctx(arr1(&[f32::NAN]).into_dyn());
        stage.run(&mut ctx).unwrap();
        assert_eq!(ctx.input[[0]], -1.0);
    }

    #[test]
    fn only_nan_values_replaced() {
        let mut stage = ImputeStage {
            default_value: 99.0,
        };
        let mut ctx = ctx(arr1(&[1.0f32, f32::NAN, 3.0]).into_dyn());
        stage.run(&mut ctx).unwrap();
        let s = ctx.input.as_slice().unwrap();
        assert_eq!(s[0], 1.0);
        assert_eq!(s[1], 99.0);
        assert_eq!(s[2], 3.0);
    }

    #[test]
    fn always_returns_ok() {
        let mut stage = ImputeStage { default_value: 0.0 };
        let mut ctx = ctx(arr1(&[f32::NAN, f32::NAN]).into_dyn());
        assert!(stage.run(&mut ctx).is_ok());
    }

    proptest! {
        #[test]
        fn impute_removes_all_nans(
            values in proptest::collection::vec(proptest::num::f32::ANY, 1..50usize),
            default_value in -1e6f32..1e6f32,
        ) {
            let mut stage = ImputeStage { default_value };
            let mut ctx = ctx(ndarray::arr1(&values).into_dyn());
            stage.run(&mut ctx).unwrap();
            for &v in ctx.input.iter() {
                prop_assert!(!v.is_nan(), "NaN survived imputation");
            }
        }

        #[test]
        fn impute_preserves_non_nan_values(
            values in proptest::collection::vec(-1e6f32..1e6f32, 1..50usize),
            default_value in proptest::num::f32::ANY,
        ) {
            let original = values.clone();
            let mut stage = ImputeStage { default_value };
            let mut ctx = ctx(ndarray::arr1(&values).into_dyn());
            stage.run(&mut ctx).unwrap();
            for (i, &v) in ctx.input.iter().enumerate() {
                prop_assert_eq!(v, original[i]);
            }
        }
    }
}
