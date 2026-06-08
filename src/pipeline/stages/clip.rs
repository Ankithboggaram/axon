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
    /// Lower bound; values below this are clamped to min.
    pub min: f32,
    /// Upper bound; values above this are clamped to max.
    pub max: f32,
}

impl Stage<InferenceScratchpad> for ClipStage {
    #[inline]
    fn run(&mut self, ctx: &mut InferenceScratchpad) -> Result<(), PipelineError> {
        ctx.input.mapv_inplace(|v| v.clamp(self.min, self.max));
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
    fn values_within_range_unchanged() {
        let mut stage = ClipStage {
            min: -1.0,
            max: 1.0,
        };
        let mut ctx = ctx(arr1(&[0.0f32, 0.5, -0.5]).into_dyn());
        stage.run(&mut ctx).unwrap();
        assert_eq!(ctx.input.as_slice().unwrap(), &[0.0, 0.5, -0.5]);
    }

    #[test]
    fn values_below_min_clamped() {
        let mut stage = ClipStage {
            min: -1.0,
            max: 1.0,
        };
        let mut ctx = ctx(arr1(&[-5.0f32]).into_dyn());
        stage.run(&mut ctx).unwrap();
        assert_eq!(ctx.input[[0]], -1.0);
    }

    #[test]
    fn values_above_max_clamped() {
        let mut stage = ClipStage {
            min: -1.0,
            max: 1.0,
        };
        let mut ctx = ctx(arr1(&[5.0f32]).into_dyn());
        stage.run(&mut ctx).unwrap();
        assert_eq!(ctx.input[[0]], 1.0);
    }

    #[test]
    fn boundary_values_unchanged() {
        let mut stage = ClipStage {
            min: -1.0,
            max: 1.0,
        };
        let mut ctx = ctx(arr1(&[-1.0f32, 1.0]).into_dyn());
        stage.run(&mut ctx).unwrap();
        assert_eq!(ctx.input.as_slice().unwrap(), &[-1.0, 1.0]);
    }

    #[test]
    fn always_returns_ok() {
        let mut stage = ClipStage { min: 0.0, max: 1.0 };
        let mut ctx = ctx(arr1(&[-999.0f32, 999.0]).into_dyn());
        assert!(stage.run(&mut ctx).is_ok());
    }

    proptest! {
        #[test]
        fn clip_output_always_in_range(
            values in proptest::collection::vec(-1e6f32..1e6f32, 1..50usize),
            min in -1e6f32..0.0f32,
            delta in 1e-3f32..2e6f32,
        ) {
            let max = min + delta;
            let mut stage = ClipStage { min, max };
            let mut ctx = ctx(ndarray::arr1(&values).into_dyn());
            stage.run(&mut ctx).unwrap();
            for &v in ctx.input.iter() {
                prop_assert!(v >= min && v <= max, "value {} outside [{}, {}]", v, min, max);
            }
        }

        #[test]
        fn clip_in_range_elements_unchanged(
            min in -1e6f32..0.0f32,
            delta in 1e-3f32..2e6f32,
            t in 0.01f32..0.99f32,
        ) {
            let max = min + delta;
            let val = min + t * delta;
            let mut stage = ClipStage { min, max };
            let mut ctx = ctx(ndarray::arr1(&[val]).into_dyn());
            stage.run(&mut ctx).unwrap();
            prop_assert_eq!(ctx.input[[0]], val);
        }
    }
}
