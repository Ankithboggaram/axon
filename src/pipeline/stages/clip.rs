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

    use super::*;
    use crate::pipeline::InferenceScratchpad;

    fn ctx(input: ndarray::ArrayD<f32>) -> InferenceScratchpad {
        InferenceScratchpad {
            entity_id: ArrayString::new(),
            request_id: ArrayString::new(),
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
}
