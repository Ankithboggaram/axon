//! Normalises the input tensor by subtracting the mean and multiplying by inv_std.

use pipex::error::PipelineError;
use pipex::stage::Stage;

use crate::pipeline::InferenceScratchpad;

/// Applies zero-mean unit-variance normalisation to the input tensor.
///
/// Each element is transformed as `(x - mean) * inv_std`. Stores the
/// reciprocal of std rather than std itself so the hot path multiplies
/// instead of divides; division is 10-30x slower than multiplication
/// on modern CPUs. The wiring code computes `1.0 / std` once at startup.
#[derive(Debug)]
pub struct NormalizeStage {
    pub mean: f32,
    /// Reciprocal of std (1.0 / std), precomputed at wiring time.
    pub inv_std: f32,
}

impl Stage<InferenceScratchpad> for NormalizeStage {
    #[inline]
    fn run(&mut self, ctx: &mut InferenceScratchpad) -> Result<(), PipelineError> {
        ctx.input -= self.mean;
        ctx.input *= self.inv_std;
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
    fn subtracts_mean() {
        let mut stage = NormalizeStage {
            mean: 2.0,
            inv_std: 1.0,
        };
        let mut ctx = ctx(arr1(&[2.0f32, 4.0]).into_dyn());
        stage.run(&mut ctx).unwrap();
        assert_eq!(ctx.input.as_slice().unwrap(), &[0.0, 2.0]);
    }

    #[test]
    fn scales_by_inv_std() {
        let mut stage = NormalizeStage {
            mean: 0.0,
            inv_std: 2.0,
        };
        let mut ctx = ctx(arr1(&[1.0f32, 3.0]).into_dyn());
        stage.run(&mut ctx).unwrap();
        assert_eq!(ctx.input.as_slice().unwrap(), &[2.0, 6.0]);
    }

    #[test]
    fn combined_transform() {
        // (3.0 - 1.0) * 0.5 = 1.0
        let mut stage = NormalizeStage {
            mean: 1.0,
            inv_std: 0.5,
        };
        let mut ctx = ctx(arr1(&[3.0f32]).into_dyn());
        stage.run(&mut ctx).unwrap();
        assert!((ctx.input[[0]] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn zero_input_minus_mean_gives_neg_mean_times_inv_std() {
        let mut stage = NormalizeStage {
            mean: 4.0,
            inv_std: 0.25,
        };
        let mut ctx = ctx(arr1(&[0.0f32]).into_dyn());
        stage.run(&mut ctx).unwrap();
        // (0 - 4) * 0.25 = -1.0
        assert!((ctx.input[[0]] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn always_returns_ok() {
        let mut stage = NormalizeStage {
            mean: 0.0,
            inv_std: 1.0,
        };
        let mut ctx = ctx(arr1(&[1.0f32]).into_dyn());
        assert!(stage.run(&mut ctx).is_ok());
    }
}
