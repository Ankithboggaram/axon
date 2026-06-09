#![no_main]

use arbitrary::Arbitrary;
use arrayvec::ArrayString;
use axon::pipeline::InferenceScratchpad;
use axon::pipeline::stages::clip::ClipStage;
use axon::pipeline::stages::validate::ValidateStage;
use libfuzzer_sys::fuzz_target;
use ndarray::{ArrayD, IxDyn};
use pipexec::stage::Stage;

#[derive(Arbitrary, Debug)]
struct Inputs {
    /// Raw shape dimensions; each is converted to a positive usize.
    shape_dims: Vec<u8>,
    /// Raw float values to fill the tensor; resized to match the shape product.
    values: Vec<f32>,
    /// Clip bounds; target guards against min >= max before calling run().
    clip_min: f32,
    clip_max: f32,
}

fuzz_target!(|data: Inputs| {
    // Build a non-empty shape. Clamp each dim to [1, 32] to avoid huge allocations.
    let shape: Vec<usize> = if data.shape_dims.is_empty() {
        vec![1]
    } else {
        data.shape_dims
            .iter()
            .map(|&d| (d as usize % 32).max(1))
            .collect()
    };

    let n: usize = shape.iter().product();
    // Guard against shapes that produce unreasonably large tensors.
    if n > 4096 {
        return;
    }

    // Pad or truncate values to match the tensor size.
    let mut values = data.values.clone();
    values.resize(n, 0.0);

    let input = match ArrayD::from_shape_vec(IxDyn(&shape), values) {
        Ok(a) => a,
        Err(_) => return,
    };

    let mut ctx = InferenceScratchpad {
        entity_id: ArrayString::new(),
        timestamp_ms: 0,
        input: input.clone(),
        outputs: Box::new([]),
    };

    // ValidateStage: should never panic. Returns Err for shape mismatch or
    // non-finite values; that is the intended behaviour.
    let mut validate = ValidateStage {
        expected_shape: shape.into_boxed_slice(),
    };
    let _ = validate.run(&mut ctx);

    // ClipStage: f32::clamp panics if min > max or if either is NaN.
    // Guard here because the stage relies on build() to enforce this invariant;
    // direct construction (as in tests and fuzz targets) bypasses that check.
    if data.clip_min < data.clip_max
        && data.clip_min.is_finite()
        && data.clip_max.is_finite()
    {
        ctx.input = input;
        let mut clip = ClipStage {
            min: data.clip_min,
            max: data.clip_max,
        };
        let _ = clip.run(&mut ctx);
    }
});
