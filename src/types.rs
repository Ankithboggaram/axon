//! Shared types used across modules.

use ndarray::{ArrayD, ArrayViewD};

/// A named n-dimensional tensor.
///
/// Data is stored as a dynamically-shaped f32 array. The name identifies
/// which model input or output this tensor corresponds to.
#[derive(Debug)]
pub struct NamedTensor {
    pub name: String,
    pub data: ArrayD<f32>,
}

/// A borrowed view of a named tensor, used on the inference hot path.
///
/// Holds references into existing data rather than owning a copy, so
/// passing inputs to the backend requires zero heap allocation.
#[derive(Debug)]
pub struct NamedTensorRef<'a> {
    pub name: &'a str,
    pub data: ArrayViewD<'a, f32>,
}
