//! Shared types used across modules.

use ndarray::ArrayD;

/// A named n-dimensional tensor.
///
/// Data is stored as a dynamically-shaped f32 array. The name identifies
/// which model input or output this tensor corresponds to.
#[derive(Debug)]
pub struct NamedTensor {
    pub name: String,
    pub data: ArrayD<f32>,
}
