//! Shared types used across modules.

/// A tensor with a flat data buffer and its shape.
///
/// Data is stored in row-major order. The shape describes how to interpret it.
pub struct Tensor {
    pub name: String,
    pub data: Vec<f32>,
    pub shape: Vec<i64>,
}
