//! Shared types used across modules.

use arrayvec::ArrayString;
use ndarray::{ArrayD, ArrayViewD};

/// Maximum byte length for output tensor names.
pub const MAX_TENSOR_NAME_LEN: usize = 64;

/// A borrowed view of a named tensor, used for backend inputs on the hot path.
///
/// Holds references into existing data rather than owning a copy, so
/// passing inputs to the backend requires zero heap allocation.
#[derive(Debug)]
pub struct NamedTensorRef<'a> {
    /// Tensor name, borrowed from the scratchpad.
    pub name: &'a str,
    /// Tensor values as a borrowed view into the scratchpad buffer.
    pub data: ArrayViewD<'a, f32>,
}

/// A pre-allocated output tensor buffer, owned by the scratchpad.
///
/// Name is set once at startup from model_schema.outputs. Data is
/// pre-allocated to the expected output shape. The backend writes into
/// this buffer each request via assign(), with no heap allocation.
#[derive(Clone, Debug)]
pub struct OutputBuffer {
    /// Name of this output tensor, set once at startup.
    pub name: ArrayString<MAX_TENSOR_NAME_LEN>,
    /// Pre-allocated data buffer, written in place by the backend each request.
    pub data: ArrayD<f32>,
}
