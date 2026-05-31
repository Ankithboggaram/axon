//! Backend trait and inference backend implementations.

use async_trait::async_trait;

use crate::types::{NamedTensorRef, OutputBuffer};

pub mod onnx;
pub mod packaging;

#[async_trait]
pub trait Backend: std::fmt::Debug + Send + Sync {
    /// Runs model inference on the given named input tensors and writes
    /// results into the pre-allocated output buffers in place.
    async fn run(
        &self,
        inputs: &[NamedTensorRef<'_>],
        outputs: &mut [OutputBuffer],
    ) -> anyhow::Result<()>;
}
