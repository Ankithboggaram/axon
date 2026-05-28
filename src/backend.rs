//! Backend trait and inference backend implementations.

use async_trait::async_trait;

use crate::types::{NamedTensorRef, OutputBuffer};

pub mod onnx;
pub mod packaging;

#[async_trait]
pub trait Backend: std::fmt::Debug + Send + Sync {
    /// Runs inference and writes results into the scratchpad's pre-allocated output buffers.
    ///
    /// The backend must call `assign()` on each `OutputBuffer::data` rather than
    /// replacing it, so that the pre-allocated memory is reused across requests.
    /// The number of outputs is fixed by model_schema at startup and must match
    /// the length of the `outputs` slice.
    async fn run(
        &self,
        inputs: &[NamedTensorRef<'_>],
        outputs: &mut [OutputBuffer],
    ) -> anyhow::Result<()>;
}
