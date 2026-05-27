//! Backend trait and inference backend implementations.

use async_trait::async_trait;

use crate::types::{NamedTensor, NamedTensorRef};

pub mod packaging;

#[async_trait]
pub trait Backend: std::fmt::Debug + Send + Sync {
    /// Runs inference and writes output tensors into `outputs`.
    ///
    /// The caller must pass the scratchpad's pre-allocated outputs Vec.
    /// The backend clears it and pushes results in, reusing the existing
    /// Vec capacity across requests to avoid heap allocation.
    async fn run(
        &self,
        inputs: &[NamedTensorRef<'_>],
        outputs: &mut Vec<NamedTensor>,
    ) -> anyhow::Result<()>;
}
