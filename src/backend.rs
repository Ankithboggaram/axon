//! Backend trait and inference backend implementations.

use async_trait::async_trait;

use crate::types::{NamedTensor, NamedTensorRef};

pub mod packaging;

#[async_trait]
pub trait Backend: std::fmt::Debug + Send + Sync {
    async fn run(&self, inputs: &[NamedTensorRef<'_>]) -> anyhow::Result<Vec<NamedTensor>>;
}
