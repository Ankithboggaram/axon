//! Backend trait and inference backend implementations.

use crate::types::Tensor;
use async_trait::async_trait;

pub mod packaging;

#[async_trait]
pub trait Backend {
    async fn run(&self, inputs: &[Tensor]) -> anyhow::Result<Vec<Tensor>>;
}
