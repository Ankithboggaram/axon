//! Backend trait and inference backend implementations.

use async_trait::async_trait;

use crate::types::NamedTensor;

pub mod packaging;

#[async_trait]
pub trait Backend {
    async fn run(&self, inputs: &[NamedTensor]) -> anyhow::Result<Vec<NamedTensor>>;
}
