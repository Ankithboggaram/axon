//! ModelRegistryClient trait and model registry implementations.

use async_trait::async_trait;

pub struct RegisteredModel {
    pub name: String,
    pub version: String,
    pub artifact_uri: String,
    pub deployable_artifact_uri: String,
}

#[async_trait]
pub trait ModelRegistryClient {
    async fn fetch_model(&self, name: &str, version: &str) -> anyhow::Result<RegisteredModel>;
}
