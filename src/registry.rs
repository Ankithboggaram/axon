//! ModelRegistryClient trait and model registry implementations.

use async_trait::async_trait;

/// A model artifact fetched from the registry.
#[derive(Debug)]
pub struct RegisteredModel {
    pub name: String,
    pub version: String,
    /// URI to the raw model file (e.g. ONNX weights), used for validation.
    pub artifact_uri: String,
    /// URI to the serving-ready bundle (weights + Triton config.pbtxt).
    pub deployable_artifact_uri: String,
}

#[async_trait]
pub trait ModelRegistryClient: std::fmt::Debug + Send + Sync {
    /// Fetches a registered model by name and version from the registry.
    async fn fetch_model(&self, name: &str, version: &str) -> anyhow::Result<RegisteredModel>;
}
