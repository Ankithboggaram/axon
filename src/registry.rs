//! ModelRegistryClient trait and model registry implementations.

use async_trait::async_trait;

use crate::config::ModelSchemaConfig;

pub mod mlflow;

/// A model artifact fetched from the registry.
#[derive(Debug)]
pub struct RegisteredModel {
    pub name: String,
    pub version: String,
    /// Local filesystem path to the downloaded ONNX artifact.
    pub local_path: String,
}

/// Subset of config values that can be seeded from the registry.
///
/// All fields are optional because registries vary in what they store.
/// Fields left as `None` must be filled in manually in the generated config.
#[derive(Debug, Default)]
pub struct ConfigSeed {
    /// Input and output tensor specs, derived from the model signature if available.
    pub model_schema: Option<ModelSchemaConfig>,
    pub mean: Option<f32>,
    pub std: Option<f32>,
    pub clip_min: Option<f32>,
    pub clip_max: Option<f32>,
    pub threshold: Option<f32>,
}

#[async_trait]
pub trait ModelRegistryClient: std::fmt::Debug + Send + Sync {
    /// Fetches the model artifact and returns its local path and metadata.
    async fn fetch_model(&self, name: &str, version: &str) -> anyhow::Result<RegisteredModel>;

    /// Fetches model signature and logged params to seed a starter config.
    async fn fetch_config_seed(&self, name: &str, version: &str) -> anyhow::Result<ConfigSeed>;
}
