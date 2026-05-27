//! TOML configuration schema, deserialisation, and validation.

use serde::Deserialize;

// --- enums ---

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendType {
    Triton,
    OnnxRuntime,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoreType {
    Redis,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegistryType {
    Mlflow,
}

/// Output type produced by the postprocess stage.
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputType {
    Binary,
    Probability,
    Raw,
}

// --- configs ---

#[derive(Deserialize)]
pub struct GrpcConfig {
    pub host: String,
    pub port: u16,
}

#[derive(Deserialize)]
pub struct StreamConfig {
    pub source: String,
    pub stream_keys: Vec<String>,
}

#[derive(Deserialize)]
pub struct BackendConfig {
    #[serde(rename = "type")]
    pub backend_type: BackendType,
    pub model: String,
    pub host: String,
    pub port: u16,
}

#[derive(Deserialize)]
pub struct RegistryConfig {
    #[serde(rename = "type")]
    pub registry_type: RegistryType,
    pub uri: String,
    pub model_name: String,
    /// Accepts a version number or "latest".
    pub model_version: String,
}

#[derive(Deserialize)]
pub struct StoreConfig {
    #[serde(rename = "type")]
    pub store_type: StoreType,
    pub host: String,
    pub port: u16,
}

#[derive(Deserialize)]
pub struct MetricsConfig {
    pub port: u16,
}

/// Describes a single input or output tensor: name, data type, and shape.
#[derive(Deserialize)]
pub struct TensorSpec {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<i64>,
}

/// Defines the input and output tensor shapes the model expects.
#[derive(Deserialize)]
pub struct ModelSchemaConfig {
    pub inputs: Vec<TensorSpec>,
    pub outputs: Vec<TensorSpec>,
}

/// Per-stage observability options.
#[derive(Deserialize)]
pub struct StageObservability {
    /// Wrap the stage with pipex::metrics::Timed to record p99/p999 latency.
    pub timed: Option<bool>,
    /// Wrap the stage with pipex::instrument::Instrumented to emit tracing spans.
    pub instrumented: Option<bool>,
    /// Wrap the stage with pipex::retry::Retry. Value is the number of retries.
    pub retries: Option<u32>,
}

/// Each variant carries only the parameters relevant to that stage type.
///
// TODO: implement drift_detect, audit, argmax stages.
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StageConfig {
    Validate {
        expected_features: usize,
        min: f32,
        max: f32,
        #[serde(flatten)]
        observability: StageObservability,
    },
    Normalize {
        mean: f32,
        std: f32,
        #[serde(flatten)]
        observability: StageObservability,
    },
    /// Clips feature values to [min, max] before normalisation.
    Clip {
        min: f32,
        max: f32,
        #[serde(flatten)]
        observability: StageObservability,
    },
    /// Replaces missing (NaN) feature values with a fixed default.
    Impute {
        default_value: f32,
        #[serde(flatten)]
        observability: StageObservability,
    },
    Infer {
        #[serde(flatten)]
        observability: StageObservability,
    },
    Postprocess {
        threshold: f32,
        output_type: OutputType,
        #[serde(flatten)]
        observability: StageObservability,
    },
}

#[derive(Deserialize)]
pub struct PipelineConfig {
    pub stages: Vec<StageConfig>,
}

/// Top-level config, owns all section configs.
#[derive(Deserialize)]
pub struct Config {
    pub grpc: GrpcConfig,
    pub stream: StreamConfig,
    pub backend: BackendConfig,
    pub registry: RegistryConfig,
    pub store: StoreConfig,
    pub metrics: MetricsConfig,
    pub model_schema: ModelSchemaConfig,
    pub pipeline: PipelineConfig,
}

pub fn load(path: &str) -> anyhow::Result<Config> {
    let contents = std::fs::read_to_string(path)?;
    let config = toml::from_str(&contents)?;
    Ok(config)
}
