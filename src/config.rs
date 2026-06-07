//! TOML configuration schema, deserialisation, and validation.

use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendType {
    /// In-process ONNX Runtime backend. Model path is resolved from the registry at startup.
    OnnxRuntime,
    /// NVIDIA Triton Inference Server backend. Not yet implemented.
    Triton,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoreType {
    Redis,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegistryType {
    Mlflow,
}

/// Output type produced by the postprocess stage.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputType {
    Binary,
    Probability,
    Raw,
}

#[derive(Clone, Debug, Deserialize)]
pub struct GrpcConfig {
    pub host: String,
    pub port: u16,
    /// How often the streaming RPC polls the feature store for updated features, in milliseconds.
    pub stream_poll_interval_ms: u64,
    /// Maximum time allowed for a single RPC before it is cancelled, in milliseconds.
    pub request_timeout_ms: u64,
    /// Number of pipelines kept in the pool for concurrent request processing.
    /// Defaults to the number of logical CPUs if not set.
    #[serde(default)]
    pub pool_size: Option<usize>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct BackendConfig {
    #[serde(rename = "type")]
    pub backend_type: BackendType,
    /// Model name registered in Triton. Required for the Triton backend only.
    pub model: Option<String>,
    /// Triton server host. Required for the Triton backend only.
    pub host: Option<String>,
    /// Triton server port. Required for the Triton backend only.
    pub port: Option<u16>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RegistryConfig {
    #[serde(rename = "type")]
    pub registry_type: RegistryType,
    pub uri: String,
    pub model_name: String,
    /// Accepts a version number or "latest".
    pub model_version: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct StoreConfig {
    #[serde(rename = "type")]
    pub store_type: StoreType,
    pub host: String,
    pub port: u16,
}

#[derive(Clone, Debug, Deserialize)]
pub struct MetricsConfig {
    pub port: u16,
}

/// Describes a single input or output tensor: name, data type, and shape.
#[derive(Clone, Debug, Deserialize)]
pub struct TensorSpec {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<i64>,
}

/// Defines the input and output tensor shapes the model expects.
#[derive(Clone, Debug, Deserialize)]
pub struct ModelSchemaConfig {
    pub inputs: Vec<TensorSpec>,
    pub outputs: Vec<TensorSpec>,
}

/// Per-stage observability options.
#[derive(Clone, Debug, Deserialize)]
pub struct StageObservability {
    /// Records p99/p999 execution latency for this stage.
    pub timed: Option<bool>,
    /// Emits a tracing span on every execution of this stage.
    pub instrumented: Option<bool>,
    /// Retries the stage on failure up to this many total attempts.
    pub retries: Option<u32>,
    /// Fails the stage if execution exceeds this budget, in milliseconds.
    pub deadline_ms: Option<u64>,
}

/// Each variant carries only the parameters relevant to that stage type.
///
// TODO: Implement drift_detect, audit, argmax stages.
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StageConfig {
    Validate {
        expected_shape: Vec<i64>,
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

#[derive(Clone, Debug, Deserialize)]
pub struct PipelineConfig {
    pub stages: Vec<StageConfig>,
}

/// Top-level config, owns all section configs.
#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub grpc: GrpcConfig,
    pub backend: BackendConfig,
    pub registry: RegistryConfig,
    pub store: StoreConfig,
    pub metrics: MetricsConfig,
    pub model_schema: ModelSchemaConfig,
    pub pipeline: PipelineConfig,
}

impl Config {
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&contents)?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        if self.pipeline.stages.is_empty() {
            anyhow::bail!("pipeline must have at least one stage");
        }

        if self.grpc.port == 0 {
            anyhow::bail!("grpc.port must not be 0");
        }

        if self.grpc.stream_poll_interval_ms == 0 {
            anyhow::bail!("grpc.stream_poll_interval_ms must not be 0");
        }

        if self.grpc.request_timeout_ms == 0 {
            anyhow::bail!("grpc.request_timeout_ms must not be 0");
        }

        if self.metrics.port == 0 {
            anyhow::bail!("metrics.port must not be 0");
        }

        match self.backend.backend_type {
            BackendType::Triton => {
                anyhow::bail!("Triton backend is not yet implemented; use onnx_runtime");
            }
            BackendType::OnnxRuntime => {}
        }

        if self.store.host.is_empty() {
            anyhow::bail!("store.host must not be empty");
        }

        if self.store.port == 0 {
            anyhow::bail!("store.port must not be 0");
        }

        if self.model_schema.inputs.is_empty() {
            anyhow::bail!("model_schema must define at least one input tensor");
        }

        if self.model_schema.inputs.len() > 1 {
            anyhow::bail!(
                "model_schema defines {} input tensors; only one input is supported in Phase 1",
                self.model_schema.inputs.len()
            );
        }

        if self.model_schema.outputs.is_empty() {
            anyhow::bail!("model_schema must define at least one output tensor");
        }

        for stage in &self.pipeline.stages {
            if let StageConfig::Normalize { std, .. } = stage
                && *std == 0.0
            {
                anyhow::bail!("normalize stage: std must not be 0 (division by zero)");
            }
        }

        Ok(())
    }
}
