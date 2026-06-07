//! TOML configuration schema, deserialisation, and validation.

use serde::Deserialize;

use crate::error::ConfigError;

#[non_exhaustive]
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendType {
    /// In-process ONNX Runtime backend. Model path is resolved from the registry at startup.
    OnnxRuntime,
    /// NVIDIA Triton Inference Server backend. Not yet implemented.
    Triton,
}

#[non_exhaustive]
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoreType {
    Redis,
}

#[non_exhaustive]
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegistryType {
    Mlflow,
}

/// Output type produced by the postprocess stage.
#[non_exhaustive]
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
#[non_exhaustive]
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
    pub fn load(path: &str) -> Result<Self, ConfigError> {
        let contents = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&contents)?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.pipeline.stages.is_empty() {
            return Err(ConfigError::Invalid {
                field: "pipeline.stages",
                reason: "must have at least one stage".into(),
            });
        }

        if self.grpc.port == 0 {
            return Err(ConfigError::Invalid {
                field: "grpc.port",
                reason: "must not be 0".into(),
            });
        }

        if self.grpc.stream_poll_interval_ms == 0 {
            return Err(ConfigError::Invalid {
                field: "grpc.stream_poll_interval_ms",
                reason: "must not be 0".into(),
            });
        }

        if self.grpc.request_timeout_ms == 0 {
            return Err(ConfigError::Invalid {
                field: "grpc.request_timeout_ms",
                reason: "must not be 0".into(),
            });
        }

        if self.metrics.port == 0 {
            return Err(ConfigError::Invalid {
                field: "metrics.port",
                reason: "must not be 0".into(),
            });
        }

        match self.backend.backend_type {
            BackendType::Triton => {
                return Err(ConfigError::Invalid {
                    field: "backend.type",
                    reason: "Triton is not yet implemented; use onnx_runtime".into(),
                });
            }
            BackendType::OnnxRuntime => {}
        }

        if self.store.host.is_empty() {
            return Err(ConfigError::Invalid {
                field: "store.host",
                reason: "must not be empty".into(),
            });
        }

        if self.store.port == 0 {
            return Err(ConfigError::Invalid {
                field: "store.port",
                reason: "must not be 0".into(),
            });
        }

        if self.model_schema.inputs.is_empty() {
            return Err(ConfigError::Invalid {
                field: "model_schema.inputs",
                reason: "must define at least one tensor".into(),
            });
        }

        if self.model_schema.inputs.len() > 1 {
            return Err(ConfigError::Invalid {
                field: "model_schema.inputs",
                reason: format!(
                    "defines {} tensors; only one input is supported",
                    self.model_schema.inputs.len()
                ),
            });
        }

        if self.model_schema.outputs.is_empty() {
            return Err(ConfigError::Invalid {
                field: "model_schema.outputs",
                reason: "must define at least one tensor".into(),
            });
        }

        for stage in &self.pipeline.stages {
            if let StageConfig::Normalize { std, .. } = stage
                && *std == 0.0
            {
                return Err(ConfigError::Invalid {
                    field: "pipeline.stages[normalize].std",
                    reason: "must not be 0 (division by zero)".into(),
                });
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs() -> StageObservability {
        StageObservability {
            timed: None,
            instrumented: None,
            retries: None,
            deadline_ms: None,
        }
    }

    fn valid_config() -> Config {
        Config {
            grpc: GrpcConfig {
                host: "0.0.0.0".to_owned(),
                port: 50051,
                stream_poll_interval_ms: 100,
                request_timeout_ms: 5000,
                pool_size: None,
            },
            backend: BackendConfig {
                backend_type: BackendType::OnnxRuntime,
                model: None,
                host: None,
                port: None,
            },
            registry: RegistryConfig {
                registry_type: RegistryType::Mlflow,
                uri: "http://localhost:5000".to_owned(),
                model_name: "model".to_owned(),
                model_version: "1".to_owned(),
            },
            store: StoreConfig {
                store_type: StoreType::Redis,
                host: "localhost".to_owned(),
                port: 6379,
            },
            metrics: MetricsConfig { port: 9090 },
            model_schema: ModelSchemaConfig {
                inputs: vec![TensorSpec {
                    name: "input".to_owned(),
                    dtype: "float32".to_owned(),
                    shape: vec![1, 10],
                }],
                outputs: vec![TensorSpec {
                    name: "output".to_owned(),
                    dtype: "float32".to_owned(),
                    shape: vec![1, 1],
                }],
            },
            pipeline: PipelineConfig {
                stages: vec![StageConfig::Infer {
                    observability: obs(),
                }],
            },
        }
    }

    #[test]
    fn valid_config_passes() {
        assert!(valid_config().validate().is_ok());
    }

    #[test]
    fn rejects_empty_pipeline() {
        let mut cfg = valid_config();
        cfg.pipeline.stages.clear();
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn rejects_zero_grpc_port() {
        let mut cfg = valid_config();
        cfg.grpc.port = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn rejects_zero_stream_poll_interval() {
        let mut cfg = valid_config();
        cfg.grpc.stream_poll_interval_ms = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn rejects_zero_request_timeout() {
        let mut cfg = valid_config();
        cfg.grpc.request_timeout_ms = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn rejects_zero_metrics_port() {
        let mut cfg = valid_config();
        cfg.metrics.port = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn rejects_triton_backend() {
        let mut cfg = valid_config();
        cfg.backend.backend_type = BackendType::Triton;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn rejects_empty_store_host() {
        let mut cfg = valid_config();
        cfg.store.host.clear();
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn rejects_zero_store_port() {
        let mut cfg = valid_config();
        cfg.store.port = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn rejects_no_inputs() {
        let mut cfg = valid_config();
        cfg.model_schema.inputs.clear();
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn rejects_multiple_inputs() {
        let mut cfg = valid_config();
        cfg.model_schema.inputs.push(TensorSpec {
            name: "input2".to_owned(),
            dtype: "float32".to_owned(),
            shape: vec![1, 5],
        });
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn rejects_no_outputs() {
        let mut cfg = valid_config();
        cfg.model_schema.outputs.clear();
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn rejects_normalize_zero_std() {
        let mut cfg = valid_config();
        cfg.pipeline.stages = vec![StageConfig::Normalize {
            mean: 0.0,
            std: 0.0,
            observability: obs(),
        }];
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn accepts_normalize_nonzero_std() {
        let mut cfg = valid_config();
        cfg.pipeline.stages = vec![
            StageConfig::Normalize {
                mean: 0.5,
                std: 1.5,
                observability: obs(),
            },
            StageConfig::Infer {
                observability: obs(),
            },
        ];
        assert!(cfg.validate().is_ok());
    }
}
