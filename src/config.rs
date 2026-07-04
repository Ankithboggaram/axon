//! TOML configuration schema, deserialisation, and validation.

use serde::Deserialize;

use crate::error::ConfigError;

/// Hardware device used for model inference.
///
/// Axon maps this to an ONNX Runtime execution provider. Non-CPU providers
/// fail at session creation if the required runtime libraries are not present;
/// they never silently fall back to CPU.
#[non_exhaustive]
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceConfig {
    /// CPU inference via ONNX Runtime's CPU execution provider. Default.
    #[default]
    Cpu,
    /// Apple Neural Engine / GPU via CoreML. macOS 10.15+ only.
    #[serde(rename = "coreml")]
    CoreMl,
    /// NVIDIA GPU via CUDA. Requires CUDA and cuDNN runtime libraries.
    Cuda,
    /// NVIDIA GPU via TensorRT. Requires CUDA, cuDNN, and TensorRT libraries.
    #[serde(rename = "tensorrt")]
    TensorRt,
}

/// Inference backend implementation.
#[non_exhaustive]
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendType {
    /// In-process ONNX Runtime backend. Model path is resolved from the registry at startup.
    OnnxRuntime,
    /// NVIDIA Triton Inference Server backend. Not yet implemented.
    Triton,
}

/// Feature store backend.
#[non_exhaustive]
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoreType {
    /// Redis feature store.
    Redis,
}

/// Model registry backend.
#[non_exhaustive]
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegistryType {
    /// MLflow model registry.
    Mlflow,
}

/// Output type produced by the postprocess stage.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputType {
    /// Thresholds the score to 1.0 (above threshold) or -1.0 (below threshold).
    Binary,
    /// Passes the raw model score through unchanged.
    Probability,
    /// Passes the raw model output through unchanged.
    Raw,
}

/// gRPC server settings.
#[derive(Clone, Debug, Deserialize)]
pub struct GrpcConfig {
    /// Host address the gRPC server binds to.
    pub host: String,
    /// Port the gRPC server listens on.
    pub port: u16,
    /// How often the streaming RPC polls the feature store for updated features, in milliseconds.
    pub stream_poll_interval_ms: u64,
    /// Maximum time allowed for a single RPC before it is cancelled, in milliseconds.
    pub request_timeout_ms: u64,
    /// Number of pipelines kept in the pool for concurrent request processing.
    /// Defaults to the number of logical CPUs if not set.
    #[serde(default)]
    pub pool_size: Option<usize>,
    /// Number of ONNX Runtime sessions in the backend pool.
    /// Defaults to the number of logical CPUs if not set.
    #[serde(default)]
    pub session_pool_size: Option<usize>,
}

/// Inference backend settings.
#[derive(Clone, Debug, Deserialize)]
pub struct BackendConfig {
    /// Which backend implementation to use.
    #[serde(rename = "type")]
    pub backend_type: BackendType,
    /// Hardware device for inference. Defaults to `"cpu"`.
    ///
    /// Valid values: `"cpu"`, `"coreml"`, `"cuda"`, `"tensorrt"`.
    /// Non-CPU devices fail fast at startup if the required runtime libraries
    /// are not present on the host.
    #[serde(default)]
    pub device: DeviceConfig,
}

/// Model registry connection settings.
#[derive(Clone, Debug, Deserialize)]
pub struct RegistryConfig {
    /// Which registry implementation to use.
    #[serde(rename = "type")]
    pub registry_type: RegistryType,
    /// Base URI of the registry server.
    pub uri: String,
    /// Name of the model to fetch from the registry.
    pub model_name: String,
    /// Accepts a version number or "latest".
    pub model_version: String,
}

/// Feature store connection settings.
#[derive(Clone, Debug, Deserialize)]
pub struct StoreConfig {
    /// Which store implementation to use.
    #[serde(rename = "type")]
    pub store_type: StoreType,
    /// Connection URL for the store, e.g. `redis://localhost:6379`.
    pub url: String,
    /// Prefix for feature keys in the store. Feature vectors are stored and
    /// looked up under `{key_prefix}:{entity_id}`. Defaults to
    /// `cortex_contract::keys::DEFAULT_KEY_PREFIX` (`"features"`).
    #[serde(default)]
    pub key_prefix: Option<String>,
    /// How often the background task pings the store to update the readiness probe, in seconds.
    /// Defaults to 10. The service is marked NOT_SERVING after two consecutive failures.
    #[serde(default)]
    pub health_check_interval_secs: Option<u64>,
}

/// Prometheus metrics HTTP exposition settings.
#[derive(Clone, Debug, Deserialize)]
pub struct MetricsConfig {
    /// Port the metrics HTTP server listens on.
    pub port: u16,
}

/// Describes a single input or output tensor: name, data type, and shape.
#[derive(Clone, Debug, Deserialize)]
pub struct TensorSpec {
    /// Tensor name as registered in the model.
    pub name: String,
    /// Data type of the tensor elements (e.g. "float32").
    pub dtype: String,
    /// Shape of the tensor, one element per dimension.
    pub shape: Vec<i64>,
}

/// Defines the input and output tensor shapes the model expects.
#[derive(Clone, Debug, Deserialize)]
pub struct ModelSchemaConfig {
    /// Specs for each input tensor the model accepts.
    pub inputs: Vec<TensorSpec>,
    /// Specs for each output tensor the model produces.
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
#[non_exhaustive]
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StageConfig {
    /// Checks that the input tensor shape matches the expected shape.
    Validate {
        /// Expected tensor dimensions; each element must be a positive size.
        expected_shape: Vec<i64>,
        /// Observability wrappers for this stage.
        #[serde(flatten)]
        observability: StageObservability,
    },
    /// Applies zero-mean unit-variance normalisation to the input tensor.
    Normalize {
        /// Mean subtracted from each element.
        mean: f32,
        /// Standard deviation used to normalise; must not be 0.
        std: f32,
        /// Observability wrappers for this stage.
        #[serde(flatten)]
        observability: StageObservability,
    },
    /// Clips feature values to [min, max] before normalisation.
    Clip {
        /// Lower bound; values below this are clamped to min.
        min: f32,
        /// Upper bound; values above this are clamped to max.
        max: f32,
        /// Observability wrappers for this stage.
        #[serde(flatten)]
        observability: StageObservability,
    },
    /// Replaces missing (NaN) feature values with a fixed default.
    Impute {
        /// Replacement value for NaN elements.
        default_value: f32,
        /// Observability wrappers for this stage.
        #[serde(flatten)]
        observability: StageObservability,
    },
    /// Runs model inference via the configured backend.
    Infer {
        /// Observability wrappers for this stage.
        #[serde(flatten)]
        observability: StageObservability,
    },
    /// Transforms raw model output into a structured prediction.
    Postprocess {
        /// Decision boundary for binary classification.
        threshold: f32,
        /// How to interpret and transform the raw model score.
        output_type: OutputType,
        /// Observability wrappers for this stage.
        #[serde(flatten)]
        observability: StageObservability,
    },
}

/// Ordered list of pipeline stages to execute per request.
#[derive(Clone, Debug, Deserialize)]
pub struct PipelineConfig {
    /// Stage definitions in execution order.
    pub stages: Vec<StageConfig>,
}

/// Top-level config, owns all section configs.
#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    /// gRPC server settings.
    pub grpc: GrpcConfig,
    /// Inference backend settings.
    pub backend: BackendConfig,
    /// Model registry settings.
    pub registry: RegistryConfig,
    /// Feature store settings.
    pub store: StoreConfig,
    /// Prometheus metrics settings.
    pub metrics: MetricsConfig,
    /// Input and output tensor specs expected by the model.
    pub model_schema: ModelSchemaConfig,
    /// Pipeline stage configuration.
    pub pipeline: PipelineConfig,
}

impl Config {
    /// Loads and validates a config from the given TOML file path.
    pub fn load(path: &str) -> Result<Self, ConfigError> {
        let contents = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&contents)?;
        config.validate()?;
        Ok(config)
    }

    /// Validates all config fields for semantic correctness.
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

        if self.store.url.is_empty() {
            return Err(ConfigError::Invalid {
                field: "store.url",
                reason: "must not be empty".into(),
            });
        }

        if self.grpc.session_pool_size == Some(0) {
            return Err(ConfigError::Invalid {
                field: "grpc.session_pool_size",
                reason: "must not be 0; omit the field to use the default".into(),
            });
        }

        if self.store.health_check_interval_secs == Some(0) {
            return Err(ConfigError::Invalid {
                field: "store.health_check_interval_secs",
                reason: "must not be 0; omit the field to use the default of 10 seconds".into(),
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

            if let StageConfig::Validate { expected_shape, .. } = stage {
                if expected_shape.is_empty() {
                    return Err(ConfigError::Invalid {
                        field: "pipeline.stages[validate].expected_shape",
                        reason: "must not be empty".into(),
                    });
                }
                for &dim in expected_shape {
                    if dim <= 0 {
                        return Err(ConfigError::Invalid {
                            field: "pipeline.stages[validate].expected_shape",
                            reason: format!("all dimensions must be positive, got {dim}"),
                        });
                    }
                }
            }

            if let StageConfig::Clip { min, max, .. } = stage
                && min >= max
            {
                return Err(ConfigError::Invalid {
                    field: "pipeline.stages[clip]",
                    reason: format!("min ({min}) must be less than max ({max})"),
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
                session_pool_size: None,
            },
            backend: BackendConfig {
                backend_type: BackendType::OnnxRuntime,
                device: DeviceConfig::Cpu,
            },
            registry: RegistryConfig {
                registry_type: RegistryType::Mlflow,
                uri: "http://localhost:5000".to_owned(),
                model_name: "model".to_owned(),
                model_version: "1".to_owned(),
            },
            store: StoreConfig {
                store_type: StoreType::Redis,
                url: "redis://localhost:6379".to_owned(),
                key_prefix: None,
                health_check_interval_secs: None,
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
    fn rejects_empty_store_url() {
        let mut cfg = valid_config();
        cfg.store.url.clear();
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn rejects_zero_session_pool_size() {
        let mut cfg = valid_config();
        cfg.grpc.session_pool_size = Some(0);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn accepts_explicit_session_pool_size() {
        let mut cfg = valid_config();
        cfg.grpc.session_pool_size = Some(4);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn rejects_zero_health_check_interval() {
        let mut cfg = valid_config();
        cfg.store.health_check_interval_secs = Some(0);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn accepts_explicit_health_check_interval() {
        let mut cfg = valid_config();
        cfg.store.health_check_interval_secs = Some(30);
        assert!(cfg.validate().is_ok());
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

    #[test]
    fn rejects_clip_min_ge_max() {
        let mut cfg = valid_config();
        cfg.pipeline.stages = vec![
            StageConfig::Clip {
                min: 1.0,
                max: 1.0,
                observability: obs(),
            },
            StageConfig::Infer {
                observability: obs(),
            },
        ];
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn accepts_clip_valid_range() {
        let mut cfg = valid_config();
        cfg.pipeline.stages = vec![
            StageConfig::Clip {
                min: -1.0,
                max: 1.0,
                observability: obs(),
            },
            StageConfig::Infer {
                observability: obs(),
            },
        ];
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn device_defaults_to_cpu_when_omitted() {
        let toml = r#"
            [grpc]
            host = "0.0.0.0"
            port = 50051
            stream_poll_interval_ms = 100
            request_timeout_ms = 5000

            [backend]
            type = "onnx_runtime"

            [registry]
            type = "mlflow"
            uri = "http://localhost:5000"
            model_name = "m"
            model_version = "1"

            [store]
            type = "redis"
            url = "redis://localhost:6379"

            [metrics]
            port = 9090

            [[model_schema.inputs]]
            name = "x"
            dtype = "float32"
            shape = [1, 4]

            [[model_schema.outputs]]
            name = "y"
            dtype = "float32"
            shape = [1, 1]

            [[pipeline.stages]]
            type = "infer"
        "#;
        let cfg: Config = toml::from_str(toml).unwrap();
        assert!(matches!(cfg.backend.device, DeviceConfig::Cpu));
    }

    #[test]
    fn device_cpu_parses() {
        let toml = r#"type = "onnx_runtime"
device = "cpu""#;
        let cfg: BackendConfig = toml::from_str(toml).unwrap();
        assert!(matches!(cfg.device, DeviceConfig::Cpu));
    }

    #[test]
    fn device_coreml_parses() {
        let toml = r#"type = "onnx_runtime"
device = "coreml""#;
        let cfg: BackendConfig = toml::from_str(toml).unwrap();
        assert!(matches!(cfg.device, DeviceConfig::CoreMl));
    }

    #[test]
    fn device_cuda_parses() {
        let toml = r#"type = "onnx_runtime"
device = "cuda""#;
        let cfg: BackendConfig = toml::from_str(toml).unwrap();
        assert!(matches!(cfg.device, DeviceConfig::Cuda));
    }

    #[test]
    fn device_tensorrt_parses() {
        let toml = r#"type = "onnx_runtime"
device = "tensorrt""#;
        let cfg: BackendConfig = toml::from_str(toml).unwrap();
        assert!(matches!(cfg.device, DeviceConfig::TensorRt));
    }

    #[test]
    fn device_unknown_value_rejected() {
        let toml = r#"type = "onnx_runtime"
device = "vulkan""#;
        assert!(toml::from_str::<BackendConfig>(toml).is_err());
    }
}
