//! Typed error enums for every fallible boundary in axon.
//!
//! All enums are `#[non_exhaustive]`: always include a wildcard arm when
//! matching to remain compatible with future variants.
//!
//! Each enum corresponds to one subsystem:
//! - [`ConfigError`]   — config file loading and validation
//! - [`BackendError`]  — inference backend (ONNX Runtime, Triton, …)
//! - [`StoreError`]    — feature store (Redis, …)
//! - [`RegistryError`] — model registry (MLflow, …)
//! - [`ServeError`]    — server startup and metrics initialisation

use thiserror::Error;

/// Errors from loading or validating a [`Config`](crate::config::Config).
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum ConfigError {
    /// The config file could not be read from disk.
    #[error("failed to read config file: {0}")]
    Io(#[from] std::io::Error),

    /// The config file content could not be parsed as valid TOML.
    #[error("failed to parse config: {0}")]
    Parse(#[from] toml::de::Error),

    /// The config was parsed successfully but failed semantic validation.
    ///
    /// `field` identifies the offending config key; `reason` explains why.
    #[error("invalid config — {field}: {reason}")]
    Invalid { field: &'static str, reason: String },
}

/// Errors from an inference [`Backend`](crate::backend::Backend).
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum BackendError {
    /// The inference session could not be created at startup.
    #[error("failed to create inference session: {0}")]
    SessionCreation(String),

    /// The input tensor shape did not match what the model expects.
    #[error("input shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch { expected: Vec<i64>, got: Vec<i64> },

    /// The model produced an unexpected number of outputs.
    #[error("output count mismatch: expected {expected}, got {got}")]
    OutputCountMismatch { expected: usize, got: usize },

    /// The inference run itself failed.
    #[error("inference failed: {0}")]
    InferenceFailed(String),
}

/// Errors from a [`FeatureStore`](crate::store::FeatureStore).
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum StoreError {
    /// A connection or pool could not be established.
    #[error("failed to connect to feature store: {0}")]
    Connection(String),

    /// A feature fetch command failed.
    #[error("feature fetch failed for entity '{entity_id}': {reason}")]
    Fetch { entity_id: String, reason: String },

    /// The raw bytes returned by the store could not be deserialised.
    #[error("failed to deserialise features for entity '{entity_id}': {reason}")]
    Deserialize { entity_id: String, reason: String },

    /// The deserialised feature vector has the wrong shape for the pipeline.
    #[error("feature shape mismatch for entity '{entity_id}': expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        entity_id: String,
        expected: Vec<usize>,
        got: Vec<usize>,
    },
}

/// Errors from a [`ModelRegistryClient`](crate::registry::ModelRegistryClient).
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum RegistryError {
    /// The HTTP client could not be built.
    #[error("failed to build HTTP client: {0}")]
    HttpClient(String),

    /// A request to the registry failed.
    #[error("registry request failed: {0}")]
    Request(String),

    /// The registry response could not be parsed.
    #[error("failed to parse registry response: {0}")]
    Parse(String),

    /// No version of the requested model exists in the registry.
    #[error("model '{name}' not found in registry")]
    ModelNotFound { name: String },

    /// A local file operation (temp dir, artifact write) failed.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Errors from server startup or metrics initialisation.
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum ServeError {
    /// A Prometheus metric could not be registered.
    #[error("failed to register metric '{name}': {reason}")]
    MetricsRegistration { name: &'static str, reason: String },

    /// The Prometheus metrics could not be encoded for the scrape endpoint.
    #[error("failed to encode metrics: {0}")]
    MetricsEncoding(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_invalid_includes_field_and_reason() {
        let e = ConfigError::Invalid {
            field: "grpc.port",
            reason: "must not be 0".into(),
        };
        let msg = format!("{e}");
        assert!(msg.contains("grpc.port"));
        assert!(msg.contains("must not be 0"));
    }

    #[test]
    fn backend_shape_mismatch_shows_both_shapes() {
        let e = BackendError::ShapeMismatch {
            expected: vec![1, 32],
            got: vec![1, 16],
        };
        let msg = format!("{e}");
        assert!(msg.contains("32"));
        assert!(msg.contains("16"));
    }

    #[test]
    fn store_fetch_includes_entity_id() {
        let e = StoreError::Fetch {
            entity_id: "user_123".into(),
            reason: "connection reset".into(),
        };
        let msg = format!("{e}");
        assert!(msg.contains("user_123"));
        assert!(msg.contains("connection reset"));
    }

    #[test]
    fn registry_not_found_includes_model_name() {
        let e = RegistryError::ModelNotFound {
            name: "fraud_model".into(),
        };
        assert!(format!("{e}").contains("fraud_model"));
    }

    #[test]
    fn serve_metrics_registration_includes_metric_name() {
        let e = ServeError::MetricsRegistration {
            name: "axon_requests_total",
            reason: "already registered".into(),
        };
        let msg = format!("{e}");
        assert!(msg.contains("axon_requests_total"));
        assert!(msg.contains("already registered"));
    }

    #[test]
    fn config_io_error_is_source_chained() {
        let io = std::io::Error::new(std::io::ErrorKind::NotFound, "no such file");
        let e = ConfigError::Io(io);
        assert!(std::error::Error::source(&e).is_some());
    }
}
