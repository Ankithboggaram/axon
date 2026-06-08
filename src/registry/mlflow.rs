//! MLflow model registry client.

use std::collections::HashMap;
use std::path::PathBuf;

use futures_util::StreamExt as _;
use tokio::io::AsyncWriteExt as _;

use async_trait::async_trait;
use serde::Deserialize;

use crate::config::{ModelSchemaConfig, TensorSpec};
use crate::error::RegistryError;
use crate::registry::{ConfigSeed, ModelRegistryClient, RegisteredModel};

/// HTTP client for the MLflow REST API.
pub struct MlflowClient {
    tracking_uri: String,
    http: reqwest::Client,
}

impl MlflowClient {
    /// Creates a new client pointing at the given MLflow tracking server URI.
    pub fn new(tracking_uri: &str) -> Result<Self, RegistryError> {
        let http = reqwest::Client::builder()
            .build()
            .map_err(|e| RegistryError::HttpClient(e.to_string()))?;
        Ok(Self {
            tracking_uri: tracking_uri.trim_end_matches('/').to_owned(),
            http,
        })
    }

    /// Resolves "latest" to a concrete version number; passes through anything else unchanged.
    async fn resolve_version(&self, name: &str, version: &str) -> Result<String, RegistryError> {
        if version != "latest" {
            return Ok(version.to_owned());
        }

        let resp = self
            .http
            .get(format!(
                "{}/api/2.0/mlflow/registered-models/get-latest-versions",
                self.tracking_uri
            ))
            .query(&[("name", name)])
            .send()
            .await
            .map_err(|e| RegistryError::Request(format!("failed to fetch latest versions: {e}")))?;

        check_status(&resp, "latest versions")?;

        let data: LatestVersionsResponse = resp.json().await.map_err(|e| {
            RegistryError::Parse(format!("failed to parse latest versions response: {e}"))
        })?;

        data.model_versions
            .into_iter()
            .max_by_key(|v| {
                v.version.parse::<u64>().unwrap_or_else(|_| {
                    tracing::warn!(version = %v.version, "non-numeric MLflow version treated as 0");
                    0
                })
            })
            .map(|v| v.version)
            .ok_or_else(|| RegistryError::ModelNotFound {
                name: name.to_owned(),
            })
    }

    async fn get_run_params(&self, run_id: &str) -> Result<HashMap<String, String>, RegistryError> {
        let resp = self
            .http
            .get(format!("{}/api/2.0/mlflow/runs/get", self.tracking_uri))
            .query(&[("run_id", run_id)])
            .send()
            .await
            .map_err(|e| RegistryError::Request(format!("failed to fetch run: {e}")))?;

        check_status(&resp, "run")?;

        let data: RunResponse = resp
            .json()
            .await
            .map_err(|e| RegistryError::Parse(format!("failed to parse run response: {e}")))?;

        Ok(data
            .run
            .data
            .params
            .unwrap_or_default()
            .into_iter()
            .map(|p| (p.key, p.value))
            .collect())
    }

    /// Downloads the MLmodel metadata file and returns its text.
    async fn download_mlmodel(&self, name: &str, version: &str) -> Result<String, RegistryError> {
        self.get_artifact(name, version, "MLmodel")
            .await?
            .text()
            .await
            .map_err(|e| RegistryError::Request(format!("failed to read MLmodel response: {e}")))
    }

    /// Downloads the ONNX artifact to a local temp directory and returns the path.
    ///
    /// Streams directly to disk to avoid buffering the entire model in memory.
    async fn download_artifact(&self, name: &str, version: &str) -> Result<PathBuf, RegistryError> {
        let dir = std::env::temp_dir().join("axon");
        tokio::fs::create_dir_all(&dir).await?;

        let path = dir.join(format!("{name}_v{version}.onnx"));
        let mut file = tokio::fs::File::create(&path).await?;

        let mut stream = self
            .get_artifact(name, version, "model.onnx")
            .await?
            .bytes_stream();

        while let Some(chunk) = stream.next().await {
            let chunk =
                chunk.map_err(|e| RegistryError::Request(format!("artifact stream error: {e}")))?;
            file.write_all(&chunk).await?;
        }

        Ok(path)
    }

    /// Sends a GET request for an artifact file and returns the response on success.
    async fn get_artifact(
        &self,
        name: &str,
        version: &str,
        path: &str,
    ) -> Result<reqwest::Response, RegistryError> {
        let resp = self
            .http
            .get(format!("{}/model-versions/get-artifact", self.tracking_uri))
            .query(&[("name", name), ("version", version), ("path", path)])
            .send()
            .await
            .map_err(|e| {
                RegistryError::Request(format!("failed to request artifact '{path}': {e}"))
            })?;

        check_status(&resp, path)?;

        Ok(resp)
    }
}

impl std::fmt::Debug for MlflowClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MlflowClient")
            .field("tracking_uri", &self.tracking_uri)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl ModelRegistryClient for MlflowClient {
    async fn fetch_model(
        &self,
        name: &str,
        version: &str,
    ) -> Result<RegisteredModel, RegistryError> {
        let resolved = self.resolve_version(name, version).await?;
        let local_path = self.download_artifact(name, &resolved).await?;

        Ok(RegisteredModel {
            name: name.to_owned(),
            version: resolved,
            local_path: local_path.to_string_lossy().into_owned(),
        })
    }

    async fn fetch_config_seed(
        &self,
        name: &str,
        version: &str,
    ) -> Result<ConfigSeed, RegistryError> {
        let resolved = self.resolve_version(name, version).await?;
        let mlmodel_text = self.download_mlmodel(name, &resolved).await?;

        // Parse once: MLmodel contains both the signature and the run_id.
        let mlmodel: MlModelFile = serde_yaml_ng::from_str(&mlmodel_text)
            .map_err(|e| RegistryError::Parse(format!("failed to parse MLmodel YAML: {e}")))?;

        let run_id = mlmodel.run_id.ok_or_else(|| {
            RegistryError::Parse(format!(
                "model '{name}' v{resolved} has no associated training run; \
                 params and schema cannot be seeded automatically"
            ))
        })?;

        let params = self.get_run_params(&run_id).await?;

        Ok(ConfigSeed {
            model_schema: Some(parse_model_schema(mlmodel.signature)?),
            mean: params.get("mean").and_then(|v| v.parse().ok()),
            std: params.get("std").and_then(|v| v.parse().ok()),
            clip_min: params.get("clip_min").and_then(|v| v.parse().ok()),
            clip_max: params.get("clip_max").and_then(|v| v.parse().ok()),
            threshold: params.get("threshold").and_then(|v| v.parse().ok()),
        })
    }
}

/// Returns an error if the response status is not 2xx.
#[cold]
#[inline(never)]
fn check_status(resp: &reqwest::Response, context: &str) -> Result<(), RegistryError> {
    if !resp.status().is_success() {
        return Err(RegistryError::Request(format!(
            "MLflow returned {} for {context}",
            resp.status()
        )));
    }
    Ok(())
}

/// Parses a `ModelSchemaConfig` from the signature block of a parsed MLmodel file.
///
/// Fails if the signature block is absent, or if any tensor in the signature
/// is not a tensor type (e.g. a column/tabular type), since axon only supports tensors.
fn parse_model_schema(
    signature: Option<MlModelSignature>,
) -> Result<ModelSchemaConfig, RegistryError> {
    let sig = signature.ok_or_else(|| {
        RegistryError::Parse(
            "MLmodel has no signature block; log the model with mlflow.models.infer_signature()"
                .into(),
        )
    })?;

    let raw_inputs: Vec<MlTensorEntry> = serde_json::from_str(&sig.inputs).map_err(|e| {
        RegistryError::Parse(format!("failed to parse MLmodel signature inputs: {e}"))
    })?;
    let raw_outputs: Vec<MlTensorEntry> = serde_json::from_str(&sig.outputs).map_err(|e| {
        RegistryError::Parse(format!("failed to parse MLmodel signature outputs: {e}"))
    })?;

    let convert = |entry: MlTensorEntry| -> Result<TensorSpec, RegistryError> {
        let spec = entry.tensor_spec.ok_or_else(|| {
            RegistryError::Parse(format!(
                "MLmodel signature field '{}' has type '{}', not 'tensor'; \
                 axon only supports tensor inputs and outputs",
                entry.name, entry.field_type,
            ))
        })?;
        Ok(TensorSpec {
            name: entry.name,
            dtype: spec.dtype,
            shape: spec.shape,
        })
    };

    Ok(ModelSchemaConfig {
        inputs: raw_inputs
            .into_iter()
            .map(convert)
            .collect::<Result<_, RegistryError>>()?,
        outputs: raw_outputs
            .into_iter()
            .map(convert)
            .collect::<Result<_, RegistryError>>()?,
    })
}

// MLflow REST API response types.

#[derive(Deserialize)]
struct LatestVersionsResponse {
    model_versions: Vec<LatestVersionData>,
}

#[derive(Deserialize)]
struct LatestVersionData {
    version: String,
}

#[derive(Deserialize)]
struct RunResponse {
    run: RunData,
}

#[derive(Deserialize)]
struct RunData {
    data: RunDataInner,
}

#[derive(Deserialize)]
struct RunDataInner {
    params: Option<Vec<Param>>,
}

#[derive(Deserialize)]
struct Param {
    key: String,
    value: String,
}

// MLmodel YAML types.

#[derive(Deserialize)]
struct MlModelFile {
    run_id: Option<String>,
    signature: Option<MlModelSignature>,
}

#[derive(Deserialize)]
struct MlModelSignature {
    inputs: String,
    outputs: String,
}

#[derive(Deserialize)]
struct MlTensorEntry {
    name: String,
    #[serde(rename = "type")]
    field_type: String,
    #[serde(rename = "tensor-spec")]
    tensor_spec: Option<MlTensorSpec>,
}

#[derive(Deserialize)]
struct MlTensorSpec {
    dtype: String,
    shape: Vec<i64>,
}
