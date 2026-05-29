//! MLflow model registry client.

use std::collections::HashMap;
use std::path::PathBuf;

use async_trait::async_trait;
use serde::Deserialize;

use crate::registry::{ConfigSeed, ModelRegistryClient, RegisteredModel};

pub struct MlflowClient {
    tracking_uri: String,
    http: reqwest::Client,
}

impl MlflowClient {
    pub fn new(tracking_uri: &str) -> anyhow::Result<Self> {
        let http = reqwest::Client::builder()
            .build()
            .map_err(|e| anyhow::anyhow!("failed to build HTTP client: {e}"))?;
        Ok(Self {
            tracking_uri: tracking_uri.trim_end_matches('/').to_owned(),
            http,
        })
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
    async fn fetch_model(&self, name: &str, version: &str) -> anyhow::Result<RegisteredModel> {
        let resolved = self.resolve_version(name, version).await?;
        let version_data = self.get_model_version(name, &resolved).await?;
        let local_path = self.download_artifact(name, &resolved).await?;

        Ok(RegisteredModel {
            name: version_data.name,
            version: version_data.version,
            local_path: local_path.to_string_lossy().into_owned(),
        })
    }

    async fn fetch_config_seed(&self, name: &str, version: &str) -> anyhow::Result<ConfigSeed> {
        let resolved = self.resolve_version(name, version).await?;
        let version_data = self.get_model_version(name, &resolved).await?;
        let params = self.get_run_params(&version_data.run_id).await?;

        Ok(ConfigSeed {
            // Parsing the MLmodel signature requires downloading and parsing a YAML file;
            // left as None for now so the user fills it in manually.
            model_schema: None,
            mean: params.get("mean").and_then(|v| v.parse().ok()),
            std: params.get("std").and_then(|v| v.parse().ok()),
            clip_min: params.get("clip_min").and_then(|v| v.parse().ok()),
            clip_max: params.get("clip_max").and_then(|v| v.parse().ok()),
            threshold: params.get("threshold").and_then(|v| v.parse().ok()),
        })
    }
}

impl MlflowClient {
    /// Resolves "latest" to a concrete version number; passes through anything else unchanged.
    async fn resolve_version(&self, name: &str, version: &str) -> anyhow::Result<String> {
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
            .map_err(|e| anyhow::anyhow!("failed to fetch latest versions: {e}"))?;

        check_status(&resp, "latest versions")?;

        let data: LatestVersionsResponse = resp
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("failed to parse latest versions response: {e}"))?;

        data.model_versions
            .into_iter()
            .max_by_key(|v| v.version.parse::<u64>().unwrap_or(0))
            .map(|v| v.version)
            .ok_or_else(|| anyhow::anyhow!("no versions found for model '{name}'"))
    }

    async fn get_model_version(
        &self,
        name: &str,
        version: &str,
    ) -> anyhow::Result<ModelVersionData> {
        let resp = self
            .http
            .get(format!(
                "{}/api/2.0/mlflow/model-versions/get",
                self.tracking_uri
            ))
            .query(&[("name", name), ("version", version)])
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("failed to fetch model version: {e}"))?;

        check_status(&resp, "model version")?;

        let data: ModelVersionResponse = resp
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("failed to parse model version response: {e}"))?;

        Ok(data.model_version)
    }

    async fn get_run_params(&self, run_id: &str) -> anyhow::Result<HashMap<String, String>> {
        let resp = self
            .http
            .get(format!("{}/api/2.0/mlflow/runs/get", self.tracking_uri))
            .query(&[("run_id", run_id)])
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("failed to fetch run: {e}"))?;

        check_status(&resp, "run")?;

        let data: RunResponse = resp
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("failed to parse run response: {e}"))?;

        let params = data
            .run
            .data
            .params
            .unwrap_or_default()
            .into_iter()
            .map(|p| (p.key, p.value))
            .collect();

        Ok(params)
    }

    /// Downloads the ONNX artifact to a local temp directory and returns the path.
    async fn download_artifact(&self, name: &str, version: &str) -> anyhow::Result<PathBuf> {
        let resp = self
            .http
            .get(format!("{}/model-versions/get-artifact", self.tracking_uri))
            .query(&[("name", name), ("version", version), ("path", "model.onnx")])
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("failed to download artifact: {e}"))?;

        check_status(&resp, "artifact download")?;

        let bytes = resp
            .bytes()
            .await
            .map_err(|e| anyhow::anyhow!("failed to read artifact bytes: {e}"))?;

        let dir = std::env::temp_dir().join("axon");
        std::fs::create_dir_all(&dir)
            .map_err(|e| anyhow::anyhow!("failed to create temp dir: {e}"))?;

        let path = dir.join(format!("{name}_v{version}.onnx"));
        std::fs::write(&path, bytes)
            .map_err(|e| anyhow::anyhow!("failed to write artifact to disk: {e}"))?;

        Ok(path)
    }
}

/// Returns an error if the response status is not 2xx.
#[cold]
fn check_status(resp: &reqwest::Response, context: &str) -> anyhow::Result<()> {
    if !resp.status().is_success() {
        anyhow::bail!("MLflow returned {} for {context}", resp.status());
    }
    Ok(())
}

// MLflow REST API response types.

#[derive(Deserialize)]
struct ModelVersionResponse {
    model_version: ModelVersionData,
}

#[derive(Deserialize)]
struct ModelVersionData {
    name: String,
    version: String,
    run_id: String,
}

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
