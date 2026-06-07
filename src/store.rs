//! FeatureStore trait and feature store implementations.

use async_trait::async_trait;
use ndarray::ArrayD;

pub mod redis;

/// Outcome of a feature fetch.
///
/// `Hit` means features were found and written into the destination buffer.
/// `Miss` means no entry exists for the entity; the caller should let the
/// impute stage handle the zeroed buffer.
#[non_exhaustive]
#[derive(Debug)]
pub enum FetchResult {
    Hit,
    Miss,
}

#[async_trait]
pub trait FeatureStore: std::fmt::Debug + Send + Sync {
    /// Checks that the store is reachable. Called once at startup before
    /// the gRPC health check is set to Serving.
    async fn ping(&self) -> anyhow::Result<()>;

    /// Fetches features for the given entity and writes them into `dest`.
    ///
    /// Writes directly into the pre-allocated scratchpad buffer to avoid any
    /// intermediate allocation. Returns `Miss` without modifying `dest` if no
    /// entry exists for the entity.
    async fn fetch_features(
        &self,
        entity_id: &str,
        dest: &mut ArrayD<f32>,
    ) -> anyhow::Result<FetchResult>;
}
