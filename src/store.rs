//! Feature store trait and implementations.
//!
//! A [`FeatureStore`] retrieves pre-computed feature vectors for a given entity
//! and writes them directly into the inference scratchpad buffer. The
//! write-in-place contract means no allocation occurs on the request hot path.
//!
//! ## FetchResult contract
//!
//! [`FeatureStore::fetch_features`] returns a [`FetchResult`] to distinguish
//! two outcomes:
//!
//! - [`FetchResult::Hit`]: features were found and written into `dest` in
//!   full. The buffer is ready for the next pipeline stage.
//! - [`FetchResult::Miss`]: no entry exists for the entity. `dest` is left
//!   unchanged (zeroed from scratchpad initialisation). The `impute` stage
//!   handles this case; callers must not treat a zeroed buffer as valid data
//!   without an explicit imputation step in the pipeline.
//!
//! ## Implementations
//!
//! - [`redis::RedisStore`]: MessagePack-encoded vectors stored in Redis

use async_trait::async_trait;
use ndarray::ArrayD;

use crate::error::StoreError;

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
    async fn ping(&self) -> Result<(), StoreError>;

    /// Fetches features for the given entity and writes them into `dest`.
    ///
    /// Writes directly into the pre-allocated scratchpad buffer to avoid any
    /// intermediate allocation. Returns `Miss` without modifying `dest` if no
    /// entry exists for the entity.
    async fn fetch_features(
        &self,
        entity_id: &str,
        dest: &mut ArrayD<f32>,
    ) -> Result<FetchResult, StoreError>;
}
