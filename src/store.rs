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

use std::pin::Pin;
use std::time::Duration;

use async_trait::async_trait;
use futures_util::{Stream, StreamExt as _};
use ndarray::ArrayD;
use tokio_stream::wrappers::IntervalStream;

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
    /// Features were found and written into the destination buffer in full.
    Hit,
    /// No entry exists for the entity; the destination buffer is left unchanged.
    Miss,
}

/// Client for fetching pre-computed feature vectors from an external store.
///
/// Add new store backends by implementing this trait; no other code needs to change.
#[async_trait]
pub trait FeatureStore: std::fmt::Debug + Send + Sync {
    /// Checks that the store is reachable. Called once at startup before
    /// the gRPC health check is set to Serving.
    ///
    /// # Implementors
    ///
    /// **Thread safety:** `&self`; concurrent pings are safe and must not
    /// block other store operations.
    ///
    /// **Error contract:** return [`crate::error::StoreError::Connection`] if
    /// the store cannot be reached. Must not modify any store data.
    ///
    /// **Idempotency:** fully idempotent; safe to call any number of times.
    async fn ping(&self) -> Result<(), StoreError>;

    /// Fetches features for the given entity and writes them into `dest`.
    ///
    /// Writes directly into the pre-allocated scratchpad buffer to avoid any
    /// intermediate allocation. Returns `Miss` without modifying `dest` if no
    /// entry exists for the entity.
    ///
    /// # Implementors
    ///
    /// **Thread safety:** called concurrently for different entity IDs. Must not
    /// hold a lock that would serialize callers against one another.
    ///
    /// **Output contract:** on [`FetchResult::Hit`], `dest` must be fully
    /// overwritten with the stored feature vector and element count must equal
    /// `dest.len()`. On [`FetchResult::Miss`], `dest` must be left unmodified.
    ///
    /// **Error contract:** return [`crate::error::StoreError::Connection`] for
    /// network failures, [`crate::error::StoreError::Deserialize`] for corrupt
    /// payloads, or [`crate::error::StoreError::ShapeMismatch`] for a stored
    /// vector that does not fit `dest`. On any error, the contents of `dest`
    /// are unspecified.
    ///
    /// **Idempotency:** the same `entity_id` returns the same result on every
    /// call. No side effects on the store.
    async fn fetch_features(
        &self,
        entity_id: &str,
        dest: &mut ArrayD<f32>,
    ) -> Result<FetchResult, StoreError>;

    /// Returns a stream that yields `()` each time new features may be
    /// available for `entity_id`.
    ///
    /// The default implementation yields on a fixed `poll_interval` timer —
    /// behaviour identical to the previous poll loop. Stores that support push
    /// notifications (e.g. Redis pub/sub) should override this to yield only
    /// when an actual update arrives, eliminating unnecessary fetches.
    ///
    /// If the stream ends (e.g. the underlying connection drops) the caller
    /// should treat the streaming response as complete.
    async fn update_stream(
        &self,
        _entity_id: &str,
        poll_interval: Duration,
    ) -> Pin<Box<dyn Stream<Item = ()> + Send>> {
        Box::pin(IntervalStream::new(tokio::time::interval(poll_interval)).map(|_| ()))
    }
}
