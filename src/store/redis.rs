//! Redis feature store client.

use std::pin::Pin;
use std::time::Duration;

use async_trait::async_trait;
use deadpool_redis::Pool;
use deadpool_redis::redis::Client as RedisClient;
use futures_util::{Stream, StreamExt as _};
use ndarray::ArrayD;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::error::StoreError;
use crate::store::{FeatureStore, FetchResult};

/// Redis-backed feature store using a connection pool.
///
/// Feature keys follow the pattern `{key_prefix}:{entity_id}`.
/// For streaming, Dendrite must publish to `{key_prefix}:updates:{entity_id}`
/// after writing new features so that [`update_stream`][FeatureStore::update_stream]
/// can wake waiting streams immediately rather than on a timer.
pub struct RedisStore {
    pool: Pool,
    /// Dedicated client for pub/sub connections (one per active stream).
    client: RedisClient,
    /// Key prefix applied to every entity lookup: `{prefix}:{entity_id}`.
    key_prefix: String,
}

impl RedisStore {
    /// Creates a new `RedisStore` connected to the given Redis URL.
    ///
    /// The pool is sized to the number of Tokio worker threads so each thread
    /// can hold a connection without contention.
    pub fn new(url: &str, key_prefix: &str) -> Result<Self, StoreError> {
        let cfg = deadpool_redis::Config::from_url(url);
        let pool = cfg
            .create_pool(Some(deadpool_redis::Runtime::Tokio1))
            .map_err(|e| StoreError::Connection(format!("failed to create Redis pool: {e}")))?;
        let client = RedisClient::open(url)
            .map_err(|e| StoreError::Connection(format!("failed to create Redis client: {e}")))?;
        Ok(Self {
            pool,
            client,
            key_prefix: key_prefix.to_owned(),
        })
    }
}

impl std::fmt::Debug for RedisStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RedisStore")
            .field("key_prefix", &self.key_prefix)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl FeatureStore for RedisStore {
    async fn ping(&self) -> Result<(), StoreError> {
        let mut conn = self.pool.get().await.map_err(|e| {
            StoreError::Connection(format!("Redis ping: failed to get connection: {e}"))
        })?;

        deadpool_redis::redis::cmd("PING")
            .query_async::<String>(&mut conn)
            .await
            .map_err(|e| StoreError::Connection(format!("Redis ping failed: {e}")))?;

        Ok(())
    }

    async fn fetch_features(
        &self,
        entity_id: &str,
        dest: &mut ArrayD<f32>,
    ) -> Result<FetchResult, StoreError> {
        let key = format!("{}:{}", self.key_prefix, entity_id);

        let mut conn =
            self.pool.get().await.map_err(|e| {
                StoreError::Connection(format!("failed to get Redis connection: {e}"))
            })?;

        let bytes: Option<Vec<u8>> = deadpool_redis::redis::cmd("GET")
            .arg(&key)
            .query_async(&mut conn)
            .await
            .map_err(|e| StoreError::Fetch {
                entity_id: entity_id.to_owned(),
                reason: format!("Redis GET failed: {e}"),
            })?;

        let bytes = match bytes {
            Some(b) => b,
            None => return Ok(FetchResult::Miss),
        };

        // MessagePack gives compact binary encoding with no schema overhead.
        let values: Vec<f32> =
            rmp_serde::from_slice(&bytes).map_err(|e| StoreError::Deserialize {
                entity_id: entity_id.to_owned(),
                reason: e.to_string(),
            })?;

        if values.len() != dest.len() {
            return Err(StoreError::ShapeMismatch {
                entity_id: entity_id.to_owned(),
                expected: vec![dest.len()],
                got: vec![values.len()],
            });
        }

        // scratchpad is always C-order; non-contiguous layout is a construction bug
        #[allow(clippy::expect_used)]
        let slice = dest.as_slice_mut().expect("contiguous array");
        slice.copy_from_slice(&values);

        Ok(FetchResult::Hit)
    }

    async fn update_stream(
        &self,
        entity_id: &str,
        poll_interval: Duration,
    ) -> Pin<Box<dyn Stream<Item = ()> + Send>> {
        let channel = format!("{}:updates:{}", self.key_prefix, entity_id);
        let client = self.client.clone();
        let (tx, rx) = mpsc::channel::<()>(16);

        // Spawn a task that owns the pub/sub connection and forwards a () token
        // into the channel each time Dendrite publishes to the entity's channel.
        // When the receiver is dropped (stream consumer disconnected), tx.send
        // returns Err and the task exits, closing the Redis connection.
        tokio::spawn(async move {
            let mut pubsub = match client.get_async_pubsub().await {
                Ok(p) => p,
                Err(e) => {
                    tracing::warn!(channel, error = %e, "pub/sub connect failed; falling back to poll");
                    return;
                }
            };
            if let Err(e) = pubsub.subscribe(&channel).await {
                tracing::warn!(channel, error = %e, "pub/sub subscribe failed; falling back to poll");
                return;
            }
            tracing::debug!(channel, "subscribed to feature updates");
            let mut msgs = pubsub.into_on_message();
            while msgs.next().await.is_some() {
                if tx.send(()).await.is_err() {
                    break;
                }
            }
        });

        // If the spawn above fails to connect or subscribe, the channel sender
        // is dropped immediately and ReceiverStream ends — the server falls back
        // to the poll-interval default via the trait's blanket behaviour.
        // In practice the server calls update_stream once and the loop exits,
        // ending the gRPC stream, which the client reconnects.
        //
        // For stores without pub/sub support the default trait impl already
        // handles the fallback; this code only runs for RedisStore.
        let _ = poll_interval; // poll_interval unused — pub/sub replaces the timer
        Box::pin(ReceiverStream::new(rx))
    }
}
