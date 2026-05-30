//! Redis feature store client.

use async_trait::async_trait;
use deadpool_redis::Pool;
use ndarray::ArrayD;

use crate::store::{FeatureStore, FetchResult};

pub struct RedisStore {
    pool: Pool,
    /// Key prefix applied to every entity lookup: `{prefix}:{entity_id}`.
    key_prefix: String,
}

impl RedisStore {
    /// Creates a new `RedisStore` connected to the given Redis URL.
    ///
    /// The pool is sized to the number of Tokio worker threads so each thread
    /// can hold a connection without contention.
    pub fn new(url: &str, key_prefix: &str) -> anyhow::Result<Self> {
        let cfg = deadpool_redis::Config::from_url(url);
        let pool = cfg
            .create_pool(Some(deadpool_redis::Runtime::Tokio1))
            .map_err(|e| anyhow::anyhow!("failed to create Redis pool: {e}"))?;
        Ok(Self {
            pool,
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
    async fn ping(&self) -> anyhow::Result<()> {
        let mut conn = self
            .pool
            .get()
            .await
            .map_err(|e| anyhow::anyhow!("Redis ping: failed to get connection: {e}"))?;

        deadpool_redis::redis::cmd("PING")
            .query_async::<String>(&mut conn)
            .await
            .map_err(|e| anyhow::anyhow!("Redis ping failed: {e}"))?;

        Ok(())
    }

    async fn fetch_features(
        &self,
        entity_id: &str,
        dest: &mut ArrayD<f32>,
    ) -> anyhow::Result<FetchResult> {
        let key = format!("{}:{}", self.key_prefix, entity_id);

        let mut conn = self
            .pool
            .get()
            .await
            .map_err(|e| anyhow::anyhow!("failed to get Redis connection: {e}"))?;

        let bytes: Option<Vec<u8>> = deadpool_redis::redis::cmd("GET")
            .arg(&key)
            .query_async(&mut conn)
            .await
            .map_err(|e| anyhow::anyhow!("Redis GET failed for key '{key}': {e}"))?;

        let bytes = match bytes {
            Some(b) => b,
            None => return Ok(FetchResult::Miss),
        };

        // MessagePack gives compact binary encoding with no schema overhead.
        let values: Vec<f32> = rmp_serde::from_slice(&bytes).map_err(|e| {
            anyhow::anyhow!("failed to deserialise features for '{entity_id}': {e}")
        })?;

        if values.len() != dest.len() {
            anyhow::bail!(
                "feature vector length mismatch for '{entity_id}': \
                 expected {}, got {}",
                dest.len(),
                values.len()
            );
        }

        dest.as_slice_mut()
            .expect("contiguous array")
            .copy_from_slice(&values);

        Ok(FetchResult::Hit)
    }
}
