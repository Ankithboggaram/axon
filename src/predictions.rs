//! Closed-loop `PredictionRecord` emission to Kafka: async, fire-and-forget,
//! off the serving hot path
//!
//! # Performance contract
//!
//! The hot path (inside `run_inference`) only ever does:
//! 1. one atomic increment to decide whether this request is sampled, and
//! 2. at most one non-blocking [`mpsc::Sender::try_send`].
//!
//! Protobuf encoding, the Kafka client call, and any
//! backpressure from the broker happens in a background task that owns the
//! [`FutureProducer`] and drains the channel. If the channel is full (the
//! background task can't keep up, or Kafka itself is slow/unreachable), the
//! record is dropped and counted; the response is never blocked or slowed.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use cortex_contract::PredictionRecord;
use prost::Message as _;
use rdkafka::ClientConfig;
use rdkafka::producer::{FutureProducer, FutureRecord};
use tokio::sync::mpsc;
use tracing::warn;

use crate::config::PredictionsConfig;
use crate::error::PredictionsError;
use crate::metrics::Metrics;

/// Bounded channel capacity between the hot path and the Kafka drain task.
///
/// Sized to absorb an ordinary burst of Kafka latency without dropping;
/// beyond this, backpressure is the point — better to drop prediction logs
/// than to let a slow broker degrade serving.
const CHANNEL_CAPACITY: usize = 4096;

/// How long a single publish may wait for a slot in librdkafka's internal
/// produce queue before giving up. This bounds a single background task, not
/// the hot path.
const QUEUE_TIMEOUT: Duration = Duration::from_secs(5);

/// Hot-path handle for emitting `PredictionRecord`s.
///
/// Cheap to hold behind an `Arc` (as [`crate::server::InferenceServer`]
/// does): the sender is a lightweight channel handle and the counter is a
/// single atomic.
pub struct PredictionSink {
    tx: mpsc::Sender<PredictionRecord>,
    /// Emit every Nth record; see [`sample_rate_to_every_n`].
    sample_every_n: u64,
    counter: AtomicU64,
    metrics: Arc<Metrics>,
}

impl PredictionSink {
    /// Builds a sink and spawns its background Kafka-draining task.
    ///
    /// Does not connect eagerly: `rdkafka` connects lazily on first send, so
    /// this only fails on malformed config, not on broker unavailability.
    ///
    /// # Errors
    /// [`PredictionsError::ProducerCreation`] if the `rdkafka` client config
    /// itself is invalid.
    pub fn new(
        config: &PredictionsConfig,
        metrics: Arc<Metrics>,
    ) -> Result<Self, PredictionsError> {
        let producer: FutureProducer = ClientConfig::new()
            .set("bootstrap.servers", &config.brokers)
            .create()
            .map_err(|e| PredictionsError::ProducerCreation(e.to_string()))?;

        let (tx, rx) = mpsc::channel(CHANNEL_CAPACITY);
        tokio::spawn(drain(rx, producer, config.topic.clone()));

        Ok(Self {
            tx,
            sample_every_n: sample_rate_to_every_n(config.sample_rate),
            counter: AtomicU64::new(0),
            metrics,
        })
    }

    /// Returns `true` if the caller should build and pass a `PredictionRecord`
    /// to [`Self::emit`] for this request.
    ///
    /// Split from `emit` so callers can skip building the record entirely
    /// (which requires copying the served feature vector) for unsampled
    /// requests, rather than building it and throwing it away.
    #[must_use]
    pub fn should_sample(&self) -> bool {
        let n = self.counter.fetch_add(1, Ordering::Relaxed);
        n % self.sample_every_n == 0
    }

    /// Enqueues `record` for async emission. Never blocks: a full channel
    /// results in the record being dropped and counted.
    ///
    /// Callers should only invoke this after [`Self::should_sample`] returned
    /// `true` for the same request.
    pub fn emit(&self, record: PredictionRecord) {
        if self.tx.try_send(record).is_ok() {
            self.metrics.predictions_emitted_total.inc();
        } else {
            self.metrics.predictions_dropped_total.inc();
        }
    }
}

/// Converts a `0.0..=1.0` sample rate into "emit every Nth record".
///
/// Deterministic subsampling (a counter, not an RNG) keeps the hot-path
/// check to a single atomic increment and needs no RNG dependency, which
/// axon otherwise has no use for.
fn sample_rate_to_every_n(sample_rate: f32) -> u64 {
    if sample_rate >= 1.0 {
        1
    } else if sample_rate <= 0.0 {
        u64::MAX
    } else {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let n = (1.0 / sample_rate).round() as u64;
        n.max(1)
    }
}

/// Owns the Kafka producer and drains the channel, publishing each record.
///
/// Each publish is spawned as its own task rather than awaited in this loop,
/// so one slow delivery can't stall the drain and back the channel up toward
/// the hot path's `try_send`. `FutureProducer` is cheap to clone (an `Arc`
/// internally to the underlying `rdkafka` client).
async fn drain(mut rx: mpsc::Receiver<PredictionRecord>, producer: FutureProducer, topic: String) {
    let topic: Arc<str> = topic.into();
    while let Some(record) = rx.recv().await {
        let producer = producer.clone();
        let topic = Arc::clone(&topic);
        tokio::spawn(async move {
            let payload = record.encode_to_vec();
            let key = record.entity_id.as_str();
            let publish = FutureRecord::to(&topic).key(key).payload(&payload);
            if let Err((e, _msg)) = producer.send(publish, QUEUE_TIMEOUT).await {
                warn!(error = %e, "failed to publish PredictionRecord to Kafka");
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_rate_one_emits_every_record() {
        assert_eq!(sample_rate_to_every_n(1.0), 1);
    }

    #[test]
    fn sample_rate_above_one_clamps_to_every_record() {
        assert_eq!(sample_rate_to_every_n(2.0), 1);
    }

    #[test]
    fn sample_rate_zero_never_emits() {
        assert_eq!(sample_rate_to_every_n(0.0), u64::MAX);
    }

    #[test]
    fn sample_rate_negative_never_emits() {
        assert_eq!(sample_rate_to_every_n(-1.0), u64::MAX);
    }

    #[test]
    fn sample_rate_half_emits_every_other_record() {
        assert_eq!(sample_rate_to_every_n(0.5), 2);
    }

    #[test]
    fn sample_rate_tenth_emits_every_tenth_record() {
        assert_eq!(sample_rate_to_every_n(0.1), 10);
    }

    fn test_config(sample_rate: f32) -> PredictionsConfig {
        PredictionsConfig {
            enabled: true,
            // Unreachable but syntactically valid; rdkafka connects lazily,
            // so construction succeeds and sends only fail asynchronously in
            // the background drain task, never on the hot path.
            brokers: "localhost:1".to_owned(),
            topic: "predictions-test".to_owned(),
            sample_rate,
        }
    }

    fn sample_record(entity_id: &str) -> PredictionRecord {
        PredictionRecord {
            entity_id: entity_id.to_owned(),
            model_name: "test-model".to_owned(),
            model_version: "1".to_owned(),
            schema_version: 1,
            event_time_ms: 0,
            predict_time_ms: 0,
            features: vec![0.1, 0.2],
            output: vec![0.9],
            request_id: String::new(),
        }
    }

    #[tokio::test]
    async fn emit_is_non_blocking_and_counts_as_emitted_even_when_broker_unreachable() {
        let metrics = Arc::new(Metrics::new(Vec::new()).expect("metrics registration"));
        let sink = PredictionSink::new(&test_config(1.0), Arc::clone(&metrics))
            .expect("producer construction is lazy; must not fail for an unreachable broker");

        for i in 0..10 {
            assert!(sink.should_sample());
            sink.emit(sample_record(&format!("e{i}")));
        }

        assert_eq!(metrics.predictions_emitted_total.get(), 10.0);
        assert_eq!(metrics.predictions_dropped_total.get(), 0.0);
    }

    #[tokio::test]
    async fn full_channel_drops_and_counts_instead_of_blocking() {
        let metrics = Arc::new(Metrics::new(Vec::new()).expect("metrics registration"));
        let sink = PredictionSink::new(&test_config(1.0), Arc::clone(&metrics))
            .expect("producer construction is lazy");

        // Far more than CHANNEL_CAPACITY, sent without yielding to the
        // runtime, so the background drain task has no chance to make room.
        for i in 0..(CHANNEL_CAPACITY * 2) {
            sink.emit(sample_record(&format!("e{i}")));
        }

        let dropped = metrics.predictions_dropped_total.get();
        assert!(dropped > 0.0, "expected some drops once the channel filled");
        assert_eq!(
            metrics.predictions_emitted_total.get() + dropped,
            (CHANNEL_CAPACITY * 2) as f64
        );
    }

    #[tokio::test]
    async fn should_sample_respects_sample_rate() {
        let metrics = Arc::new(Metrics::new(Vec::new()).expect("metrics registration"));
        let sink = PredictionSink::new(&test_config(0.5), metrics).expect("producer construction");

        let sampled = (0..10).filter(|_| sink.should_sample()).count();
        assert_eq!(sampled, 5);
    }
}
