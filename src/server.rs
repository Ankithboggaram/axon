//! gRPC service implementation and the async/sync handoff.
//!
//! [`InferenceServer`] implements the generated `InferenceService` trait for
//! both the unary (`Predict`) and server-streaming (`PredictStream`) RPCs.
//!
//! ## Async/sync boundary
//!
//! gRPC I/O runs on the tokio multi-thread runtime. The inference pipeline is
//! synchronous: stages mutate a scratchpad in place with no await points.
//! [`InferStage`][crate::pipeline::stages::infer::InferStage] is the only stage
//! that crosses back into async (to call the backend); it uses
//! `tokio::task::block_in_place` + `block_on` internally to drive the backend
//! future on the current thread without spawning a new task.
//!
//! The pattern is: async shell, sync core. All network I/O is async; all
//! computation is sync. The boundary is always inside `InferStage`, never in
//! the server layer.

use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use cortex_contract::PredictionRecord;
use cortex_contract::store::{FetchResult, OnlineStoreReader, RecordHeader};
use futures_util::StreamExt as _;
use pipexec::pool::ScratchpadPool;
use tokio_stream::Stream;
use tonic::{Request, Response, Status};
use tracing::{error, info, info_span, warn};

use crate::config::{FreshnessAction, FreshnessConfig};
use crate::metrics::Metrics;
use crate::pipeline::InferenceScratchpad;
use crate::pipeline::pool::PipelinePool;
use crate::predictions::PredictionSink;
use crate::proto::{
    OutputTensor, PredictRequest, PredictResponse, PredictStreamRequest, PredictStreamResponse,
    inference_service_server::InferenceService,
};

/// The gRPC service implementation.
///
/// All fields are cheaply cloneable (`Arc`s) so tonic can clone the server
/// per connection without duplicating the underlying state.
#[derive(Clone)]
pub struct InferenceServer {
    store: Arc<dyn OnlineStoreReader>,
    pipeline_pool: Arc<PipelinePool>,
    scratchpad_pool: Arc<ScratchpadPool<InferenceScratchpad>>,
    metrics: Arc<Metrics>,
    stream_poll_interval: Duration,
    /// Freshness enforcement settings. `None` disables enforcement (age is
    /// still recorded as a metric).
    freshness: Option<FreshnessConfig>,
    /// The model's trained `schema_version` to enforce served features
    /// against. `None` disables schema-version enforcement.
    expected_schema_version: Option<u32>,
    /// Closed-loop prediction logging. `None` disables it entirely (a single
    /// `is_none` check on the hot path, no channel or Kafka producer exists).
    predictions: Option<Arc<PredictionSink>>,
    /// Name of the currently served model, stamped into every `PredictionRecord`.
    model_name: String,
    /// Version of the currently served model, stamped into every `PredictionRecord`.
    model_version: String,
}

impl InferenceServer {
    /// Creates a new inference server wiring together all subsystems.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        store: Arc<dyn OnlineStoreReader>,
        pipeline_pool: Arc<PipelinePool>,
        scratchpad_pool: Arc<ScratchpadPool<InferenceScratchpad>>,
        metrics: Arc<Metrics>,
        stream_poll_interval_ms: u64,
        freshness: Option<FreshnessConfig>,
        expected_schema_version: Option<u32>,
        predictions: Option<Arc<PredictionSink>>,
        model_name: String,
        model_version: String,
    ) -> Self {
        Self {
            store,
            pipeline_pool,
            scratchpad_pool,
            metrics,
            stream_poll_interval: Duration::from_millis(stream_poll_interval_ms),
            freshness,
            expected_schema_version,
            predictions,
            model_name,
            model_version,
        }
    }

    /// Fetches features, runs the pipeline, and returns a response.
    ///
    /// If `inline_features` is non-empty, writes them directly into the
    /// scratchpad. Otherwise fetches from the feature store by entity_id.
    #[allow(clippy::expect_used)] // scratchpad is always C-order; non-contiguous layout is a construction bug
    async fn run_inference(
        &self,
        entity_id: &str,
        inline_features: &[f32],
    ) -> Result<PredictResponse, Status> {
        // Acquire a pre-allocated scratchpad. The pool resets it on return, so
        // all tensor buffers are already zeroed and metadata fields are cleared.
        let mut ctx = self.scratchpad_pool.acquire();

        ctx.entity_id
            .try_push_str(entity_id)
            .map_err(|_| Status::invalid_argument("entity_id exceeds maximum length"))?;
        ctx.timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        // Populated by a store fetch (Hit); stays default (zeroed) for inline
        // features or a Miss, which is reflected as-is in a logged PredictionRecord.
        let mut header = RecordHeader::default();

        if inline_features.is_empty() {
            let fetch_start = Instant::now();
            let result = self
                .store
                .fetch(
                    entity_id,
                    &mut header,
                    ctx.input.as_slice_mut().expect("contiguous array"),
                )
                .await
                .map_err(|e| {
                    error!(entity_id, error = %e, "feature store error");
                    Status::unavailable(format!("feature store error: {e}"))
                })?;

            self.metrics
                .store_fetch_duration_seconds
                .observe(fetch_start.elapsed().as_secs_f64());

            match result {
                FetchResult::Miss => {
                    self.metrics.store_misses_total.inc();
                    warn!(
                        entity_id,
                        "feature store miss; pipeline will run on zeroed input"
                    );
                }
                FetchResult::Hit => {
                    // Clock skew between Dendrite and Axon could otherwise make this negative.
                    let age_ms = (ctx.timestamp_ms - header.event_time_ms).max(0);
                    self.metrics
                        .served_feature_age_seconds
                        .observe(age_ms as f64 / 1000.0);

                    if let Some(freshness) = &self.freshness {
                        Self::enforce_freshness(freshness, age_ms, entity_id)?;
                    }

                    if let Some(expected) = self.expected_schema_version
                        && header.schema_version != expected
                    {
                        self.metrics.schema_version_rejects_total.inc();
                        Self::enforce_schema_version(expected, header.schema_version, entity_id)?;
                    }
                }
                // FetchResult is #[non_exhaustive]; treat any future variant like a miss.
                _ => {}
            }
        } else {
            if inline_features.len() != ctx.input.len() {
                return Err(Status::invalid_argument(format!(
                    "expected {} features, got {}",
                    ctx.input.len(),
                    inline_features.len()
                )));
            }
            ctx.input
                .as_slice_mut()
                .expect("contiguous array")
                .copy_from_slice(inline_features);
        }

        // Snapshot the exact served input before the pipeline transforms it in
        // place (impute/clip/normalize all mutate ctx.input). should_sample's
        // atomic increment is the only cost paid when predictions are enabled
        // but this particular request isn't sampled; the Vec clone only
        // happens for requests that will actually be logged.
        let sampled_prediction = self
            .predictions
            .as_ref()
            .filter(|sink| sink.should_sample())
            .map(|sink| {
                (
                    Arc::clone(sink),
                    ctx.input.iter().copied().collect::<Vec<f32>>(),
                )
            });

        // Acquire a pipeline, run it against the scratchpad, return it to pool on drop.
        self.pipeline_pool.acquire().run(&mut *ctx).map_err(|e| {
            error!(entity_id, error = %e, "pipeline error");
            Status::internal(format!("pipeline error: {e}"))
        })?;

        if let Some((sink, features)) = sampled_prediction {
            let output = ctx
                .outputs
                .first()
                .map(|out| out.data.iter().copied().collect())
                .unwrap_or_default();

            sink.emit(PredictionRecord {
                entity_id: entity_id.to_owned(),
                model_name: self.model_name.clone(),
                model_version: self.model_version.clone(),
                schema_version: header.schema_version,
                event_time_ms: header.event_time_ms,
                predict_time_ms: ctx.timestamp_ms,
                features,
                output,
                request_id: String::new(),
            });
        }

        let outputs = ctx
            .outputs
            .iter()
            .map(|out| OutputTensor {
                name: out.name.to_string(),
                values: out.data.iter().copied().collect(),
                shape: out.data.shape().iter().map(|&d| d as i64).collect(),
            })
            .collect();

        Ok(PredictResponse {
            entity_id: entity_id.to_owned(),
            outputs,
            timestamp_ms: ctx.timestamp_ms,
        })
    }

    /// Applies `[freshness]` enforcement to a served feature vector's age.
    ///
    /// Returns `Ok` if the vector is within `max_feature_age_ms`, or if it
    /// isn't but `on_stale` is [`FreshnessAction::Flag`] (a warning is logged
    /// instead). Returns `Err` only for [`FreshnessAction::Reject`] on a
    /// vector that exceeds the bound.
    #[allow(clippy::result_large_err)] // Status is axon's standard gRPC error type throughout this file
    fn enforce_freshness(
        freshness: &FreshnessConfig,
        age_ms: i64,
        entity_id: &str,
    ) -> Result<(), Status> {
        #[allow(clippy::cast_sign_loss)] // age_ms is clamped to >= 0 by the caller
        if (age_ms as u64) <= freshness.max_feature_age_ms {
            return Ok(());
        }

        match freshness.on_stale {
            FreshnessAction::Flag => {
                warn!(
                    entity_id,
                    age_ms,
                    max_feature_age_ms = freshness.max_feature_age_ms,
                    "served features exceed max age"
                );
                Ok(())
            }
            FreshnessAction::Reject => Err(Status::failed_precondition(format!(
                "features stale for entity '{entity_id}': age {age_ms}ms exceeds max {}ms",
                freshness.max_feature_age_ms
            ))),
        }
    }

    /// Applies schema-version enforcement: `Err` iff `got != expected`.
    #[allow(clippy::result_large_err)] // Status is axon's standard gRPC error type throughout this file
    fn enforce_schema_version(expected: u32, got: u32, entity_id: &str) -> Result<(), Status> {
        if got == expected {
            return Ok(());
        }
        Err(Status::failed_precondition(format!(
            "schema_version mismatch for entity '{entity_id}': model trained on {expected}, served record is {got}"
        )))
    }
}

#[tonic::async_trait]
impl InferenceService for InferenceServer {
    #[tracing::instrument(skip(self, request), fields(entity_id))]
    async fn predict(
        &self,
        request: Request<PredictRequest>,
    ) -> Result<Response<PredictResponse>, Status> {
        let req = request.into_inner();
        tracing::Span::current().record("entity_id", req.entity_id.as_str());

        let start = Instant::now();
        let result = self.run_inference(&req.entity_id, &req.features).await;
        let elapsed = start.elapsed().as_secs_f64();

        self.metrics
            .request_duration_seconds
            .with_label_values(&["predict"])
            .observe(elapsed);

        match result {
            Ok(response) => {
                self.metrics
                    .requests_total
                    .with_label_values(&["predict", "success"])
                    .inc();
                info!(
                    entity_id = req.entity_id,
                    latency_ms = elapsed * 1000.0,
                    "predict ok"
                );
                Ok(Response::new(response))
            }
            Err(e) => {
                self.metrics
                    .requests_total
                    .with_label_values(&["predict", "error"])
                    .inc();
                Err(e)
            }
        }
    }

    type PredictStreamStream =
        Pin<Box<dyn Stream<Item = Result<PredictStreamResponse, Status>> + Send>>;

    #[tracing::instrument(skip(self, request), fields(entity_id))]
    async fn predict_stream(
        &self,
        request: Request<PredictStreamRequest>,
    ) -> Result<Response<Self::PredictStreamStream>, Status> {
        let req = request.into_inner();
        tracing::Span::current().record("entity_id", req.entity_id.as_str());

        let entity_id = req.entity_id.clone();
        let server = self.clone();

        let stream = async_stream::try_stream! {
            let span = info_span!("predict_stream", entity_id = %entity_id);
            let _enter = span.enter();

            let mut updates = server.store.updates(&entity_id, server.stream_poll_interval).await;
            while let Some(()) = updates.next().await {
                let start = Instant::now();
                let result = server.run_inference(&entity_id, &[]).await;
                let elapsed = start.elapsed().as_secs_f64();

                server
                    .metrics
                    .request_duration_seconds
                    .with_label_values(&["predict_stream"])
                    .observe(elapsed);

                match result {
                    Ok(resp) => {
                        server
                            .metrics
                            .requests_total
                            .with_label_values(&["predict_stream", "success"])
                            .inc();
                        yield PredictStreamResponse {
                            entity_id: resp.entity_id,
                            outputs: resp.outputs,
                            timestamp_ms: resp.timestamp_ms,
                        };
                    }
                    Err(e) => {
                        server
                            .metrics
                            .requests_total
                            .with_label_values(&["predict_stream", "error"])
                            .inc();
                        error!(entity_id = %entity_id, error = %e, "predict_stream error");
                        Err(e)?;
                    }
                }
            }
        };

        Ok(Response::new(Box::pin(stream)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn freshness(max_feature_age_ms: u64, on_stale: FreshnessAction) -> FreshnessConfig {
        FreshnessConfig {
            max_feature_age_ms,
            on_stale,
        }
    }

    #[test]
    fn freshness_within_bound_is_ok() {
        let cfg = freshness(1000, FreshnessAction::Reject);
        assert!(InferenceServer::enforce_freshness(&cfg, 500, "e1").is_ok());
    }

    #[test]
    fn freshness_at_exact_bound_is_ok() {
        let cfg = freshness(1000, FreshnessAction::Reject);
        assert!(InferenceServer::enforce_freshness(&cfg, 1000, "e1").is_ok());
    }

    #[test]
    fn freshness_flag_never_errors_even_when_stale() {
        let cfg = freshness(100, FreshnessAction::Flag);
        assert!(InferenceServer::enforce_freshness(&cfg, 5000, "e1").is_ok());
    }

    #[test]
    fn freshness_reject_errors_when_stale() {
        let cfg = freshness(100, FreshnessAction::Reject);
        let err = InferenceServer::enforce_freshness(&cfg, 5000, "e1").unwrap_err();
        assert_eq!(err.code(), tonic::Code::FailedPrecondition);
        assert!(err.message().contains("e1"));
    }

    #[test]
    fn schema_version_match_is_ok() {
        assert!(InferenceServer::enforce_schema_version(1, 1, "e1").is_ok());
    }

    #[test]
    fn schema_version_mismatch_errors() {
        let err = InferenceServer::enforce_schema_version(1, 2, "e1").unwrap_err();
        assert_eq!(err.code(), tonic::Code::FailedPrecondition);
        assert!(err.message().contains("e1"));
    }
}
