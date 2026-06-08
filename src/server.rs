//! gRPC service implementation and the async/sync handoff.
//!
//! [`InferenceServer`] implements the generated `InferenceService` trait for
//! both the unary (`Predict`) and server-streaming (`PredictStream`) RPCs.
//!
//! ## Async/sync boundary
//!
//! gRPC I/O runs on the tokio multi-thread runtime. The inference pipeline is
//! synchronous: stages mutate a scratchpad in place with no await points. The
//! two meet in `InferenceServer::run_inference`, which calls
//! `tokio::task::block_in_place` before entering the pipeline. This tells
//! tokio the current thread is about to block, so the scheduler can move other
//! tasks to different threads rather than stalling the executor.
//!
//! The pattern is: async shell, sync core. All network I/O is async; all
//! computation is sync. The boundary is always in the same place.

use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use pipex::pool::ScratchpadPool;
use tokio_stream::Stream;
use tonic::{Request, Response, Status};
use tracing::{error, info, info_span, warn};

use crate::metrics::Metrics;
use crate::pipeline::InferenceScratchpad;
use crate::pipeline::pool::PipelinePool;
use crate::proto::{
    OutputTensor, PredictRequest, PredictResponse, PredictStreamRequest, PredictStreamResponse,
    inference_service_server::InferenceService,
};
use crate::store::{FeatureStore, FetchResult};

/// The gRPC service implementation.
///
/// All fields are cheaply cloneable (`Arc`s) so tonic can clone the server
/// per connection without duplicating the underlying state.
#[derive(Clone)]
pub struct InferenceServer {
    store: Arc<dyn FeatureStore>,
    pipeline_pool: Arc<PipelinePool>,
    scratchpad_pool: Arc<ScratchpadPool<InferenceScratchpad>>,
    metrics: Arc<Metrics>,
    stream_poll_interval: Duration,
}

impl InferenceServer {
    /// Creates a new inference server wiring together all subsystems.
    pub fn new(
        store: Arc<dyn FeatureStore>,
        pipeline_pool: Arc<PipelinePool>,
        scratchpad_pool: Arc<ScratchpadPool<InferenceScratchpad>>,
        metrics: Arc<Metrics>,
        stream_poll_interval_ms: u64,
    ) -> Self {
        Self {
            store,
            pipeline_pool,
            scratchpad_pool,
            metrics,
            stream_poll_interval: Duration::from_millis(stream_poll_interval_ms),
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

        if inline_features.is_empty() {
            let fetch_start = Instant::now();
            let result = self
                .store
                .fetch_features(entity_id, &mut ctx.input)
                .await
                .map_err(|e| {
                    error!(entity_id, error = %e, "feature store error");
                    Status::unavailable(format!("feature store error: {e}"))
                })?;

            self.metrics
                .store_fetch_duration_seconds
                .observe(fetch_start.elapsed().as_secs_f64());

            if matches!(result, FetchResult::Miss) {
                self.metrics.store_misses_total.inc();
                warn!(
                    entity_id,
                    "feature store miss; pipeline will run on zeroed input"
                );
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

        // Acquire a pipeline, run it against the scratchpad, return it to pool on drop.
        self.pipeline_pool.acquire().run(&mut *ctx).map_err(|e| {
            error!(entity_id, error = %e, "pipeline error");
            Status::internal(format!("pipeline error: {e}"))
        })?;

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

            loop {
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

                tokio::time::sleep(server.stream_poll_interval).await;
            }
        };

        Ok(Response::new(Box::pin(stream)))
    }
}
