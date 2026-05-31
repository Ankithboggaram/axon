//! gRPC server: unary and streaming RPC handlers.

use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use tokio::sync::Mutex;

use pipex::dynamic_pipeline::Pipeline;
use pipex::scratchpad::Scratchpad;
use tokio_stream::Stream;
use tonic::{Request, Response, Status};
use tracing::{error, info, info_span, warn};

use crate::metrics::Metrics;
use crate::pipeline::InferenceScratchpad;
use crate::proto::{
    OutputTensor, PredictRequest, PredictResponse, PredictStreamRequest, PredictStreamResponse,
    inference_service_server::InferenceService,
};
use crate::store::{FeatureStore, FetchResult};

/// Bundles the pipeline and its scratchpad together under one Mutex.
///
/// Both require exclusive access during a request. Keeping them together
/// avoids the possibility of acquiring them in different orders across call
/// sites, which would risk deadlock.
struct PipelineBundle {
    pipeline: Pipeline<InferenceScratchpad>,
    scratchpad: InferenceScratchpad,
}

/// The gRPC service implementation.
///
/// All fields are cheaply cloneable (`Arc`s) so tonic can clone the server
/// per connection without duplicating the underlying state.
#[derive(Clone)]
pub struct InferenceServer {
    store: Arc<dyn FeatureStore>,
    bundle: Arc<Mutex<PipelineBundle>>,
    metrics: Arc<Metrics>,
    stream_poll_interval: Duration,
}

impl InferenceServer {
    pub fn new(
        store: Arc<dyn FeatureStore>,
        pipeline: Pipeline<InferenceScratchpad>,
        scratchpad: InferenceScratchpad,
        metrics: Arc<Metrics>,
        stream_poll_interval_ms: u64,
    ) -> Self {
        Self {
            store,
            bundle: Arc::new(Mutex::new(PipelineBundle {
                pipeline,
                scratchpad,
            })),
            metrics,
            stream_poll_interval: Duration::from_millis(stream_poll_interval_ms),
        }
    }

    /// Fetches features, runs the pipeline, and returns a response.
    ///
    /// If `inline_features` is non-empty, writes them directly into the
    /// scratchpad. Otherwise fetches from the feature store by entity_id.
    async fn run_inference(
        &self,
        entity_id: &str,
        inline_features: &[f32],
    ) -> Result<PredictResponse, Status> {
        let mut bundle = self.bundle.lock().await;
        let PipelineBundle {
            ref mut pipeline,
            ref mut scratchpad,
        } = *bundle;

        scratchpad.reset();
        scratchpad
            .entity_id
            .try_push_str(entity_id)
            .map_err(|_| Status::invalid_argument("entity_id exceeds maximum length"))?;
        scratchpad.timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        if inline_features.is_empty() {
            let fetch_start = Instant::now();
            let result = self
                .store
                .fetch_features(entity_id, &mut scratchpad.input)
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
            if inline_features.len() != scratchpad.input.len() {
                return Err(Status::invalid_argument(format!(
                    "expected {} features, got {}",
                    scratchpad.input.len(),
                    inline_features.len()
                )));
            }
            scratchpad
                .input
                .as_slice_mut()
                .expect("contiguous array")
                .copy_from_slice(inline_features);
        }

        pipeline.run(scratchpad).map_err(|e| {
            error!(entity_id, error = %e, "pipeline error");
            Status::internal(format!("pipeline error: {e}"))
        })?;

        let outputs = scratchpad
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
            timestamp_ms: scratchpad.timestamp_ms,
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
