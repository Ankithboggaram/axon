//! Prometheus metrics registry and HTTP exposition.

use std::sync::Arc;

use pipex::metrics::StageMetrics;
use prometheus::{
    Counter, CounterVec, GaugeVec, Histogram, HistogramOpts, HistogramVec, Opts, Registry,
    TextEncoder,
};

use crate::error::ServeError;

/// Holds all Prometheus metric handles for the axon application.
///
/// Construct once at startup and share via `Arc<Metrics>`. The individual
/// metric types are internally thread-safe, so no external locking required.
pub struct Metrics {
    /// Prometheus registry that all metrics are registered with.
    pub registry: Registry,

    /// Total inference requests, labelled by RPC name and outcome.
    pub requests_total: CounterVec,

    /// End-to-end request latency from gRPC receive to response sent.
    pub request_duration_seconds: HistogramVec,

    /// Redis feature fetch latency.
    pub store_fetch_duration_seconds: Histogram,

    /// Total feature store lookups that returned no entry for the entity.
    pub store_misses_total: Counter,

    // Per-stage latency and error rate snapshots, refreshed on each scrape.
    stage_p99_ns: GaugeVec,
    stage_p999_ns: GaugeVec,
    stage_count_total: GaugeVec,
    stage_error_rate: GaugeVec,
    stage_handles: Vec<Arc<StageMetrics>>,
}

impl Metrics {
    /// Registers all metrics into a new Prometheus registry.
    pub fn new(stage_handles: Vec<Arc<StageMetrics>>) -> Result<Self, ServeError> {
        let registry = Registry::new();

        let requests_total = CounterVec::new(
            Opts::new(
                "axon_requests_total",
                "Total inference requests by RPC and status.",
            ),
            &["rpc", "status"],
        )
        .map_err(|e| ServeError::MetricsRegistration {
            name: "axon_requests_total",
            reason: e.to_string(),
        })?;

        let request_duration_seconds = HistogramVec::new(
            HistogramOpts::new(
                "axon_request_duration_seconds",
                "End-to-end request latency in seconds.",
            )
            .buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]),
            &["rpc"],
        )
        .map_err(|e| ServeError::MetricsRegistration {
            name: "axon_request_duration_seconds",
            reason: e.to_string(),
        })?;

        let store_fetch_duration_seconds = Histogram::with_opts(
            HistogramOpts::new(
                "axon_store_fetch_duration_seconds",
                "Redis feature fetch latency in seconds.",
            )
            .buckets(vec![0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]),
        )
        .map_err(|e| ServeError::MetricsRegistration {
            name: "axon_store_fetch_duration_seconds",
            reason: e.to_string(),
        })?;

        let store_misses_total = Counter::new(
            "axon_store_misses_total",
            "Total feature store lookups that returned no entry.",
        )
        .map_err(|e| ServeError::MetricsRegistration {
            name: "axon_store_misses_total",
            reason: e.to_string(),
        })?;

        let stage_p99_ns = GaugeVec::new(
            Opts::new("axon_stage_p99_ns", "Stage p99 latency in nanoseconds."),
            &["stage"],
        )
        .map_err(|e| ServeError::MetricsRegistration {
            name: "axon_stage_p99_ns",
            reason: e.to_string(),
        })?;

        let stage_p999_ns = GaugeVec::new(
            Opts::new("axon_stage_p999_ns", "Stage p999 latency in nanoseconds."),
            &["stage"],
        )
        .map_err(|e| ServeError::MetricsRegistration {
            name: "axon_stage_p999_ns",
            reason: e.to_string(),
        })?;

        let stage_count_total = GaugeVec::new(
            Opts::new("axon_stage_count_total", "Total stage executions."),
            &["stage"],
        )
        .map_err(|e| ServeError::MetricsRegistration {
            name: "axon_stage_count_total",
            reason: e.to_string(),
        })?;

        let stage_error_rate = GaugeVec::new(
            Opts::new(
                "axon_stage_error_rate",
                "Stage error rate as a fraction between 0.0 and 1.0.",
            ),
            &["stage"],
        )
        .map_err(|e| ServeError::MetricsRegistration {
            name: "axon_stage_error_rate",
            reason: e.to_string(),
        })?;

        registry
            .register(Box::new(requests_total.clone()))
            .map_err(|e| ServeError::MetricsRegistration {
                name: "axon_requests_total",
                reason: e.to_string(),
            })?;
        registry
            .register(Box::new(request_duration_seconds.clone()))
            .map_err(|e| ServeError::MetricsRegistration {
                name: "axon_request_duration_seconds",
                reason: e.to_string(),
            })?;
        registry
            .register(Box::new(store_fetch_duration_seconds.clone()))
            .map_err(|e| ServeError::MetricsRegistration {
                name: "axon_store_fetch_duration_seconds",
                reason: e.to_string(),
            })?;
        registry
            .register(Box::new(store_misses_total.clone()))
            .map_err(|e| ServeError::MetricsRegistration {
                name: "axon_store_misses_total",
                reason: e.to_string(),
            })?;
        registry
            .register(Box::new(stage_p99_ns.clone()))
            .map_err(|e| ServeError::MetricsRegistration {
                name: "axon_stage_p99_ns",
                reason: e.to_string(),
            })?;
        registry
            .register(Box::new(stage_p999_ns.clone()))
            .map_err(|e| ServeError::MetricsRegistration {
                name: "axon_stage_p999_ns",
                reason: e.to_string(),
            })?;
        registry
            .register(Box::new(stage_count_total.clone()))
            .map_err(|e| ServeError::MetricsRegistration {
                name: "axon_stage_count_total",
                reason: e.to_string(),
            })?;
        registry
            .register(Box::new(stage_error_rate.clone()))
            .map_err(|e| ServeError::MetricsRegistration {
                name: "axon_stage_error_rate",
                reason: e.to_string(),
            })?;

        Ok(Self {
            registry,
            requests_total,
            request_duration_seconds,
            store_fetch_duration_seconds,
            store_misses_total,
            stage_p99_ns,
            stage_p999_ns,
            stage_count_total,
            stage_error_rate,
            stage_handles,
        })
    }

    /// Renders all metrics in Prometheus text format.
    ///
    /// Refreshes stage gauge values from the latest per-stage snapshots immediately
    /// before encoding, so Prometheus always receives up-to-date values without
    /// requiring a background task.
    pub fn render(&self) -> Result<String, ServeError> {
        self.refresh_stage_snapshots();
        let families = self.registry.gather();
        TextEncoder::new()
            .encode_to_string(&families)
            .map_err(|e| ServeError::MetricsEncoding(e.to_string()))
    }

    /// Pulls the latest snapshot from each stage metrics handle and updates gauges.
    fn refresh_stage_snapshots(&self) {
        for sm in &self.stage_handles {
            let snap = sm.snapshot();
            let label = snap.label.as_str();
            self.stage_p99_ns
                .with_label_values(&[label])
                .set(snap.p99_ns as f64);
            self.stage_p999_ns
                .with_label_values(&[label])
                .set(snap.p999_ns as f64);
            self.stage_count_total
                .with_label_values(&[label])
                .set(snap.count as f64);
            self.stage_error_rate
                .with_label_values(&[label])
                .set(snap.error_rate);
        }
    }
}

impl std::fmt::Debug for Metrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Metrics")
            .field("stage_count", &self.stage_handles.len())
            .finish_non_exhaustive()
    }
}
