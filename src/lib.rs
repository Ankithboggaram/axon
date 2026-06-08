//! Axon is a configuration-driven ML inference server for real-time model serving.
//!
//! # Design
//!
//! Axon is built around three ideas:
//!
//! 1. **Config drives the pipeline.** `config.toml` is the only place where stage
//!    order, preprocessing parameters, and observability flags live. No code changes
//!    are needed to add a clip stage, tune a normalisation mean, or enable per-stage
//!    timing. The running server is a direct, faithful translation of that file.
//!
//! 2. **Hot path allocates nothing.** The scratchpad and ONNX session are both
//!    pre-allocated at startup and pooled across requests. Each request acquires a
//!    scratchpad and a pipeline from their respective pools, runs all stages against
//!    the scratchpad in place, then returns both on drop. No heap allocation occurs
//!    between request receipt and response dispatch.
//!
//! 3. **Extensibility via traits, not config flags.** [`backend::Backend`],
//!    [`store::FeatureStore`], and [`registry::ModelRegistryClient`] are traits.
//!    Adding a new backend (e.g. Triton) or a new feature store (e.g. Feast) means
//!    writing one trait `impl` and adding a config enum variant; the pipeline,
//!    server, and pooling logic require no changes.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────┐
//! │  gRPC server  (tonic, tokio multi-thread)       │
//! │  async: I/O, store fetch, response dispatch     │
//! ├─────────────────────────────────────────────────┤
//! │  InferStage   (block_in_place boundary)         │
//! ├─────────────────────────────────────────────────┤
//! │  pipex pipeline  (sync, zero-allocation stages) │
//! │  impute → validate → clip → normalize → infer   │
//! │                          → postprocess          │
//! ├─────────────────────────────────────────────────┤
//! │  ONNX Runtime backend  (session pool)           │
//! └─────────────────────────────────────────────────┘
//! ```
//!
//! **This server intentionally does not provide:** model training, batch
//! inference, model versioning, A/B routing, or feature engineering. Those
//! belong in Dendrite (feature pipeline) and Synapse (training pipeline).
//! Axon serves one model, one entity, one request at a time — and does it fast.

#![deny(clippy::unwrap_used, clippy::expect_used)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
#![warn(missing_docs)]

pub mod backend;
pub mod config;
pub mod error;
#[allow(missing_docs)]
pub mod proto {
    tonic::include_proto!("axon.inference.v1");
}
pub mod metrics;
pub mod pipeline;
pub mod registry;
pub mod server;
pub mod store;
pub mod types;
