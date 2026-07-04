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
//!    `cortex_contract::store::OnlineStoreReader`, and [`registry::ModelRegistryClient`]
//!    are traits. Adding a new backend (e.g. Triton) or a new feature store backend
//!    means writing one trait `impl` and adding a config enum variant; the pipeline,
//!    server, and pooling logic require no changes.
//!
//! # Architecture
//!
//! A request flows through four layers:
//!
//! - **gRPC server** (tonic, tokio multi-thread) — async I/O, feature store
//!   fetch, response dispatch. Never touches the model directly.
//! - **pipexec pipeline** — synchronous, zero-allocation stage chain. Stages
//!   mutate an `InferenceScratchpad` in place:
//!   `impute → validate → clip → normalize → infer → postprocess`
//! - **InferStage** — the async/sync boundary. Uses `block_in_place` to drive
//!   the backend future on the current thread without spawning a new task.
//! - **ONNX Runtime backend** — in-process inference with a session pool; one
//!   session per concurrent request, no serialisation under normal load.
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
pub mod predictions;
pub mod registry;
pub mod server;
pub mod types;
