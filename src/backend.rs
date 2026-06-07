//! Inference backend trait and implementations.
//!
//! A [`Backend`] is anything that accepts named input tensors and writes
//! prediction results into pre-allocated output buffers. It is the single
//! extension point for adding new inference runtimes without touching the
//! pipeline, server, or config logic.
//!
//! ## Threading model
//!
//! Backends must be `Send + Sync`: one instance is shared across all concurrent
//! requests via `Arc<dyn Backend>`. [`Backend::run`] takes `&self` so multiple
//! threads can call it simultaneously without any external locking.
//!
//! At the call site in [`crate::pipeline::stages::infer::InferStage`], `run` is
//! bridged from the synchronous pipeline into the async runtime via
//! `tokio::task::block_in_place`. This keeps inference off the async executor
//! without spawning a new thread per request.
//!
//! ## Implementations
//!
//! - [`onnx::OnnxBackend`]: in-process ONNX Runtime; the current default
//! - Triton: planned; see [`packaging`] for config generation helpers

use async_trait::async_trait;

use crate::error::BackendError;
use crate::types::{NamedTensorRef, OutputBuffer};

pub mod onnx;
pub mod packaging;

#[async_trait]
pub trait Backend: std::fmt::Debug + Send + Sync {
    /// Runs model inference on the given named input tensors and writes
    /// results into the pre-allocated output buffers in place.
    async fn run(
        &self,
        inputs: &[NamedTensorRef<'_>],
        outputs: &mut [OutputBuffer],
    ) -> Result<(), BackendError>;
}
