//! Delegates inference to the configured backend.

use std::sync::Arc;

use pipex::error::PipelineError;
use pipex::stage::Stage;

use crate::backend::Backend;
use crate::pipeline::InferenceScratchpad;
use crate::types::NamedTensorRef;

/// Packages the input tensor and runs it through the configured backend.
///
/// Passes the scratchpad's pre-allocated output buffers directly to the
/// backend, which writes into them in place — zero allocation on the hot path.
#[derive(Debug)]
pub struct InferStage {
    /// Shared reference to the inference backend (Triton or ONNX Runtime).
    pub backend: Arc<dyn Backend>,
    /// Name of the input tensor as defined in model_schema.inputs.
    pub input_name: String,
}

impl Stage<InferenceScratchpad> for InferStage {
    fn run(&mut self, ctx: &mut InferenceScratchpad) -> Result<(), PipelineError> {
        // Stack-allocated reference into ctx.input — no heap allocation.
        let inputs = [NamedTensorRef {
            name: &self.input_name,
            data: ctx.input.view(),
        }];

        // Backend::run is async. block_in_place parks the current thread so
        // Tokio can schedule other tasks while we block on the network call.
        // The backend writes directly into ctx.outputs (pre-allocated buffers),
        // so no allocation occurs on this path.
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current()
                .block_on(async { self.backend.run(&inputs, &mut ctx.outputs).await })
        })
        .map_err(|e| PipelineError::StageFailed(e.to_string()))?;

        Ok(())
    }
}
