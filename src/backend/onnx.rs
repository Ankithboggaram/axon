//! ONNX Runtime inference backend, in-process with no network hop.

use std::sync::Mutex;

use async_trait::async_trait;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::TensorRef;

use crate::backend::Backend;
use crate::error::BackendError;
use crate::types::{NamedTensorRef, OutputBuffer};

/// In-process inference backend powered by ONNX Runtime.
///
/// The model is loaded and optimized once at startup. Each call to `run`
/// executes on the calling thread with no serialization or network overhead.
pub struct OnnxBackend {
    // Mutex because Session::run requires &mut self; the trait contract is &self.
    session: Mutex<Session>,
}

impl OnnxBackend {
    /// Loads an ONNX model from disk and prepares it for inference.
    ///
    /// Applies level-3 graph optimization at load time so the cost is paid
    /// once at startup rather than on the first request.
    pub fn new(model_path: &str) -> Result<Self, BackendError> {
        let session = Session::builder()
            .map_err(|e| {
                BackendError::SessionCreation(format!("failed to create session builder: {e}"))
            })?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| {
                BackendError::SessionCreation(format!("failed to set optimization level: {e}"))
            })?
            .commit_from_file(model_path)
            .map_err(|e| {
                BackendError::SessionCreation(format!(
                    "failed to load model from {model_path}: {e}"
                ))
            })?;
        Ok(Self {
            session: Mutex::new(session),
        })
    }
}

impl std::fmt::Debug for OnnxBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxBackend").finish_non_exhaustive()
    }
}

#[async_trait]
impl Backend for OnnxBackend {
    #[allow(clippy::unwrap_used)] // Mutex::lock() panics only on poison, which means a prior panic already occurred
    async fn run(
        &self,
        inputs: &[NamedTensorRef<'_>],
        outputs: &mut [OutputBuffer],
    ) -> Result<(), BackendError> {
        // TensorRef borrows from the scratchpad, so no copy on the input side.
        let ort_inputs = inputs
            .iter()
            .map(|t| -> Result<_, BackendError> {
                let tensor = TensorRef::from_array_view(t.data.view())
                    .map_err(|e| {
                        BackendError::InferenceFailed(format!("failed to create input tensor: {e}"))
                    })?
                    .into_dyn();
                Ok((t.name, tensor))
            })
            .collect::<Result<Vec<_>, BackendError>>()?;

        // Guard must outlive ort_outputs, which borrows from the session.
        let mut session = self.session.lock().unwrap();
        let ort_outputs = session
            .run(ort_inputs)
            .map_err(|e| BackendError::InferenceFailed(format!("inference failed: {e}")))?;

        // One copy per output: ORT owns its output buffers, so we copy into
        // the scratchpad's pre-allocated OutputBuffer for the rest of the pipeline.
        for (i, out_buf) in outputs.iter_mut().enumerate() {
            let view = ort_outputs[i].try_extract_array::<f32>().map_err(|e| {
                BackendError::InferenceFailed(format!("failed to extract output {i}: {e}"))
            })?;
            out_buf.data.assign(&view);
        }

        Ok(())
    }
}
