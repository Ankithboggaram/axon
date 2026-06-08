//! ONNX Runtime inference backend, in-process with no network hop.

use async_trait::async_trait;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::TensorRef;
use parking_lot::Mutex;

use crate::backend::Backend;
use crate::error::BackendError;
use crate::types::{NamedTensorRef, OutputBuffer};

/// In-process inference backend powered by ONNX Runtime.
///
/// Sessions are pooled: each concurrent call to [`Backend::run`] pops a session
/// from the pool, runs inference with exclusive ownership, then returns it. When
/// all pool slots are in use an overflow session is created for that call only.
pub struct OnnxBackend {
    sessions: Mutex<Vec<Session>>,
    model_path: String,
    capacity: usize,
}

impl OnnxBackend {
    /// Loads `pool_size` ONNX sessions from disk and prepares them for inference.
    ///
    /// All sessions share the same model and optimization level. Graph optimization
    /// is applied once per session at load time so the cost is not paid on requests.
    pub fn new(model_path: &str, pool_size: usize) -> Result<Self, BackendError> {
        let capacity = pool_size.max(1);
        let mut sessions = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            sessions.push(Self::create_session(model_path)?);
        }
        Ok(Self {
            sessions: Mutex::new(sessions),
            model_path: model_path.to_owned(),
            capacity,
        })
    }

    fn create_session(model_path: &str) -> Result<Session, BackendError> {
        Session::builder()
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
            })
    }
}

impl std::fmt::Debug for OnnxBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxBackend")
            .field("capacity", &self.capacity)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl Backend for OnnxBackend {
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

        // Pop a session from the pool, releasing the lock before inference so
        // other threads can acquire their own session concurrently.
        let maybe_session = self.sessions.lock().pop();
        let mut session = match maybe_session {
            Some(s) => s,
            None => Self::create_session(&self.model_path)?,
        };

        // ort_outputs may borrow from session, so session must outlive it.
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
        // SessionOutputs borrows from session; drop it explicitly before returning
        // session to the pool so the borrow ends before the move.
        drop(ort_outputs);

        // Return session to pool if capacity has not been reached.
        let mut pool = self.sessions.lock();
        if pool.len() < self.capacity {
            pool.push(session);
        }

        Ok(())
    }
}
