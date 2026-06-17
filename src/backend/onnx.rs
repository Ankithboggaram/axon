//! ONNX Runtime inference backend, in-process with no network hop.
//!
//! [`OnnxBackend`] loads one or more ONNX Runtime sessions at startup and
//! pools them for concurrent inference. Each call to [`Backend::run`] pops a
//! session from the pool, runs inference exclusively on that session, then
//! returns it. If all pool slots are in use, an overflow session is created
//! on the fly for that call and discarded afterward.
//!
//! ## Graph optimization
//!
//! Every session is built with [`GraphOptimizationLevel::Level3`] (the maximum).
//! Optimization runs once at session creation and is not paid on requests.
//!
//! ## Execution providers
//!
//! The execution provider is selected by [`DeviceConfig`] at construction time:
//! `cpu` (default), `coreml`, `cuda`, or `tensorrt`. Non-CPU providers use
//! `error_on_failure()` to guarantee a clear startup error rather than a silent
//! fallback to CPU that would be invisible to operators.
//!
//! ## Output copy
//!
//! ORT owns its output buffers, so results are copied into the scratchpad's
//! pre-allocated [`OutputBuffer`] with `ndarray::assign`. For small outputs
//! (e.g. `[1, 1]` scores) this copy is negligible; for larger outputs
//! (embeddings, classification heads) it can be eliminated using ORT's
//! `IoBinding` API — tracked in the architectural backlog.
//!
//! [`DeviceConfig`]: crate::config::DeviceConfig
//! [`OutputBuffer`]: crate::types::OutputBuffer

use async_trait::async_trait;
use ort::ep;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::TensorRef;
use parking_lot::Mutex;

use crate::backend::Backend;
use crate::config::DeviceConfig;
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
    device: DeviceConfig,
    capacity: usize,
}

impl OnnxBackend {
    /// Loads `pool_size` ONNX sessions from disk and prepares them for inference.
    ///
    /// All sessions share the same model and optimization level. Graph optimization
    /// is applied once per session at load time so the cost is not paid on requests.
    ///
    /// Returns [`BackendError::SessionCreation`] if the requested device is not
    /// available on this host (e.g. `Cuda` without CUDA libraries installed).
    pub fn new(
        model_path: &str,
        pool_size: usize,
        device: DeviceConfig,
    ) -> Result<Self, BackendError> {
        let capacity = pool_size.max(1);
        let mut sessions = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            sessions.push(Self::create_session(model_path, &device)?);
        }
        Ok(Self {
            sessions: Mutex::new(sessions),
            model_path: model_path.to_owned(),
            device,
            capacity,
        })
    }

    fn create_session(model_path: &str, device: &DeviceConfig) -> Result<Session, BackendError> {
        let builder = Session::builder()
            .map_err(|e| {
                BackendError::SessionCreation(format!("failed to create session builder: {e}"))
            })?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| {
                BackendError::SessionCreation(format!("failed to set optimization level: {e}"))
            })?;

        // For non-CPU devices, error_on_failure() ensures a clear startup error
        // rather than a silent fallback to CPU that would be invisible to operators.
        let mut builder = match device {
            DeviceConfig::Cpu => builder,
            DeviceConfig::CoreMl => builder
                .with_execution_providers([ep::CoreML::default().build().error_on_failure()])
                .map_err(|e| {
                    BackendError::SessionCreation(format!("CoreML EP unavailable: {e}"))
                })?,
            DeviceConfig::Cuda => builder
                .with_execution_providers([ep::CUDA::default().build().error_on_failure()])
                .map_err(|e| BackendError::SessionCreation(format!("CUDA EP unavailable: {e}")))?,
            DeviceConfig::TensorRt => builder
                .with_execution_providers([ep::TensorRT::default().build().error_on_failure()])
                .map_err(|e| {
                    BackendError::SessionCreation(format!("TensorRT EP unavailable: {e}"))
                })?,
        };

        builder.commit_from_file(model_path).map_err(|e| {
            BackendError::SessionCreation(format!("failed to load model from {model_path}: {e}"))
        })
    }
}

impl std::fmt::Debug for OnnxBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxBackend")
            .field("device", &self.device)
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
            None => Self::create_session(&self.model_path, &self.device)?,
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

#[cfg(test)]
mod tests {
    use super::*;

    const MODEL: &str = "tests/fixtures/mnist-8.onnx";

    #[test]
    fn cpu_device_creates_session() {
        OnnxBackend::new(MODEL, 1, DeviceConfig::Cpu).unwrap();
    }

    #[test]
    fn pool_size_respected() {
        let backend = OnnxBackend::new(MODEL, 3, DeviceConfig::Cpu).unwrap();
        assert_eq!(backend.capacity, 3);
        assert_eq!(backend.sessions.lock().len(), 3);
    }

    #[test]
    fn pool_size_zero_clamps_to_one() {
        let backend = OnnxBackend::new(MODEL, 0, DeviceConfig::Cpu).unwrap();
        assert_eq!(backend.capacity, 1);
    }
}
