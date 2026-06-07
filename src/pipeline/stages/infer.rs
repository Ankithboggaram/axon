//! Delegates inference to the configured backend.

use std::sync::Arc;

use arrayvec::ArrayString;
use pipex::error::PipelineError;
use pipex::stage::Stage;

use crate::backend::Backend;
use crate::pipeline::InferenceScratchpad;
use crate::types::{MAX_TENSOR_NAME_LEN, NamedTensorRef};

/// Packages the input tensor and runs it through the configured backend.
///
/// Passes the scratchpad's pre-allocated output buffers directly to the
/// backend, which writes into them in place; zero allocation on the hot path.
#[derive(Debug)]
pub struct InferStage {
    /// Shared reference to the configured inference backend.
    pub backend: Arc<dyn Backend>,
    /// Name of the input tensor as defined in model_schema.inputs.
    pub input_name: ArrayString<MAX_TENSOR_NAME_LEN>,
}

impl Stage<InferenceScratchpad> for InferStage {
    #[inline]
    fn run(&mut self, ctx: &mut InferenceScratchpad) -> Result<(), PipelineError> {
        let inputs = [NamedTensorRef {
            name: &self.input_name,
            data: ctx.input.view(),
        }];

        // block_in_place bridges the sync Stage trait with the async Backend trait.
        // It tells Tokio this thread is about to block so the scheduler can use other threads.
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current()
                .block_on(async { self.backend.run(&inputs, &mut ctx.outputs).await })
        })
        .map_err(|e| PipelineError::StageFailed {
            stage: "InferStage",
            message: e.to_string(),
        })?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrayvec::ArrayString;
    use async_trait::async_trait;
    use ndarray::{ArrayD, IxDyn};
    use pipex::error::PipelineError;
    use pipex::stage::Stage;

    use super::*;
    use crate::backend::Backend;
    use crate::error::BackendError;
    use crate::pipeline::InferenceScratchpad;
    use crate::types::{NamedTensorRef, OutputBuffer};

    fn ctx_with_output(output_shape: &[usize]) -> InferenceScratchpad {
        InferenceScratchpad {
            entity_id: ArrayString::new(),
            request_id: ArrayString::new(),
            timestamp_ms: 0,
            input: ArrayD::zeros(IxDyn(&[3])),
            outputs: vec![OutputBuffer {
                name: "output".parse().unwrap(),
                data: ArrayD::zeros(IxDyn(output_shape)),
            }]
            .into_boxed_slice(),
        }
    }

    #[derive(Debug)]
    struct MockBackend {
        output_value: f32,
    }

    #[async_trait]
    impl Backend for MockBackend {
        async fn run(
            &self,
            _inputs: &[NamedTensorRef<'_>],
            outputs: &mut [OutputBuffer],
        ) -> Result<(), BackendError> {
            for out in outputs.iter_mut() {
                out.data.fill(self.output_value);
            }
            Ok(())
        }
    }

    #[derive(Debug)]
    struct FailingBackend;

    #[async_trait]
    impl Backend for FailingBackend {
        async fn run(
            &self,
            _inputs: &[NamedTensorRef<'_>],
            _outputs: &mut [OutputBuffer],
        ) -> Result<(), BackendError> {
            Err(BackendError::InferenceFailed("backend exploded".into()))
        }
    }

    // Captures the input tensor name so it can be asserted after run().
    #[derive(Debug)]
    struct CapturingBackend {
        captured: std::sync::Mutex<String>,
    }

    #[async_trait]
    impl Backend for CapturingBackend {
        async fn run(
            &self,
            inputs: &[NamedTensorRef<'_>],
            _outputs: &mut [OutputBuffer],
        ) -> Result<(), BackendError> {
            *self.captured.lock().unwrap() = inputs[0].name.to_owned();
            Ok(())
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn backend_output_written_to_scratchpad() {
        let mut stage = InferStage {
            backend: Arc::new(MockBackend { output_value: 42.0 }),
            input_name: "features".parse().unwrap(),
        };
        let mut ctx = ctx_with_output(&[1]);
        stage.run(&mut ctx).unwrap();
        assert_eq!(ctx.outputs[0].data[[0]], 42.0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn backend_error_propagates_as_stage_failed() {
        let mut stage = InferStage {
            backend: Arc::new(FailingBackend),
            input_name: "features".parse().unwrap(),
        };
        let err = stage.run(&mut ctx_with_output(&[1])).unwrap_err();
        assert!(matches!(
            err,
            PipelineError::StageFailed {
                stage: "InferStage",
                ..
            }
        ));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn input_name_forwarded_to_backend() {
        let capturing = Arc::new(CapturingBackend {
            captured: std::sync::Mutex::new(String::new()),
        });
        let mut stage = InferStage {
            backend: Arc::clone(&capturing) as Arc<dyn Backend>,
            input_name: "my_input".parse().unwrap(),
        };
        stage.run(&mut ctx_with_output(&[1])).unwrap();
        assert_eq!(*capturing.captured.lock().unwrap(), "my_input");
    }
}
