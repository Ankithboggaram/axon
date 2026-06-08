//! Pool of ready-to-run inference pipelines for concurrent request handling.
//!
//! Each pipeline requires exclusive access during a run, so one slot is checked
//! out per concurrent request. The first slot is supplied at construction time
//! (it carries the registered `StageMetrics`); additional slots are built by
//! the factory.

use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use parking_lot::Mutex;
use pipex::dynamic_pipeline::Pipeline;

use crate::pipeline::InferenceScratchpad;

/// A pool of inference pipelines for concurrent request serving.
///
/// `acquire` returns a [`PipelineGuard`] that gives exclusive access to one
/// pipeline. On drop the pipeline is returned to the pool. If all slots are
/// in use the factory is called to create an overflow pipeline that is
/// discarded when dropped.
pub struct PipelinePool {
    slots: Mutex<Vec<Pipeline<InferenceScratchpad>>>,
    factory: Box<dyn Fn() -> Pipeline<InferenceScratchpad> + Send + Sync>,
    capacity: usize,
}

impl PipelinePool {
    /// Creates a pool pre-populated to `capacity` slots.
    ///
    /// `first` is inserted as the initial slot. The remaining `capacity - 1`
    /// slots are filled by `factory`. If `capacity` is 0 or 1 the factory is
    /// never called.
    pub fn new(
        first: Pipeline<InferenceScratchpad>,
        capacity: usize,
        factory: impl Fn() -> Pipeline<InferenceScratchpad> + Send + Sync + 'static,
    ) -> Self {
        let factory: Box<dyn Fn() -> Pipeline<InferenceScratchpad> + Send + Sync> =
            Box::new(factory);
        let mut slots = Vec::with_capacity(capacity.max(1));
        slots.push(first);
        for _ in 1..capacity {
            slots.push(factory());
        }
        Self {
            slots: Mutex::new(slots),
            factory,
            capacity: capacity.max(1),
        }
    }

    /// Checks out a pipeline from the pool.
    ///
    /// The guard owns an `Arc` clone, so it is `Send` and safe to hold across
    /// `await` points without borrowing from the pool directly.
    #[must_use]
    pub fn acquire(self: &Arc<Self>) -> PipelineGuard {
        let pipeline = self.slots.lock().pop().unwrap_or_else(|| (self.factory)());
        PipelineGuard {
            pipeline: Some(pipeline),
            pool: Arc::clone(self),
        }
    }
}

/// RAII guard that holds one pipeline checked out from a [`PipelinePool`].
///
/// Derefs to `Pipeline<InferenceScratchpad>`. On drop the pipeline is returned
/// to the pool if capacity has not been reached; otherwise it is dropped.
pub struct PipelineGuard {
    pipeline: Option<Pipeline<InferenceScratchpad>>,
    pool: Arc<PipelinePool>,
}

impl Drop for PipelineGuard {
    fn drop(&mut self) {
        if let Some(pipeline) = self.pipeline.take() {
            let mut slots = self.pool.slots.lock();
            if slots.len() < self.pool.capacity {
                slots.push(pipeline);
            }
        }
    }
}

#[allow(clippy::expect_used)] // pipeline is Some until Drop::drop runs; reaching this after drop is a bug
impl Deref for PipelineGuard {
    type Target = Pipeline<InferenceScratchpad>;

    fn deref(&self) -> &Self::Target {
        self.pipeline.as_ref().expect("guard used after drop")
    }
}

#[allow(clippy::expect_used)] // pipeline is Some until Drop::drop runs; reaching this after drop is a bug
impl DerefMut for PipelineGuard {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.pipeline.as_mut().expect("guard used after drop")
    }
}
