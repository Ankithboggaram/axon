//! Pipeline wiring and scratchpad definition.

use ndarray::ArrayD;
use pipex::scratchpad::Scratchpad;

use crate::types::NamedTensor;

pub mod stages;

pub struct InferenceScratchpad {
    pub entity_id: String,
    pub request_id: String,
    pub timestamp_ms: i64,
    /// Input tensor - shape is determined by the model schema at runtime.
    pub input: ArrayD<f32>,
    /// Output tensors populated by the infer stage.
    pub outputs: Vec<NamedTensor>,
}

impl Scratchpad for InferenceScratchpad {
    fn reset(&mut self) {
        self.entity_id.clear();
        self.request_id.clear();
        self.timestamp_ms = 0;
        self.input.fill(0.0);
        self.outputs.clear();
    }

    fn validate(&self) -> bool {
        !self.entity_id.is_empty() && !self.input.is_empty()
    }
}
