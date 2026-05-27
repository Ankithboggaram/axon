//! Pipeline wiring and scratchpad definition.

use crate::types::Tensor;
use pipex::scratchpad::Scratchpad;

pub mod stages;

pub struct InferenceScratchpad {
    pub entity_id: String,
    pub request_id: String,
    pub timestamp_ms: i64,
    pub features: Vec<f32>,
    pub outputs: Vec<Tensor>,
}

impl Scratchpad for InferenceScratchpad {
    fn reset(&mut self) {
        self.entity_id.clear();
        self.request_id.clear();
        self.timestamp_ms = 0;
        self.features.clear();
        self.outputs.clear();
    }

    fn validate(&self) -> bool {
        !self.entity_id.is_empty() && !self.features.is_empty()
    }
}
