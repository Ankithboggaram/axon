//! Pipeline wiring and scratchpad definition.

use arrayvec::ArrayString;
use ndarray::ArrayD;
use pipex::scratchpad::Scratchpad;

use crate::types::OutputBuffer;

pub mod stages;

/// Maximum byte length for entity_id and request_id.
///
/// 128 bytes covers UUIDs (36 bytes), compound identifiers, and symbol names
/// with room to spare. The server layer must reject IDs that exceed this limit.
pub const MAX_ID_LEN: usize = 128;

#[derive(Debug)]
pub struct InferenceScratchpad {
    /// Entity identifier — stack-allocated, never heap-allocates.
    pub entity_id: ArrayString<MAX_ID_LEN>,
    /// Request identifier — stack-allocated, never heap-allocates.
    pub request_id: ArrayString<MAX_ID_LEN>,
    pub timestamp_ms: i64,
    /// Input tensor — pre-allocated at startup to the shape from model_schema.inputs.
    pub input: ArrayD<f32>,
    /// Output buffers — pre-allocated at startup, one per model_schema.outputs entry.
    /// The backend writes into these in place each request via assign().
    pub outputs: Box<[OutputBuffer]>,
}

impl Scratchpad for InferenceScratchpad {
    fn reset(&mut self) {
        self.entity_id.clear();
        self.request_id.clear();
        self.timestamp_ms = 0;
        // Zeroes the input buffer without deallocating. The server layer will
        // overwrite all elements before the pipeline runs, so this is a
        // protective measure against stale data leaking across requests.
        self.input.fill(0.0);
        // Zero each pre-allocated output buffer. Shape and allocation are preserved.
        for out in self.outputs.iter_mut() {
            out.data.fill(0.0);
        }
    }

    fn validate(&self) -> bool {
        !self.entity_id.is_empty() && !self.input.is_empty()
    }
}
