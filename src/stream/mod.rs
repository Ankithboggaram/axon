//! StreamSource trait and stream source implementations.

use async_trait::async_trait;

#[async_trait]
pub trait StreamSource<T: Send> {
    /// Returns the next item from the stream, blocking until one is available.
    async fn next(&mut self) -> anyhow::Result<T>;
}
