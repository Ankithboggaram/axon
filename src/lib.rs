//! Axon is a configuration-driven ML inference server for real-time model serving.

#![deny(clippy::unwrap_used, clippy::expect_used)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
#![warn(missing_docs)]

pub mod backend;
pub mod config;
pub mod error;
#[allow(missing_docs)]
pub mod proto {
    tonic::include_proto!("axon.inference.v1");
}
pub mod metrics;
pub mod pipeline;
pub mod registry;
pub mod server;
pub mod store;
pub mod types;
