//! Minimal unary inference client.
//!
//! Sends one `Predict` call with inline features (no feature store required)
//! and prints the response.
//!
//! # Usage
//!
//! ```bash
//! # against a local server on the default port
//! cargo run --example client
//!
//! # against a specific address
//! cargo run --example client -- http://[::1]:50051
//! ```

use anyhow::{Context, Result};

mod proto {
    tonic::include_proto!("axon.inference.v1");
}

use proto::PredictRequest;
use proto::inference_service_client::InferenceServiceClient;

#[tokio::main]
async fn main() -> Result<()> {
    let addr = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "http://[::1]:50051".to_owned());

    let mut client = InferenceServiceClient::connect(addr.clone())
        .await
        .with_context(|| {
            format!(
                "could not connect to axon at '{addr}'\n  hint: is the server running?\
            \n        try: axon serve --config config.toml"
            )
        })?;

    // Inline features bypass the feature store. Adjust the length to match
    // your model's expected input shape.
    let request = tonic::Request::new(PredictRequest {
        entity_id: "example-entity".to_owned(),
        features: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        metadata: Default::default(),
    });

    let response = client
        .predict(request)
        .await
        .context("predict RPC failed")?
        .into_inner();

    println!("entity_id:    {}", response.entity_id);
    println!("timestamp_ms: {}", response.timestamp_ms);
    for output in &response.outputs {
        println!(
            "output '{}': {:?}  shape {:?}",
            output.name, output.values, output.shape
        );
    }

    Ok(())
}
