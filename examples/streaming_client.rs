//! Streaming inference client.
//!
//! Subscribes to continuous predictions for a single entity. Prints each
//! prediction as it arrives. The server pushes a new prediction each time the
//! entity's features update in the store.
//!
//! Press Ctrl-C to stop.
//!
//! # Usage
//!
//! ```bash
//! # against a local server on the default port
//! cargo run --example streaming_client
//!
//! # against a specific address
//! cargo run --example streaming_client -- http://[::1]:50051
//! ```

use anyhow::{Context, Result};

mod proto {
    tonic::include_proto!("axon.inference.v1");
}

use proto::PredictStreamRequest;
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

    let request = tonic::Request::new(PredictStreamRequest {
        entity_id: "example-entity".to_owned(),
        metadata: Default::default(),
    });

    let mut stream = client
        .predict_stream(request)
        .await
        .context("predict_stream RPC failed")?
        .into_inner();

    println!("streaming predictions for 'example-entity' — press Ctrl-C to stop");
    println!();

    loop {
        tokio::select! {
            msg = stream.message() => {
                match msg.context("stream error")? {
                    None => {
                        println!("stream closed by server");
                        break;
                    }
                    Some(resp) => {
                        print!("[{}ms] ", resp.timestamp_ms);
                        for output in &resp.outputs {
                            print!("{}={:?}  ", output.name, output.values);
                        }
                        println!();
                    }
                }
            }
            _ = tokio::signal::ctrl_c() => {
                println!("\nshutdown signal received, stopping");
                break;
            }
        }
    }

    Ok(())
}
