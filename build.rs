//! Compiles proto/axon/inference/v1/inference.proto into Rust at build time.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(true)
        .build_client(false)
        .compile_protos(&["proto/axon/inference/v1/inference.proto"], &["proto"])?;
    Ok(())
}
