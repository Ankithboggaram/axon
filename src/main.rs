//! Entry point. Wires everything together at startup.

pub mod backend;
pub mod config;
pub mod metrics;
pub mod pipeline;
pub mod registry;
pub mod server;
pub mod store;
pub mod stream;

fn main() {
    println!("Axon starting up");
}
