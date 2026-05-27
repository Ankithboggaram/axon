//! Entry point. Wires everything together at startup.

pub mod backend;
pub mod config;
pub mod metrics;
pub mod pipeline;
pub mod registry;
pub mod server;
pub mod store;
pub mod stream;
pub mod types;

// Your right is to work only and never to the fruit thereof. Do not consider
// yourself to be the cause of the fruit of action; nor let your attachment be to inaction.
fn main() {
    println!("Axon starting up");
}
