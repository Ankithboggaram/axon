//! Benchmarks for OnnxBackend inference latency and session pool behavior.
//!
//! Key numbers to watch:
//!   inference_sequential_pool_1  - raw ONNX execution time; baseline for all other numbers
//!   inference_sequential_pool_4  - should match pool_1; confirms pop/push overhead is negligible
//!   inference_concurrent_pool_4  - throughput scaling with a right-sized pool
//!   inference_overflow_pool_1    - cost of creating an overflow session (compare at threads=8)

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;
use std::sync::LazyLock;

use ndarray::{ArrayD, IxDyn};

use axon::backend::Backend;
use axon::backend::onnx::OnnxBackend;
use axon::types::{NamedTensorRef, OutputBuffer};

const MODEL: &str = "tests/fixtures/mnist-8.onnx";

// Loaded once at process start; session pool is the only shared mutable state.
static BACKEND_POOL_1: LazyLock<Arc<OnnxBackend>> =
    LazyLock::new(|| Arc::new(OnnxBackend::new(MODEL, 1).unwrap()));
static BACKEND_POOL_4: LazyLock<Arc<OnnxBackend>> =
    LazyLock::new(|| Arc::new(OnnxBackend::new(MODEL, 4).unwrap()));

fn main() {
    divan::main();
}

fn make_input() -> ArrayD<f32> {
    ArrayD::<f32>::zeros(IxDyn(&[1, 1, 28, 28]))
}

fn make_outputs() -> Vec<OutputBuffer> {
    vec![OutputBuffer {
        name: "Plus214_Output_0".parse().unwrap(),
        data: ArrayD::zeros(IxDyn(&[1, 10])),
    }]
}

// Sequential inference, pool_size=1.
// Pure ONNX execution time with zero session contention.
#[divan::bench]
fn inference_sequential_pool_1(bencher: divan::Bencher) {
    let backend = Arc::clone(&BACKEND_POOL_1);
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .build()
        .unwrap();
    bencher
        .with_inputs(|| (make_input(), make_outputs()))
        .bench_local_values(|(input, mut outputs)| {
            rt.block_on(async {
                let inputs = [NamedTensorRef {
                    name: "Input3",
                    data: input.view(),
                }];
                backend.run(&inputs, &mut outputs).await.unwrap();
            });
        });
}

// Sequential inference, pool_size=4.
// Should match pool_1; confirms session pop/push overhead is negligible vs inference.
#[divan::bench]
fn inference_sequential_pool_4(bencher: divan::Bencher) {
    let backend = Arc::clone(&BACKEND_POOL_4);
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .build()
        .unwrap();
    bencher
        .with_inputs(|| (make_input(), make_outputs()))
        .bench_local_values(|(input, mut outputs)| {
            rt.block_on(async {
                let inputs = [NamedTensorRef {
                    name: "Input3",
                    data: input.view(),
                }];
                backend.run(&inputs, &mut outputs).await.unwrap();
            });
        });
}

// N threads simultaneously calling run() on a pool of 4 sessions.
// At threads <= 4 every caller gets a pooled session immediately (no contention).
// At threads > 4 overflow sessions are created on the fly.
#[divan::bench(threads = [1, 2, 4, 8])]
fn inference_concurrent_pool_4(bencher: divan::Bencher) {
    let backend = Arc::clone(&BACKEND_POOL_4);
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .build()
        .unwrap();
    bencher
        .with_inputs(|| (make_input(), make_outputs()))
        .bench_local_values(|(input, mut outputs)| {
            rt.block_on(async {
                let inputs = [NamedTensorRef {
                    name: "Input3",
                    data: input.view(),
                }];
                backend.run(&inputs, &mut outputs).await.unwrap();
            });
        });
}

// 8 callers against pool_size=1.
// 7 of 8 callers always hit the overflow path (create + discard a session per request).
// Compare the threads=8 number against inference_concurrent_pool_4 at threads=8
// to quantify the cost of undersizing the session pool.
#[divan::bench(threads = [1, 8])]
fn inference_overflow_pool_1(bencher: divan::Bencher) {
    let backend = Arc::clone(&BACKEND_POOL_1);
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .build()
        .unwrap();
    bencher
        .with_inputs(|| (make_input(), make_outputs()))
        .bench_local_values(|(input, mut outputs)| {
            rt.block_on(async {
                let inputs = [NamedTensorRef {
                    name: "Input3",
                    data: input.view(),
                }];
                backend.run(&inputs, &mut outputs).await.unwrap();
            });
        });
}
