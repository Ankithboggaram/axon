//! Benchmarks for RAII pool acquire/release overhead vs fresh allocation.
//!
//! The central claim: acquiring a pre-allocated scratchpad from the pool costs
//! only a mutex lock + Vec::pop, while fresh allocation requires ndarray to
//! zero-initialize 784 floats (3136 bytes).
//!
//! Key comparisons:
//!   scratchpad_pool_acquire_release vs scratchpad_fresh_alloc
//!     quantifies the zero-allocation hot path benefit
//!   pipeline_pool_acquire_release
//!     shows pipeline RAII overhead (should be similar to scratchpad)
//!   scratchpad_pool_contention
//!     N threads competing for pool slots; overflow stays inside lock + pop

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use pipex::pool::ScratchpadPool;

use axon::config::{
    BackendConfig, BackendType, Config, DeviceConfig, GrpcConfig, MetricsConfig, ModelSchemaConfig,
    PipelineConfig, RegistryConfig, RegistryType, StageConfig, StageObservability, StoreConfig,
    StoreType, TensorSpec,
};
use axon::pipeline::build::{build, build_scratchpad};

fn main() {
    divan::main();
}

fn no_obs() -> StageObservability {
    StageObservability {
        timed: None,
        instrumented: None,
        retries: None,
        deadline_ms: None,
    }
}

fn mnist_config() -> Config {
    Config {
        grpc: GrpcConfig {
            host: "0.0.0.0".to_owned(),
            port: 50051,
            stream_poll_interval_ms: 1,
            request_timeout_ms: 5000,
            pool_size: None,
            session_pool_size: None,
        },
        backend: BackendConfig {
            backend_type: BackendType::OnnxRuntime,
            device: DeviceConfig::Cpu,
        },
        registry: RegistryConfig {
            registry_type: RegistryType::Mlflow,
            uri: "http://localhost:5000".to_owned(),
            model_name: "mnist".to_owned(),
            model_version: "1".to_owned(),
        },
        store: StoreConfig {
            store_type: StoreType::Redis,
            host: "localhost".to_owned(),
            port: 6379,
            health_check_interval_secs: None,
        },
        metrics: MetricsConfig { port: 9090 },
        model_schema: ModelSchemaConfig {
            inputs: vec![TensorSpec {
                name: "Input3".to_owned(),
                dtype: "float32".to_owned(),
                shape: vec![1, 1, 28, 28],
            }],
            outputs: vec![TensorSpec {
                name: "Plus214_Output_0".to_owned(),
                dtype: "float32".to_owned(),
                shape: vec![1, 10],
            }],
        },
        pipeline: PipelineConfig {
            stages: vec![StageConfig::Infer {
                observability: no_obs(),
            }],
        },
    }
}

// Acquire and immediately release a scratchpad from a warm pool.
// Measures: mutex lock + Vec::pop + RAII drop (Vec::push + mutex unlock).
#[divan::bench]
fn scratchpad_pool_acquire_release(bencher: divan::Bencher) {
    let config = mnist_config();
    let pool = Arc::new(ScratchpadPool::new(4, move || {
        build_scratchpad(&config).unwrap()
    }));
    bencher.bench_local(|| {
        let _guard = pool.acquire();
    });
}

// Allocate a fresh scratchpad each iteration (no pool).
// Measures: ndarray zeros allocation for [1,1,28,28] input + [1,10] output = 3144 bytes.
// Compare against scratchpad_pool_acquire_release to see how much pool reuse saves.
#[divan::bench]
fn scratchpad_fresh_alloc(bencher: divan::Bencher) {
    let config = mnist_config();
    bencher.bench_local(|| {
        let _ctx = build_scratchpad(&config).unwrap();
    });
}

// Acquire and immediately release a pipeline from the pool.
// Measures: mutex lock + Vec::pop + RAII drop for a pipeline slot.
// Pipeline slots are heavier than scratchpads (they hold Arc<dyn Backend> + stage state).
#[divan::bench]
fn pipeline_pool_acquire_release(bencher: divan::Bencher) {
    use axon::backend::onnx::OnnxBackend;
    use axon::pipeline::pool::PipelinePool;

    let config = mnist_config();
    let backend =
        Arc::new(OnnxBackend::new("tests/fixtures/mnist-8.onnx", 1, DeviceConfig::Cpu).unwrap())
            as Arc<dyn axon::backend::Backend>;
    let (first, _) = build(&config, Arc::clone(&backend)).unwrap();
    let pool = Arc::new(PipelinePool::new(first, 4, {
        let config = config.clone();
        let backend = Arc::clone(&backend);
        move || build(&config, Arc::clone(&backend)).unwrap().0
    }));
    bencher.bench_local(|| {
        let _guard = pool.acquire();
    });
}

// N threads competing for a pool of 4 scratchpad slots.
// At threads <= 4: every acquire gets a pooled slot immediately.
// At threads > 4: overflow callers call the factory (fresh alloc).
// Shows that pool contention degrades gracefully rather than blocking.
#[divan::bench(threads = [1, 2, 4, 8])]
fn scratchpad_pool_contention(bencher: divan::Bencher) {
    let config = mnist_config();
    let pool = Arc::new(ScratchpadPool::new(4, move || {
        build_scratchpad(&config).unwrap()
    }));
    bencher.bench_local(|| {
        let _guard = pool.acquire();
    });
}
