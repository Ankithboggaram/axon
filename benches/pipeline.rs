//! Benchmarks for individual pipeline stage costs and end-to-end pipeline latency.
//!
//! Stage benchmarks isolate each transform on a [1,1,28,28] tensor.
//! They require no async runtime because all stages are synchronous.
//!
//! Pipeline benchmarks require a multi-thread tokio runtime because InferStage
//! uses block_in_place internally to bridge the sync Stage trait with the async
//! Backend trait.
//!
//! Key comparisons:
//!   stage_* numbers should be tiny (<1 µs); confirms preprocessing is not the bottleneck
//!   pipeline_infer_only vs pipeline_preprocess_and_infer shows preprocessing overhead
//!   Both pipeline numbers should be dominated by inference_sequential_pool_1 from backend.rs

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::{Arc, Mutex};

use arrayvec::ArrayString;
use ndarray::{ArrayD, IxDyn};
use pipexec::stage::Stage;

use axon::backend::onnx::OnnxBackend;
use axon::config::{
    BackendConfig, BackendType, Config, DeviceConfig, GrpcConfig, MetricsConfig, ModelSchemaConfig,
    PipelineConfig, RegistryConfig, RegistryType, StageConfig, StageObservability, StoreConfig,
    StoreType, TensorSpec,
};
use axon::pipeline::InferenceScratchpad;
use axon::pipeline::build::{build, build_scratchpad};
use axon::pipeline::stages::clip::ClipStage;
use axon::pipeline::stages::impute::ImputeStage;
use axon::pipeline::stages::normalize::NormalizeStage;
use axon::pipeline::stages::validate::ValidateStage;
use axon::types::OutputBuffer;

const MODEL: &str = "tests/fixtures/mnist-8.onnx";

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
            key_prefix: None,
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

fn make_mnist_ctx() -> InferenceScratchpad {
    InferenceScratchpad {
        entity_id: ArrayString::new(),
        timestamp_ms: 0,
        input: ArrayD::<f32>::from_elem(IxDyn(&[1, 1, 28, 28]), 127.0),
        outputs: Box::new([OutputBuffer {
            name: "Plus214_Output_0".parse().unwrap(),
            data: ArrayD::zeros(IxDyn(&[1, 10])),
        }]),
    }
}

// ImputeStage: replaces NaN values with a default.
// Input has 50% NaN elements to give the branch predictor real work.
#[divan::bench]
fn stage_impute(bencher: divan::Bencher) {
    let mut stage = ImputeStage { default_value: 0.0 };
    bencher
        .with_inputs(|| {
            let mut ctx = make_mnist_ctx();
            ctx.input.iter_mut().step_by(2).for_each(|v| *v = f32::NAN);
            ctx
        })
        .bench_local_values(|mut ctx| {
            stage.run(&mut ctx).unwrap();
        });
}

// NormalizeStage: (x - mean) * inv_std on every element of [1,1,28,28].
#[divan::bench]
fn stage_normalize(bencher: divan::Bencher) {
    let mut stage = NormalizeStage {
        mean: 127.5,
        inv_std: 1.0 / 255.0,
    };
    bencher
        .with_inputs(make_mnist_ctx)
        .bench_local_values(|mut ctx| {
            stage.run(&mut ctx).unwrap();
        });
}

// ClipStage: element-wise clamp to [0, 255].
#[divan::bench]
fn stage_clip(bencher: divan::Bencher) {
    let mut stage = ClipStage {
        min: 0.0,
        max: 255.0,
    };
    bencher
        .with_inputs(make_mnist_ctx)
        .bench_local_values(|mut ctx| {
            stage.run(&mut ctx).unwrap();
        });
}

// ValidateStage: shape check only, O(1) regardless of tensor size.
#[divan::bench]
fn stage_validate(bencher: divan::Bencher) {
    let mut stage = ValidateStage {
        expected_shape: Box::new([1, 1, 28, 28]),
    };
    bencher
        .with_inputs(make_mnist_ctx)
        .bench_local_values(|mut ctx| {
            stage.run(&mut ctx).unwrap();
        });
}

// Full pipeline: Infer stage only.
// Measures input packaging + backend dispatch + output writing.
// Compare against inference_sequential_pool_4 in backend.rs to see pipeline wrapper cost.
#[divan::bench]
fn pipeline_infer_only(bencher: divan::Bencher) {
    let config = mnist_config();
    let backend = Arc::new(OnnxBackend::new(MODEL, 4, DeviceConfig::Cpu).unwrap())
        as Arc<dyn axon::backend::Backend>;
    let pipeline = Arc::new(Mutex::new(build(&config, Arc::clone(&backend)).unwrap().0));
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .build()
        .unwrap();
    bencher
        .with_inputs(|| build_scratchpad(&config).unwrap())
        .bench_local_values(|ctx| {
            let pipeline = Arc::clone(&pipeline);
            rt.block_on(async move {
                tokio::spawn(async move {
                    let mut p = pipeline.lock().unwrap();
                    let mut c = ctx;
                    p.run(&mut c).unwrap();
                })
                .await
                .unwrap();
            });
        });
}

// Full pipeline: Impute + Clip + Normalize + Infer.
// Measures total hot-path cost of a realistic preprocessing configuration.
// The delta vs pipeline_infer_only is the preprocessing overhead.
#[divan::bench]
fn pipeline_preprocess_and_infer(bencher: divan::Bencher) {
    let mut config = mnist_config();
    config.pipeline.stages = vec![
        StageConfig::Impute {
            default_value: 0.0,
            observability: no_obs(),
        },
        StageConfig::Clip {
            min: 0.0,
            max: 255.0,
            observability: no_obs(),
        },
        StageConfig::Normalize {
            mean: 127.5,
            std: 255.0,
            observability: no_obs(),
        },
        StageConfig::Infer {
            observability: no_obs(),
        },
    ];
    let backend = Arc::new(OnnxBackend::new(MODEL, 4, DeviceConfig::Cpu).unwrap())
        as Arc<dyn axon::backend::Backend>;
    let pipeline = Arc::new(Mutex::new(build(&config, Arc::clone(&backend)).unwrap().0));
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .build()
        .unwrap();
    bencher
        .with_inputs(|| build_scratchpad(&config).unwrap())
        .bench_local_values(|ctx| {
            let pipeline = Arc::clone(&pipeline);
            rt.block_on(async move {
                tokio::spawn(async move {
                    let mut p = pipeline.lock().unwrap();
                    let mut c = ctx;
                    p.run(&mut c).unwrap();
                })
                .await
                .unwrap();
            });
        });
}
