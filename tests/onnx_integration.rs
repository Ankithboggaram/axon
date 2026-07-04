//! Integration tests using the MNIST-8 fixture model.
//!
//! Model spec (ONNX model zoo mnist-8):
//!   input  "Input3"            float32  [1, 1, 28, 28]  (pixel values 0–255)
//!   output "Plus214_Output_0"  float32  [1, 10]         (raw logits, one per digit class)
//!
//! Fixtures (both represent the same digit — the first image in the MNIST test set, label 7):
//!   mnist_sample_7.png  — human-verifiable image; open it to confirm it looks like a 7
//!   mnist_sample_7.bin  — 784 raw float32 pixel values; mirrors what the feature store holds

use std::sync::Arc;

use ndarray::{ArrayD, IxDyn};

use axon::backend::Backend;
use axon::backend::onnx::OnnxBackend;
use axon::config::{
    BackendConfig, BackendType, Config, DeviceConfig, GrpcConfig, MetricsConfig, ModelSchemaConfig,
    PipelineConfig, RegistryConfig, RegistryType, StageConfig, StageObservability, StoreConfig,
    StoreType, TensorSpec,
};
use axon::pipeline::build::{build, build_scratchpad};
use axon::types::{NamedTensorRef, OutputBuffer};

const MODEL: &str = "tests/fixtures/mnist-8.onnx";
const SAMPLE_7_BIN: &str = "tests/fixtures/mnist_sample_7.bin";
const SAMPLE_7_PNG: &str = "tests/fixtures/mnist_sample_7.png";

// Backend tests

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
            url: "redis://localhost:6379".to_owned(),
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

#[tokio::test(flavor = "multi_thread")]
async fn backend_loads_model() {
    OnnxBackend::new(MODEL, 1, DeviceConfig::Cpu).expect("failed to load MNIST model");
}

#[tokio::test(flavor = "multi_thread")]
async fn backend_produces_correct_output_shape() {
    let backend = OnnxBackend::new(MODEL, 1, DeviceConfig::Cpu).unwrap();
    let input = ArrayD::<f32>::zeros(IxDyn(&[1, 1, 28, 28]));
    let inputs = [NamedTensorRef {
        name: "Input3",
        data: input.view(),
    }];
    let mut outputs = vec![OutputBuffer {
        name: "Plus214_Output_0".parse().unwrap(),
        data: ArrayD::zeros(IxDyn(&[1, 10])),
    }];

    backend.run(&inputs, &mut outputs).await.unwrap();

    assert_eq!(outputs[0].data.shape(), &[1, 10]);
}

#[tokio::test(flavor = "multi_thread")]
async fn backend_output_is_finite() {
    let backend = OnnxBackend::new(MODEL, 1, DeviceConfig::Cpu).unwrap();
    let input = ArrayD::<f32>::zeros(IxDyn(&[1, 1, 28, 28]));
    let inputs = [NamedTensorRef {
        name: "Input3",
        data: input.view(),
    }];
    let mut outputs = vec![OutputBuffer {
        name: "Plus214_Output_0".parse().unwrap(),
        data: ArrayD::zeros(IxDyn(&[1, 10])),
    }];

    backend.run(&inputs, &mut outputs).await.unwrap();

    assert!(outputs[0].data.iter().all(|v| v.is_finite()));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn session_pool_handles_concurrent_requests() {
    let backend = Arc::new(OnnxBackend::new(MODEL, 2, DeviceConfig::Cpu).unwrap());

    // 8 concurrent requests against a pool of 2 — 6 must go through overflow sessions.
    let tasks: Vec<_> = (0..8)
        .map(|_| {
            let b = Arc::clone(&backend);
            tokio::spawn(async move {
                let input = ArrayD::<f32>::zeros(IxDyn(&[1, 1, 28, 28]));
                let inputs = [NamedTensorRef {
                    name: "Input3",
                    data: input.view(),
                }];
                let mut outputs = vec![OutputBuffer {
                    name: "Plus214_Output_0".parse().unwrap(),
                    data: ArrayD::zeros(IxDyn(&[1, 10])),
                }];
                b.run(&inputs, &mut outputs).await.unwrap();
                outputs[0].data.shape().to_vec()
            })
        })
        .collect();

    for task in tasks {
        assert_eq!(task.await.unwrap(), vec![1, 10]);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn pipeline_runs_end_to_end() {
    let config = mnist_config();
    let backend = Arc::new(OnnxBackend::new(MODEL, 1, DeviceConfig::Cpu).unwrap())
        as Arc<dyn axon::backend::Backend>;
    let (mut pipeline, _metrics) = build(&config, backend).unwrap();
    let mut ctx = build_scratchpad(&config).unwrap();

    pipeline.run(&mut ctx).unwrap();

    assert_eq!(ctx.outputs[0].data.shape(), &[1, 10]);
    assert!(ctx.outputs[0].data.iter().all(|v| v.is_finite()));
}

// Pipeline tests

#[tokio::test(flavor = "multi_thread")]
async fn pipeline_output_changes_with_different_inputs() {
    let config = mnist_config();
    let backend = Arc::new(OnnxBackend::new(MODEL, 1, DeviceConfig::Cpu).unwrap())
        as Arc<dyn axon::backend::Backend>;
    let (mut pipeline, _) = build(&config, Arc::clone(&backend)).unwrap();

    let mut ctx = build_scratchpad(&config).unwrap();
    pipeline.run(&mut ctx).unwrap();
    let output_zeros: Vec<f32> = ctx.outputs[0].data.iter().copied().collect();

    // Fill input with 1.0 and re-run.
    ctx.input.fill(1.0);
    pipeline.run(&mut ctx).unwrap();
    let output_ones: Vec<f32> = ctx.outputs[0].data.iter().copied().collect();

    assert_ne!(
        output_zeros, output_ones,
        "different inputs must produce different outputs"
    );
}

// Classification tests

// In production the feature store holds pre-processed float32 vectors. Axon fetches and
// feeds them into the model with no additional preprocessing. The .bin fixture is the
// actual input format axon receives at inference time.
#[tokio::test(flavor = "multi_thread")]
async fn model_classifies_feature_store_vector_correctly() {
    let bytes = std::fs::read(SAMPLE_7_BIN).unwrap();
    let pixels: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    assert_eq!(pixels.len(), 784, "feature vector must be 784 elements");

    let backend = OnnxBackend::new(MODEL, 1, DeviceConfig::Cpu).unwrap();
    let input = ArrayD::from_shape_vec(IxDyn(&[1, 1, 28, 28]), pixels).unwrap();
    let inputs = [NamedTensorRef {
        name: "Input3",
        data: input.view(),
    }];
    let mut outputs = vec![OutputBuffer {
        name: "Plus214_Output_0".parse().unwrap(),
        data: ArrayD::zeros(IxDyn(&[1, 10])),
    }];

    backend.run(&inputs, &mut outputs).await.unwrap();

    let predicted = argmax(&outputs[0].data);
    assert_eq!(predicted, 7);
}

// Not a production path. Open tests/fixtures/mnist_sample_7.png to visually verify the
// fixture is a 7. The PNG and .bin are derived from the same source pixel data.
#[tokio::test(flavor = "multi_thread")]
async fn model_classifies_png_image_correctly() {
    let img = image::open(SAMPLE_7_PNG).unwrap().to_luma8();
    assert_eq!(img.dimensions(), (28, 28));

    let pixels: Vec<f32> = img.pixels().map(|p| p.0[0] as f32).collect();

    let backend = OnnxBackend::new(MODEL, 1, DeviceConfig::Cpu).unwrap();
    let input = ArrayD::from_shape_vec(IxDyn(&[1, 1, 28, 28]), pixels).unwrap();
    let inputs = [NamedTensorRef {
        name: "Input3",
        data: input.view(),
    }];
    let mut outputs = vec![OutputBuffer {
        name: "Plus214_Output_0".parse().unwrap(),
        data: ArrayD::zeros(IxDyn(&[1, 10])),
    }];

    backend.run(&inputs, &mut outputs).await.unwrap();

    let predicted = argmax(&outputs[0].data);
    assert_eq!(predicted, 7);
}

fn argmax(array: &ArrayD<f32>) -> usize {
    array
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}
