# axon

[![CI](https://github.com/Ankithboggaram/axon/actions/workflows/ci.yml/badge.svg)](https://github.com/Ankithboggaram/axon/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![MSRV](https://img.shields.io/badge/rustc-1.85+-orange.svg)](rust-toolchain.toml)

Axon is the serving component of **Cortex**, a real-time ML platform, and runs standalone as a TOML-configured ML inference server. Describe a model and a preprocessing pipeline in one file; Axon serves predictions over gRPC. The binary never changes; only the config does.

```toml
[[pipeline.stages]]
type = "impute"
default_value = 0.0

[[pipeline.stages]]
type = "clip"
min = -3.0
max = 3.0

[[pipeline.stages]]
type = "normalize"
mean = 0.0
std = 1.0

[[pipeline.stages]]
type = "infer"

[[pipeline.stages]]
type = "postprocess"
threshold = 0.5
output_type = "binary"
```

That's a complete preprocessing pipeline: impute → clip → normalize → run the model → threshold the output, stage order and all. Reorder stages, tune a threshold, or point at a different model by editing this file and restarting; nothing else changes.

## What you configure

|                       |                                                                                                                                                              |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **The model**         | Which registry to pull it from (MLflow built in) and which backend runs it (ONNX Runtime built in)                                                           |
| **The pipeline**      | An ordered list of stages (`impute`, `validate`, `clip`, `normalize`, `infer`, `postprocess`), each with its own parameters, in whatever order you list them |
| **The feature store** | Where per-entity feature vectors come from (Redis built in), with optional freshness and schema-version enforcement                                          |
| **Observability**     | Per-stage latency, tracing, retries, deadlines: opt in per stage, in the same file                                                                           |

Backends, stores, and registries are Rust traits underneath, so the config file isn't a ceiling; see [Extensibility](#extensibility) if the built-ins aren't enough.

**Where Docker fits in:** Axon ships as one Docker image, the compiled interpreter for your config file. Mount a different `config.toml` into the same container and you get a different pipeline; the image itself never changes.

---

## Quickstart

This walks through a real end-to-end run: train a demo model, register it, seed some features, and serve a prediction.

**Prerequisites:** Docker, plus Redis and MLflow running somewhere reachable.

```bash
git clone https://github.com/Ankithboggaram/axon
cd axon

# Redis and MLflow, if you don't already have them
docker run -d -p 6379:6379 redis:7-alpine
pip install mlflow
mlflow server --host 0.0.0.0 --port 5000
```

**1. Seed a demo model and some features.** The included script trains a small fraud-detection model, registers it in MLflow, and writes feature vectors to Redis:

```bash
git clone https://github.com/Ankithboggaram/cortex-contract ../cortex-contract
pip install numpy scikit-learn onnx mlflow redis protobuf
python scripts/seed_demo.py
```

**2. Build the image, then generate a config from the registry:**

```bash
docker build -t axon .

docker run --rm --network host \
  -v $(pwd):/output \
  axon init fraud_demo 1 \
  --registry-uri http://localhost:5000 \
  --output /output/config.toml
```

`axon init` reads the model's signature and logged training params (`mean`, `std`, `clip_min`, `clip_max`, `threshold`) straight from MLflow and writes them into `config.toml` for you. Open it and fill in `registry.uri`; that's the only value left blank.

**3. Run it, and send a request:**

```bash
docker run -d -p 50051:50051 -p 9090:9090 \
  -v $(pwd)/config.toml:/app/config.toml \
  axon

grpcurl -plaintext \
  -d '{"entity_id": "entity_0001"}' \
  localhost:50051 axon.inference.v1.InferenceService/Predict
```

That's the whole loop: config in, prediction out. See [Configuration](#configuration) for what else you can put in that file.

---

## Configuration

The reference config, with every section:

```toml
[grpc]
host = "0.0.0.0"
port = 50051
stream_poll_interval_ms = 1    # polling interval for streaming RPCs
request_timeout_ms = 5000      # requests exceeding this are cancelled
pool_size = 4                  # pipeline pool slots (default: logical CPU count)
session_pool_size = 4          # ONNX session pool slots (default: logical CPU count)

[backend]
type = "onnx_runtime"
device = "cpu"             # "cpu" | "coreml" (macOS) | "cuda" | "tensorrt"

[registry]
type          = "mlflow"
uri           = "http://localhost:5000"
model_name    = "my_model"
model_version = "1"            # version number or "latest"

[store]
type                       = "redis"
url                        = "redis://localhost:6379"
key_prefix                 = "features"  # keys stored as {key_prefix}:{entity_id}
health_check_interval_secs = 10          # readiness probe polling interval

# Optional. Omit this whole section to disable freshness enforcement; the
# axon_served_feature_age_seconds metric is still recorded either way.
[freshness]
max_feature_age_ms = 5000     # reject/flag features older than this
on_stale            = "flag"  # "flag" (serve anyway, just warn) | "reject"

# Optional. Omit this whole section (or set enabled = false) to disable
# closed-loop prediction logging entirely; no Kafka producer is created and
# emission is a single no-op check on the hot path.
# [predictions]
# enabled     = true
# brokers     = "localhost:9092"
# topic       = "predictions"
# sample_rate = 1.0   # 0.0..=1.0; fraction of predictions emitted (deterministic, every Nth)

[metrics]
port = 9090

# Tensor names, types, and shapes must match the model exactly.
[[model_schema.inputs]]
name  = "features"
dtype = "float32"
shape = [1, 32]

[[model_schema.outputs]]
name  = "score"
dtype = "float32"
shape = [1, 1]

# Stages run in the order listed.
[[pipeline.stages]]
type          = "impute"
default_value = 0.0

[[pipeline.stages]]
type           = "validate"
expected_shape = [1, 32]

[[pipeline.stages]]
type = "clip"
min  = -3.0
max  = 3.0

[[pipeline.stages]]
type = "normalize"
mean = 0.0
std  = 1.0

[[pipeline.stages]]
type = "infer"

[[pipeline.stages]]
type        = "postprocess"
threshold   = 0.5
output_type = "binary"    # "binary" (±1) | "probability" | "raw"
```

Supported dtypes: `float32` `float64` `int8` `int16` `int32` `int64` `uint8` `uint16` `uint32` `uint64` `bool` `string`

### Pipeline stages

| Stage         | What it does                                       | Parameters                 |
| ------------- | -------------------------------------------------- | -------------------------- |
| `impute`      | Replaces NaN values with a fixed default           | `default_value`            |
| `validate`    | Rejects wrong shapes and non-finite values         | `expected_shape`           |
| `clip`        | Clamps values to `[min, max]` before normalisation | `min`, `max`               |
| `normalize`   | Applies zero-mean unit-variance normalisation      | `mean`, `std`              |
| `infer`       | Runs the model via the configured backend          | (none)                     |
| `postprocess` | Transforms raw model output into a prediction      | `threshold`, `output_type` |

Any stage also takes optional observability and reliability flags:

```toml
[[pipeline.stages]]
type         = "infer"
timed        = true    # records p99/p999 latency in Prometheus
instrumented = true    # emits a tracing span per execution
retries      = 3       # retries up to N times on failure
deadline_ms  = 50      # fails the stage if it exceeds this budget
```

---

## Inference modes

**Unary**: one request, one prediction. Populate `features` in the request to bypass the feature store entirely and pass values inline.

**Streaming**: one subscription, a prediction pushed every time the entity's features update (via Redis pub/sub; polls at `stream_poll_interval_ms` for stores without push support).

```bash
grpcurl -plaintext -d '{"entity_id": "user_123"}' \
  localhost:50051 axon.inference.v1.InferenceService/Predict

grpcurl -plaintext -d '{"entity_id": "user_123"}' \
  localhost:50051 axon.inference.v1.InferenceService/PredictStream
```

Client examples for Rust and Python are in [`examples/`](examples/).

---

## Extensibility

Backends, stores, and registries are traits. Add a new implementation and a config enum variant; the pipeline and server don't change.

```rust
use axon::backend::Backend;
use axon::error::BackendError;
use axon::types::{NamedTensorRef, OutputBuffer};
use async_trait::async_trait;

#[derive(Debug)]
pub struct TritonBackend {
    url: String,
}

#[async_trait]
impl Backend for TritonBackend {
    async fn run(
        &self,
        inputs: &[NamedTensorRef<'_>],
        outputs: &mut [OutputBuffer],
    ) -> Result<(), BackendError> {
        // Send inputs to Triton over gRPC, write results into outputs.
        Ok(())
    }
}
```

| Trait                 | Implement to add                                                                |
| --------------------- | ------------------------------------------------------------------------------- |
| `Backend`             | A new inference runtime (e.g. Triton)                                           |
| `OnlineStoreReader`   | A new feature store backend (e.g. Feast, Cassandra), added to `cortex-contract` |
| `ModelRegistryClient` | A new model registry (e.g. Vertex AI, custom HTTP)                              |

---

## Observability

Prometheus metrics at `http://localhost:<metrics.port>/metrics` (default `9090`): request counts and latency, feature-store fetch latency and staleness, per-stage p99/p999, and prediction-logging throughput. Structured logs go to stdout: `RUST_LOG` for verbosity, `AXON_LOG_JSON=1` for JSON output.

Axon speaks the [gRPC health checking protocol](https://github.com/grpc/grpc/blob/master/doc/health-checking.md): liveness is `SERVING` once the process is up, readiness is `SERVING` only while the feature store is confirmed reachable (checked every `health_check_interval_secs`, drops to `NOT_SERVING` after two consecutive failures).

```bash
grpc_health_probe -addr=localhost:50051
grpc_health_probe -addr=localhost:50051 -service=axon.inference.v1.InferenceService
```

---

## Roadmap

- [x] Session-pooled ONNX backend and pipeline pool for concurrent inference
- [x] Event-driven streaming via Redis pub/sub
- [x] Closed-loop prediction logging to Kafka
- [ ] Triton Inference Server backend
- [ ] Additional pipeline stages (`drift_detect`, `audit`, `argmax`)
- [ ] WASM custom pipeline stages

---

## Building from source

```bash
git clone https://github.com/Ankithboggaram/axon
cd axon
cargo build --release
cp examples/config.toml config.toml   # fill in TODOs, or generate one via `axon init`
./target/release/axon serve --config config.toml
```

Requires Rust 1.85+ (see [rust-toolchain.toml](rust-toolchain.toml)) and, to compile `rdkafka` and the `.proto` files from source: `cmake`, a C/C++ toolchain, `protobuf-compiler`, and OpenSSL/zlib/curl development headers (`libssl-dev`, `zlib1g-dev`, `libcurl4-openssl-dev` on Debian/Ubuntu).

---

## License

MIT
