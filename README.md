# axon

[![CI](https://github.com/Ankithboggaram/axon/actions/workflows/ci.yml/badge.svg)](https://github.com/Ankithboggaram/axon/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![MSRV](https://img.shields.io/badge/rustc-1.85+-orange.svg)](rust-toolchain.toml)

Axon is a configuration-driven ML inference server for real-time model serving.

Part of the **Cortex** platform. Works alongside Dendrite (feature pipeline) and Synapse (training pipeline), but deployable on its own.

---

## Prerequisites

- A model registry with a model registered (built-in: MLflow)
- A feature store populated with entity feature vectors (built-in: Redis)
- Docker

---

## Quickstart

The steps below use MLflow as the model registry and Redis as the feature store. Start them locally if you don't have them running:

```bash
# Redis
docker run -d -p 6379:6379 redis:7-alpine

# MLflow
pip install mlflow
mlflow server --host 0.0.0.0 --port 5000
```

**Step 1: Build the image**

```bash
git clone https://github.com/Ankithboggaram/axon
cd axon
docker build -t axon .
```

**Step 2: Generate a config**

```bash
docker run --rm --network host \
  -v $(pwd):/output \
  axon init <model_name> <model_version> \
  --registry-uri http://localhost:5000 \
  --output /output/config.toml
```

This queries the registry for the model's signature and any logged training parameters, then writes a `config.toml` pre-filled with what it can determine.

**Step 3: Fill in the TODOs**

Open `config.toml` and replace any remaining `TODO` values. The most common are preprocessing parameters that were not logged to the registry at training time:

```toml
# generated
[[pipeline.stages]]
type = "normalize"
mean = "TODO"
std  = "TODO"

# filled in
[[pipeline.stages]]
type = "normalize"
mean = 0.143
std  = 0.892
```

**Step 4: Run the server**

```bash
docker run -d \
  -p 50051:50051 \
  -p 9090:9090 \
  -v $(pwd)/config.toml:/app/config.toml \
  axon
```

**Step 5: Send a request**

```bash
grpcurl -plaintext \
  -d '{"entity_id": "user_123", "features": [0.1, 0.5, 0.3, 0.8]}' \
  localhost:50051 axon.inference.v1.InferenceService/Predict
```

Passing `features` directly bypasses the feature store. Useful for smoke-testing without any data in the store. Remove it to trigger a store lookup by `entity_id`.

---

## Inference modes

**Unary**: one request, one prediction:

```bash
grpcurl -plaintext \
  -d '{"entity_id": "user_123"}' \
  localhost:50051 axon.inference.v1.InferenceService/Predict
```

Populate `features` in the request body to bypass the store.

**Streaming**: one subscription, predictions streamed as features update in the store:

```bash
grpcurl -plaintext \
  -d '{"entity_id": "user_123"}' \
  localhost:50051 axon.inference.v1.InferenceService/PredictStream
```

Ctrl-C to disconnect. Axon polls the store at `stream_poll_interval_ms` (configurable) and emits a response on each poll.

For Rust clients, see [`examples/client.rs`](examples/client.rs) (unary) and [`examples/streaming_client.rs`](examples/streaming_client.rs) (streaming). Run with:

```bash
cargo run --example client
cargo run --example streaming_client
```

For Python clients, first generate the stubs from the proto file:

```bash
pip install grpcio grpcio-tools
python -m grpc_tools.protoc \
  -I proto \
  --python_out=. \
  --grpc_python_out=. \
  proto/axon/inference/v1/inference.proto
```

Then see [`examples/client.py`](examples/client.py) (unary) and [`examples/streaming_client.py`](examples/streaming_client.py) (streaming). Run with:

```bash
python examples/client.py
python examples/streaming_client.py
```

---

## Configuration

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

[registry]
type          = "mlflow"
uri           = "http://localhost:5000"
model_name    = "my_model"
model_version = "1"            # version number or "latest"

[store]
type                       = "redis"
host                       = "localhost"
port                       = 6379
health_check_interval_secs = 10   # readiness probe polling interval

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

---

## Pipeline stages

| Stage | What it does | Parameters |
|---|---|---|
| `impute` | Replaces NaN values with a fixed default | `default_value` |
| `validate` | Rejects wrong shapes and non-finite values | `expected_shape` |
| `clip` | Clamps values to `[min, max]` before normalisation | `min`, `max` |
| `normalize` | Applies zero-mean unit-variance normalisation | `mean`, `std` |
| `infer` | Runs the model via the configured backend | (none) |
| `postprocess` | Transforms raw model output into a prediction | `threshold`, `output_type` |

Each stage accepts optional observability and reliability flags:

```toml
[[pipeline.stages]]
type         = "infer"
timed        = true    # records p99/p999 latency in Prometheus
instrumented = true    # emits a tracing span per execution
retries      = 3       # retries up to N times on failure
```

---

## Observability

Prometheus metrics are scraped from `http://localhost:<metrics.port>/metrics` (default port `9090`).

| Metric | Description |
|---|---|
| `axon_requests_total{rpc, status}` | Total requests by RPC and outcome |
| `axon_request_duration_seconds{rpc}` | End-to-end request latency |
| `axon_store_fetch_duration_seconds` | Feature store fetch latency |
| `axon_stage_p99_ns{stage}` | Per-stage p99 latency in nanoseconds |
| `axon_stage_p999_ns{stage}` | Per-stage p999 latency in nanoseconds |

Structured logs are written to stdout. Control verbosity with `RUST_LOG`:

```bash
RUST_LOG=info axon serve --config config.toml
RUST_LOG=axon=trace,warn axon serve --config config.toml
```

---

## Health checking

Axon implements the [gRPC health checking protocol](https://github.com/grpc/grpc/blob/master/doc/health-checking.md) with distinct liveness and readiness states.

**Liveness** (service `""`): `SERVING` as long as the process is running and the gRPC listener is bound.

**Readiness** (service `axon.inference.v1.InferenceService`): `SERVING` only when the backing feature store is confirmed reachable. Checked every `health_check_interval_secs`. Flips to `NOT_SERVING` after two consecutive failures to avoid flapping on transient blips.

```bash
# liveness
grpc_health_probe -addr=localhost:50051

# readiness
grpc_health_probe -addr=localhost:50051 \
  -service=axon.inference.v1.InferenceService
```

Kubernetes probe configuration:

```yaml
livenessProbe:
  grpc:
    port: 50051
  initialDelaySeconds: 5
  periodSeconds: 10

readinessProbe:
  grpc:
    port: 50051
    service: axon.inference.v1.InferenceService
  initialDelaySeconds: 10
  periodSeconds: 15
```

---

## Extensibility

Backends, stores, and registries are traits. Add a new implementation by writing the trait `impl` and adding a config enum variant. No changes to the pipeline or server required.

**Adding a new backend**

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

Then add `BackendConfig::Triton { url: String }` and a matching build arm in `src/pipeline/build.rs`.

The same pattern applies for:

| Trait | Implement to add |
|---|---|
| `FeatureStore` | A new feature store (e.g. Feast, Cassandra) |
| `ModelRegistryClient` | A new model registry (e.g. Vertex AI, custom HTTP) |

---

## Roadmap

- [x] OnnxBackend session pool for concurrent inference
- [ ] Triton Inference Server backend
- [ ] Event-driven streaming via Redis pub/sub (replaces polling)
- [ ] Add additional pipeline stages like `drift_detect`, `audit` and `argmax`
- [ ] WASM custom pipeline stages

---

## Development

```bash
git clone https://github.com/Ankithboggaram/axon
cd axon
cargo build --release
./target/release/axon serve --config config.toml
```

Requires Rust 1.85 or later. See [rust-toolchain.toml](rust-toolchain.toml).

---

## License

MIT
