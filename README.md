# axon

A configurable real-time ML inference engine written in Rust. Axon sits between your feature store and your clients. It fetches pre-computed features, runs them through a TOML-configured preprocessing pipeline, executes a model, and returns predictions over gRPC.

Part of the **Cortex** platform. Designed to work alongside Dendrite (streaming feature pipeline) and Synapse (training pipeline), but deployable independently.

---

## Prerequisites

- An ONNX model registered in a model registry
- A feature store populated with feature vectors for your entities
- Rust toolchain

---

## Getting started

The steps below use the built-in implementations: MLflow as the model registry, Redis as the feature store, and ONNX Runtime as the inference backend. See [Extensibility](#extensibility) to swap any of them out.

**Step 1: Generate a starter config**

```bash
axon init <model_name> <model_version> \
  --registry-uri http://localhost:5000 \
  --output config.toml
```

This connects to MLflow, reads the model signature and logged training parameters, and writes a `config.toml` pre-filled with whatever it can infer. Fields it cannot determine are written as `TODO` placeholders.

**Step 2: Fill in the TODOs**

Open `config.toml` and complete any remaining `TODO` values, typically the preprocessing parameters (`mean`, `std`, `clip.min`, `clip.max`) if they were not logged to the registry during training.

**Step 3: Start the server**

```bash
axon serve --config config.toml
```

Axon fetches the model artifact from MLflow, loads it into ONNX Runtime, pings Redis to confirm connectivity, and starts accepting gRPC connections.

---

## Configuration

All runtime behaviour is driven by `config.toml`. No recompilation required to change pipeline stages, backends, or infrastructure connections.

```toml
[grpc]
host = "0.0.0.0"
port = 50051
stream_poll_interval_ms = 1    # polling interval for streaming RPCs
request_timeout_ms = 5000      # requests exceeding this are cancelled

[backend]
type = "onnx_runtime"          # see Extensibility for adding new backends

[registry]
type = "mlflow"                # see Extensibility for adding new registries
uri = "http://localhost:5000"
model_name = "my_model"
model_version = "1"            # version number or "latest"

[store]
type = "redis"                 # see Extensibility for adding new stores
host = "localhost"
port = 6379

[metrics]
port = 9090

# Tensor names, types, and shapes must match the ONNX model exactly.
[[model_schema.inputs]]
name  = "features"
dtype = "float32"
shape = [1, 32]

[[model_schema.outputs]]
name  = "score"
dtype = "float32"
shape = [1, 1]

# Stages run in the order they appear.
[[pipeline.stages]]
type          = "impute"
default_value = 0.0

[[pipeline.stages]]
type           = "validate"
expected_shape = [1, 32]

[[pipeline.stages]]
type = "normalize"
mean = 0.0
std  = 1.0

[[pipeline.stages]]
type = "infer"

[[pipeline.stages]]
type        = "postprocess"
threshold   = 0.5
output_type = "binary"         # "binary" (±1) | "probability" | "raw"
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

Each stage accepts optional observability flags:

```toml
[[pipeline.stages]]
type         = "infer"
timed        = true   # records p99/p999 latency via Prometheus
instrumented = true   # emits a tracing span on each execution
retries      = 3      # retries up to N times on failure
```

---

## Observability

Prometheus metrics are exposed on `metrics.port` (default `9090`).

| Metric | Description |
|---|---|
| `axon_requests_total{rpc, status}` | Total requests by RPC and outcome |
| `axon_request_duration_seconds{rpc}` | End-to-end request latency |
| `axon_store_fetch_duration_seconds` | Feature store fetch latency |
| `axon_stage_p99_ns{stage}` | Stage p99 latency in nanoseconds |
| `axon_stage_p999_ns{stage}` | Stage p999 latency in nanoseconds |

Structured logs are written to stdout. Control verbosity via `RUST_LOG`:

```bash
RUST_LOG=debug axon serve --config config.toml
```

---

## Inference modes

**Unary**: send a `PredictRequest` with an `entity_id`, get back one `PredictResponse`. Optionally populate the `features` field directly to bypass the feature store lookup.

**Streaming**: send a `PredictStreamRequest` with an `entity_id`. Axon polls the feature store at `stream_poll_interval_ms` and streams a `PredictStreamResponse` on each poll until the client disconnects.

---

## gRPC health check

Axon implements the standard [gRPC health checking protocol](https://github.com/grpc/grpc/blob/master/doc/health-checking.md). The health status is set to `SERVING` only after all components are initialised and the feature store is confirmed reachable.

```bash
grpc_health_probe -addr=localhost:50051
```

---

## Extensibility

Axon's backends, stores, and registries are all trait-based. New implementations require no changes to the pipeline or serving logic. Add a new variant to the relevant config enum and implement the trait.

| Trait | Implement to add |
|---|---|
| `Backend` | A new inference runtime (e.g. Triton, TensorRT) |
| `FeatureStore` | A new feature store (e.g. Feast, Cassandra) |
| `ModelRegistryClient` | A new model registry (e.g. Vertex AI, custom) |

---

## Roadmap

- [ ] Triton Inference Server backend
- [ ] Event-driven streaming via Redis pub/sub (replaces polling)
- [ ] Scratchpad pool for concurrent request processing
- [ ] OnnxBackend session pool for concurrent inference
- [ ] Registry-driven preprocessing parameter loading
- [ ] `drift_detect`, `audit`, `argmax` pipeline stages
- [ ] WASM custom pipeline stages
- [ ] Async-native pipeline (requires pipex async stage support)
