# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

**Core inference**

- Config-driven preprocessing pipeline (impute, normalize, clip, validate, postprocess) — all stages declared in `config.toml`, no recompilation required to change them
- `OnnxBackend` for ONNX Runtime inference with graph optimization at Level3
- Execution provider selection: `cpu`, `coreml`, `cuda`, `tensorrt` via `[backend].device`
- Scratchpad pool and ONNX session pool for concurrent request handling with zero per-request heap allocation on the hot path
- Panic-free guarantee on the request path: `unwrap_used` and `expect_used` are compile errors in `src/`

**Serving**

- gRPC unary and server-streaming inference RPCs (`Predict` / `PredictStream`)
- `deadline_ms` per-request deadline enforcement
- Liveness vs readiness health check split — readiness tracks store reachability via a background ping; liveness reflects process health only
- `axon init` config seed generator for bootstrapping a valid `config.toml`

**Integrations**

- Redis feature store with pub/sub streaming (entity key subscriptions; no polling)
- MLflow model registry client — fetches and caches the model artifact at startup

**Error handling**

- Typed error enums (`ConfigError`, `BackendError`, `StoreError`, `RegistryError`, `PipelineError`, `ServeError`) at all public boundaries; `anyhow` confined to `main()`
- Structured CLI diagnostics for config load failures with field location and valid-values hints
- All public enums marked `#[non_exhaustive]` for forward compatibility

**Testing and quality**

- Property-based tests for all numerical pipeline stages with `proptest`
- ONNX integration tests using an MNIST-8 fixture model
- Fuzz targets for config parsing, pipeline construction, and tensor shapes
- Pipeline, backend, and pool benchmarks with `divan`

**Examples**

- Unary and streaming gRPC client examples in Rust and Python
