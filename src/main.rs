//! Entry point: wires together all subsystems and starts the gRPC server.
//!
//! ## Startup sequence
//!
//! 1. Parse CLI args and load `config.toml` via [`axon::config::Config::load`]
//! 2. Construct the model registry client and fetch the ONNX artifact
//! 3. Initialise the inference backend ([`axon::backend::onnx::OnnxBackend`])
//! 4. Build the processing pipeline from config ([`axon::pipeline::build::build`])
//! 5. Register Prometheus metrics and spawn the metrics HTTP listener
//! 6. Ping the feature store (readiness is NOT set to `SERVING` until this passes)
//! 7. Pre-populate the scratchpad and pipeline pools
//! 8. If `[predictions]` is configured and enabled, construct the
//!    [`axon::predictions::PredictionSink`] (spawns its Kafka-draining task)
//! 9. Bind the gRPC listener; set liveness (`""`) and readiness
//!    (`axon.inference.v1.InferenceService`) to `SERVING`
//! 10. Spawn a background task that pings the store every
//!     `store.health_check_interval_secs` seconds; two consecutive failures flip
//!     readiness to `NOT_SERVING`; recovery flips it back to `SERVING`
//!
//! ## Liveness vs readiness
//!
//! The empty-string service (`""`) represents the overall process. It is set to
//! `SERVING` once at startup and never changed; a Kubernetes liveness probe
//! uses this to determine whether to restart the pod.
//!
//! The named service (`axon.inference.v1.InferenceService`) represents
//! readiness: can this instance actually serve traffic right now? The background
//! health task drives it. A readiness probe that fails causes the pod to be
//! removed from load balancer rotation without restarting it.

#![deny(clippy::unwrap_used, clippy::expect_used)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use clap::{Parser, Subcommand};
use tonic_health::ServingStatus;
use tracing::{error, info};

use axon::backend::Backend;
use axon::backend::onnx::OnnxBackend;
use axon::backend::packaging::generate_triton_config;
use axon::config::{BackendType, Config, RegistryType, StoreType};
use axon::metrics::Metrics;
use axon::pipeline::build::{build, build_scratchpad};
use axon::pipeline::pool::PipelinePool;
use axon::predictions::PredictionSink;
use axon::proto::inference_service_server::InferenceServiceServer;
use axon::registry::ModelRegistryClient;
use axon::registry::mlflow::MlflowClient;
use axon::server::InferenceServer;
use cortex_contract::store::OnlineStoreReader;
use cortex_contract::store::redis::RedisOnlineStore;
use pipexec::pool::ScratchpadPool;

#[derive(Parser)]
#[command(name = "axon", about = "A configurable real-time ML inference engine")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Start the inference server.
    Serve {
        #[arg(long, default_value = "config.toml")]
        config: String,
    },
    /// Generate a starter config.toml from a model registry.
    Init {
        model_name: String,
        model_version: String,
        #[arg(long, default_value = "mlflow")]
        registry_type: String,
        #[arg(long)]
        registry_uri: String,
        #[arg(long, default_value = "config.toml")]
        output: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    init_tracing();

    match cli.command {
        Command::Init {
            model_name,
            model_version,
            registry_type,
            registry_uri,
            output,
        } => {
            let registry: Arc<dyn ModelRegistryClient> = match registry_type.as_str() {
                "mlflow" => Arc::new(MlflowClient::new(&registry_uri)?),
                other => anyhow::bail!("unknown registry type: '{other}'"),
            };

            info!(
                model_name,
                model_version, "fetching config seed from registry"
            );
            let seed = registry
                .fetch_config_seed(&model_name, &model_version)
                .await?;
            let config_text = seed.generate_config(&model_name, &model_version);

            std::fs::write(&output, &config_text)
                .map_err(|e| anyhow::anyhow!("failed to write config to '{output}': {e}"))?;

            info!(
                path = output,
                "config written; fill in all TODO values before running axon serve"
            );
        }

        Command::Serve {
            config: config_path,
        } => {
            let config =
                Config::load(&config_path).map_err(|e| report_config_error(&config_path, e))?;
            info!(path = config_path, "config loaded");

            let registry: Arc<dyn ModelRegistryClient> = match config.registry.registry_type {
                RegistryType::Mlflow => Arc::new(MlflowClient::new(&config.registry.uri)?),
                _ => anyhow::bail!("unsupported registry type"),
            };

            let store: Arc<dyn OnlineStoreReader> = match config.store.store_type {
                StoreType::Redis => {
                    let key_prefix = config
                        .store
                        .key_prefix
                        .as_deref()
                        .unwrap_or(cortex_contract::keys::DEFAULT_KEY_PREFIX);
                    Arc::new(RedisOnlineStore::new(&config.store.url, key_prefix)?)
                }
                _ => anyhow::bail!("unsupported store type"),
            };

            info!(
                model = config.registry.model_name,
                version = config.registry.model_version,
                "fetching model from registry"
            );
            let model = registry
                .fetch_model(&config.registry.model_name, &config.registry.model_version)
                .await?;
            info!(
                model = model.name,
                version = model.version,
                path = model.local_path,
                "model artifact ready"
            );

            // The registry's schema_version tag wins; the config value is only
            // a fallback for registries or models that don't stamp one.
            let expected_schema_version =
                model.schema_version.or(config.model_schema.schema_version);
            match expected_schema_version {
                Some(v) => info!(schema_version = v, "schema-version enforcement enabled"),
                None => info!(
                    "no schema_version available from the registry or config; \
                     schema-version enforcement disabled"
                ),
            }

            // Generate Triton config.pbtxt. Non-fatal: ONNX backend does not require it.
            match generate_triton_config(&model.name, &config.model_schema) {
                Ok(pbtxt) => {
                    let dir = format!("models/{}", model.name);
                    match std::fs::create_dir_all(&dir) {
                        Err(e) => tracing::warn!("could not create Triton model dir '{dir}': {e}"),
                        Ok(()) => {
                            let path = format!("{dir}/config.pbtxt");
                            match std::fs::write(&path, &pbtxt) {
                                Err(e) => {
                                    tracing::warn!("could not write Triton config to '{path}': {e}")
                                }
                                Ok(()) => info!(path, "Triton config.pbtxt written"),
                            }
                        }
                    }
                }
                Err(e) => tracing::warn!("could not generate Triton config: {e}"),
            }

            let session_pool_size = config
                .grpc
                .session_pool_size
                .unwrap_or_else(default_pool_size);

            let backend: Arc<dyn Backend> = match config.backend.backend_type {
                BackendType::OnnxRuntime => Arc::new(OnnxBackend::new(
                    &model.local_path,
                    session_pool_size,
                    config.backend.device.clone(),
                )?),
                _ => unreachable!("unsupported backend rejected by Config::validate"),
            };
            info!(session_pool_size, device = ?config.backend.device, "session pool ready");

            let (first_pipeline, stage_metrics) = build(&config, Arc::clone(&backend))?;

            let metrics = Arc::new(Metrics::new(stage_metrics)?);

            let metrics_port = config.metrics.port;
            tokio::spawn(serve_metrics(Arc::clone(&metrics), metrics_port));
            info!(port = metrics_port, "metrics server listening");

            store
                .ping()
                .await
                .map_err(|e| anyhow::anyhow!("startup readiness check failed: {e}"))?;
            info!("feature store reachable");

            let pool_size = config.grpc.pool_size.unwrap_or_else(default_pool_size);

            let config_s = config.clone();
            // FnMut() -> T has no error channel; expect is the only option here.
            #[allow(clippy::expect_used)]
            let scratchpad_factory =
                move || build_scratchpad(&config_s).expect("scratchpad pool factory failed");
            let scratchpad_pool = Arc::new(ScratchpadPool::new(pool_size, scratchpad_factory));

            let config_p = config.clone();
            let backend_p = Arc::clone(&backend);
            #[allow(clippy::expect_used)]
            let pipeline_factory = move || {
                build(&config_p, Arc::clone(&backend_p))
                    .expect("pipeline pool factory failed")
                    .0
            };
            let pipeline_pool = Arc::new(PipelinePool::new(
                first_pipeline,
                pool_size,
                pipeline_factory,
            ));

            info!(pool_size, "pipeline pool ready");

            let (mut health_reporter, health_service) = tonic_health::server::health_reporter();
            // Liveness: the process is alive. Never changes after this point.
            health_reporter
                .set_service_status("", ServingStatus::Serving)
                .await;
            // Readiness: the store ping just passed, so this instance can serve traffic.
            health_reporter
                .set_serving::<InferenceServiceServer<InferenceServer>>()
                .await;

            let health_interval =
                Duration::from_secs(config.store.health_check_interval_secs.unwrap_or(10));
            tokio::spawn(store_health_check(
                Arc::clone(&store),
                health_reporter.clone(),
                health_interval,
            ));

            let predictions = match &config.predictions {
                Some(p) if p.enabled => {
                    let sink = PredictionSink::new(p, Arc::clone(&metrics))
                        .map_err(|e| anyhow::anyhow!("failed to set up prediction logging: {e}"))?;
                    info!(
                        brokers = p.brokers,
                        topic = p.topic,
                        "prediction logging enabled"
                    );
                    Some(Arc::new(sink))
                }
                _ => None,
            };

            let inference_server = InferenceServer::new(
                store,
                pipeline_pool,
                scratchpad_pool,
                metrics,
                config.grpc.stream_poll_interval_ms,
                config.freshness.clone(),
                expected_schema_version,
                predictions,
                model.name.clone(),
                model.version.clone(),
            );

            let grpc_addr: SocketAddr = format!("{}:{}", config.grpc.host, config.grpc.port)
                .parse()
                .map_err(|e| anyhow::anyhow!("invalid gRPC address: {e}"))?;

            let timeout = Duration::from_millis(config.grpc.request_timeout_ms);

            info!(addr = %grpc_addr, "gRPC server listening");

            let serve = tonic::transport::Server::builder()
                .layer(tower::ServiceBuilder::new().timeout(timeout).into_inner())
                .add_service(health_service)
                .add_service(InferenceServiceServer::new(inference_server))
                .serve(grpc_addr);

            tokio::select! {
                result = serve => {
                    if let Err(e) = result {
                        error!("gRPC server error: {e}");
                    }
                }
                _ = tokio::signal::ctrl_c() => {
                    info!("shutdown signal received, stopping");
                }
            }
        }
    }

    Ok(())
}

fn report_config_error(path: &str, e: axon::error::ConfigError) -> anyhow::Error {
    use axon::error::ConfigError;
    match e {
        ConfigError::Io(io) => anyhow::anyhow!(
            "could not read config file '{path}'\n  cause:  {io}\n  hint:   check the path is correct and the file is readable"
        ),
        ConfigError::Parse(p) => anyhow::anyhow!(
            "config file '{path}' contains invalid TOML\n  cause:  {p}\n  hint:   run `axon init` to generate a valid starter config"
        ),
        ConfigError::Invalid { field, reason } => anyhow::anyhow!(
            "config file '{path}' failed validation\n  field:  {field}\n  reason: {reason}\n  hint:   run `axon init` to generate a valid starter config"
        ),
        _ => anyhow::anyhow!("config file '{path}' failed to load: {e:?}"),
    }
}

fn default_pool_size() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

fn init_tracing() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));

    // Set AXON_LOG_JSON=1 for structured JSON output (recommended in production).
    let json = std::env::var("AXON_LOG_JSON")
        .map(|v| v == "1" || v == "true")
        .unwrap_or(false);

    if json {
        tracing_subscriber::fmt()
            .json()
            .with_env_filter(filter)
            .init();
    } else {
        tracing_subscriber::fmt().with_env_filter(filter).init();
    }
}

async fn store_health_check(
    store: Arc<dyn OnlineStoreReader>,
    mut health_reporter: tonic_health::server::HealthReporter,
    interval: Duration,
) {
    let mut consecutive_failures: u32 = 0;
    loop {
        tokio::time::sleep(interval).await;
        match store.ping().await {
            Ok(()) => {
                if consecutive_failures >= 2 {
                    health_reporter
                        .set_serving::<InferenceServiceServer<InferenceServer>>()
                        .await;
                    info!("feature store reachable again; readiness restored");
                }
                consecutive_failures = 0;
            }
            Err(e) => {
                consecutive_failures += 1;
                tracing::warn!(
                    consecutive_failures,
                    "feature store health check failed: {e}"
                );
                if consecutive_failures == 2 {
                    health_reporter
                        .set_not_serving::<InferenceServiceServer<InferenceServer>>()
                        .await;
                    tracing::warn!(
                        "feature store unreachable for 2 consecutive checks; \
                         readiness set to NOT_SERVING"
                    );
                }
            }
        }
    }
}

async fn serve_metrics(metrics: Arc<Metrics>, port: u16) {
    use tokio::io::AsyncWriteExt as _;

    let addr = format!("0.0.0.0:{port}");
    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            error!("failed to bind metrics listener on {addr}: {e}");
            return;
        }
    };

    loop {
        let (mut stream, _) = match listener.accept().await {
            Ok(s) => s,
            Err(e) => {
                error!("metrics accept error: {e}");
                continue;
            }
        };

        let metrics = Arc::clone(&metrics);
        tokio::spawn(async move {
            let body = metrics.render().unwrap_or_else(|e| {
                tracing::warn!("metrics encode failed: {e}");
                String::new()
            });
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/plain; version=0.0.4\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = stream.write_all(response.as_bytes()).await;
        });
    }
}
