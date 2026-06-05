//! Entry point. Wires everything together at startup.

pub mod backend;
pub mod config;
pub mod proto {
    tonic::include_proto!("axon.inference.v1");
}
pub mod metrics;
pub mod pipeline;
pub mod registry;
pub mod server;
pub mod store;
pub mod types;

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use clap::{Parser, Subcommand};
use tracing::{error, info};

use crate::backend::Backend;
use crate::backend::onnx::OnnxBackend;
use crate::backend::packaging::generate_triton_config;
use crate::config::{BackendType, Config, RegistryType, StoreType};
use crate::metrics::Metrics;
use crate::pipeline::build::{build, build_with_metrics};
use crate::proto::inference_service_server::InferenceServiceServer;
use crate::registry::ModelRegistryClient;
use crate::registry::mlflow::MlflowClient;
use crate::server::InferenceServer;
use crate::store::FeatureStore;
use crate::store::redis::RedisStore;

use crate::pipeline::InferenceScratchpad;
use pipex::dynamic_pipeline::Pipeline;
use pipex::pool::PipelinePool;

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
            let config = Config::load(&config_path)
                .map_err(|e| anyhow::anyhow!("failed to load config from '{config_path}': {e}"))?;
            info!(path = config_path, "config loaded");

            let registry: Arc<dyn ModelRegistryClient> = match config.registry.registry_type {
                RegistryType::Mlflow => Arc::new(MlflowClient::new(&config.registry.uri)?),
            };

            let store: Arc<dyn FeatureStore> = match config.store.store_type {
                StoreType::Redis => {
                    let url = format!("redis://{}:{}", config.store.host, config.store.port);
                    Arc::new(RedisStore::new(&url, "features")?)
                }
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

            let backend: Arc<dyn Backend> = match config.backend.backend_type {
                BackendType::OnnxRuntime => Arc::new(OnnxBackend::new(&model.local_path)?),
                BackendType::Triton => unreachable!("Triton rejected by Config::validate"),
            };

            let (first_pipeline, stage_metrics) = build(&config, Arc::clone(&backend))?;

            let metrics = Arc::new(Metrics::new(stage_metrics.clone())?);

            let metrics_port = config.metrics.port;
            tokio::spawn(serve_metrics(Arc::clone(&metrics), metrics_port));
            info!(port = metrics_port, "metrics server listening");

            store
                .ping()
                .await
                .map_err(|e| anyhow::anyhow!("startup readiness check failed: {e}"))?;
            info!("feature store reachable");

            let pool_size = config.grpc.pool_size.unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4)
            });

            let config_f = config.clone();
            let backend_f = Arc::clone(&backend);
            let metrics_f = stage_metrics.clone();
            let pool: PipelinePool<Pipeline<InferenceScratchpad>> =
                PipelinePool::new(pool_size, move || {
                    build_with_metrics(&config_f, Arc::clone(&backend_f), &metrics_f)
                        .expect("pool factory: failed to build pipeline")
                });

            // Pre-warm: acquire pool_size guards to trigger factory creation upfront,
            // then drop them so all pipelines are returned to the pool before serving.
            {
                let guards: Vec<_> = (0..pool_size.saturating_sub(1))
                    .map(|_| pool.acquire())
                    .collect();
                drop(guards);
            }
            // first_pipeline slots into the pool as the initial entry.
            drop(first_pipeline);

            let pool = Arc::new(pool);
            info!(pool_size, "pipeline pool ready");

            let (mut health_reporter, health_service) = tonic_health::server::health_reporter();
            health_reporter
                .set_serving::<InferenceServiceServer<InferenceServer>>()
                .await;

            let inference_server = InferenceServer::new(
                store,
                Arc::clone(&pool),
                metrics,
                config.grpc.stream_poll_interval_ms,
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

fn init_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
}

/// Serves Prometheus metrics over a minimal HTTP listener on the given port.
///
/// Any TCP connection receives the full metrics payload regardless of the
/// request path or method. Prometheus always scrapes GET /metrics, and we
/// have no other routes to protect.
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
            let body = metrics.render().unwrap_or_default();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/plain; version=0.0.4\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = stream.write_all(response.as_bytes()).await;
        });
    }
}
