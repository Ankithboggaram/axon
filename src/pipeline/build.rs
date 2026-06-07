//! Wiring functions that construct the pipeline and scratchpad from config.

use std::sync::Arc;
use std::time::Duration;

use arrayvec::ArrayString;
use ndarray::{ArrayD, IxDyn};
use pipex::deadline::Deadline;
use pipex::dynamic_pipeline::Pipeline;
use pipex::instrument::Instrumented;
use pipex::metrics::{StageMetrics, Timed};
use pipex::retry::Retry;
use pipex::stage::Stage;

use crate::backend::Backend;
use crate::config::{Config, StageConfig, StageObservability};
use crate::pipeline::InferenceScratchpad;
use crate::pipeline::stages::clip::ClipStage;
use crate::pipeline::stages::impute::ImputeStage;
use crate::pipeline::stages::infer::InferStage;
use crate::pipeline::stages::normalize::NormalizeStage;
use crate::pipeline::stages::postprocess::PostprocessStage;
use crate::pipeline::stages::validate::ValidateStage;
use crate::types::{MAX_TENSOR_NAME_LEN, OutputBuffer};

/// Constructs a ready-to-run pipeline and its stage metrics from config.
///
/// Returns the pipeline and a vec of metrics handles in stage order, one per
/// stage that has `timed = true`. Register the handles with `Metrics::new` so
/// Prometheus can pull snapshots on each scrape.
pub fn build(
    config: &Config,
    backend: Arc<dyn Backend>,
) -> anyhow::Result<(Pipeline<InferenceScratchpad>, Vec<Arc<StageMetrics>>)> {
    validate_ordering(&config.pipeline.stages)?;

    let mut pipeline = Pipeline::new();
    let mut stage_metrics: Vec<Arc<StageMetrics>> = Vec::new();

    for stage_config in &config.pipeline.stages {
        match stage_config {
            StageConfig::Validate {
                expected_shape,
                observability,
            } => {
                if expected_shape.is_empty() {
                    anyhow::bail!("validate stage: expected_shape must not be empty");
                }
                for &dim in expected_shape {
                    if dim <= 0 {
                        anyhow::bail!(
                            "validate stage: all dimensions must be positive, got {}",
                            dim
                        );
                    }
                }
                let shape: Box<[usize]> = expected_shape
                    .iter()
                    .map(|&d| d as usize)
                    .collect::<Vec<_>>()
                    .into_boxed_slice();
                let stage = ValidateStage {
                    expected_shape: shape,
                };
                let (boxed, m) = wrap(Box::new(stage), observability);
                pipeline.push_boxed(boxed);
                if let Some(m) = m {
                    stage_metrics.push(m);
                }
            }

            StageConfig::Normalize {
                mean,
                std,
                observability,
            } => {
                let stage = NormalizeStage {
                    mean: *mean,
                    inv_std: 1.0 / std,
                };
                let (boxed, m) = wrap(Box::new(stage), observability);
                pipeline.push_boxed(boxed);
                if let Some(m) = m {
                    stage_metrics.push(m);
                }
            }

            StageConfig::Clip {
                min,
                max,
                observability,
            } => {
                if min >= max {
                    anyhow::bail!("clip stage: min ({}) must be less than max ({})", min, max);
                }
                let stage = ClipStage {
                    min: *min,
                    max: *max,
                };
                let (boxed, m) = wrap(Box::new(stage), observability);
                pipeline.push_boxed(boxed);
                if let Some(m) = m {
                    stage_metrics.push(m);
                }
            }

            StageConfig::Impute {
                default_value,
                observability,
            } => {
                let stage = ImputeStage {
                    default_value: *default_value,
                };
                let (boxed, m) = wrap(Box::new(stage), observability);
                pipeline.push_boxed(boxed);
                if let Some(m) = m {
                    stage_metrics.push(m);
                }
            }

            StageConfig::Infer { observability } => {
                let raw_name = &config.model_schema.inputs[0].name;
                let input_name = raw_name
                    .parse::<ArrayString<MAX_TENSOR_NAME_LEN>>()
                    .map_err(|_| {
                        anyhow::anyhow!(
                            "infer stage: input tensor name '{}' exceeds {} bytes",
                            raw_name,
                            MAX_TENSOR_NAME_LEN
                        )
                    })?;
                let stage = InferStage {
                    backend: Arc::clone(&backend),
                    input_name,
                };
                let (boxed, m) = wrap(Box::new(stage), observability);
                pipeline.push_boxed(boxed);
                if let Some(m) = m {
                    stage_metrics.push(m);
                }
            }

            StageConfig::Postprocess {
                threshold,
                output_type,
                observability,
            } => {
                let stage = PostprocessStage {
                    threshold: *threshold,
                    output_type: *output_type,
                };
                let (boxed, m) = wrap(Box::new(stage), observability);
                pipeline.push_boxed(boxed);
                if let Some(m) = m {
                    stage_metrics.push(m);
                }
            }
        }
    }

    Ok((pipeline, stage_metrics))
}

/// Pre-allocates a scratchpad from the model schema for use in a scratchpad pool.
pub fn build_scratchpad(config: &Config) -> anyhow::Result<InferenceScratchpad> {
    let input_shape: Vec<usize> = config.model_schema.inputs[0]
        .shape
        .iter()
        .map(|&d| d as usize)
        .collect();

    let outputs: Box<[OutputBuffer]> = config
        .model_schema
        .outputs
        .iter()
        .map(|spec| {
            let shape: Vec<usize> = spec.shape.iter().map(|&d| d as usize).collect();
            let name = spec
                .name
                .parse::<ArrayString<MAX_TENSOR_NAME_LEN>>()
                .map_err(|_| {
                    anyhow::anyhow!(
                        "output tensor name '{}' exceeds {} bytes",
                        spec.name,
                        MAX_TENSOR_NAME_LEN
                    )
                })?;
            Ok(OutputBuffer {
                name,
                data: ArrayD::zeros(IxDyn(&shape)),
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?
        .into_boxed_slice();

    Ok(InferenceScratchpad {
        entity_id: ArrayString::new(),
        request_id: ArrayString::new(),
        timestamp_ms: 0,
        input: ArrayD::zeros(IxDyn(&input_shape)),
        outputs,
    })
}

/// Checks that the stage sequence is structurally valid before building.
fn validate_ordering(stages: &[StageConfig]) -> anyhow::Result<()> {
    let infer_pos = stages
        .iter()
        .position(|s| matches!(s, StageConfig::Infer { .. }));

    if infer_pos.is_none() {
        anyhow::bail!("pipeline must contain an infer stage");
    }

    if let Some(post_pos) = stages
        .iter()
        .position(|s| matches!(s, StageConfig::Postprocess { .. }))
        && post_pos < infer_pos.unwrap()
    {
        anyhow::bail!("pipeline: postprocess stage must come after infer");
    }

    if let (Some(impute_pos), Some(validate_pos)) = (
        stages
            .iter()
            .position(|s| matches!(s, StageConfig::Impute { .. })),
        stages
            .iter()
            .position(|s| matches!(s, StageConfig::Validate { .. })),
    ) && impute_pos > validate_pos
    {
        anyhow::bail!("pipeline: impute stage must come before validate");
    }

    Ok(())
}

/// Applies observability wrappers to a stage.
///
/// Wrapping order: Retry -> Instrumented -> Deadline -> Timed.
/// Timed is outermost so it measures total latency including all inner wrappers.
/// Returns the wrapped stage and its metrics handle if `timed = true`.
fn wrap(
    stage: Box<dyn Stage<InferenceScratchpad>>,
    obs: &StageObservability,
) -> (
    Box<dyn Stage<InferenceScratchpad>>,
    Option<Arc<StageMetrics>>,
) {
    let stage: Box<dyn Stage<InferenceScratchpad>> = match obs.retries {
        Some(r) => Box::new(Retry::new(stage, r)),
        None => stage,
    };

    let stage: Box<dyn Stage<InferenceScratchpad>> = if obs.instrumented.unwrap_or(false) {
        Box::new(Instrumented::new(stage))
    } else {
        stage
    };

    let stage: Box<dyn Stage<InferenceScratchpad>> = match obs.deadline_ms {
        Some(ms) => Box::new(Deadline::new(stage, Duration::from_millis(ms))),
        None => stage,
    };

    if obs.timed.unwrap_or(false) {
        let (timed, metrics) = Timed::new(stage);
        (Box::new(timed), Some(metrics))
    } else {
        (stage, None)
    }
}
