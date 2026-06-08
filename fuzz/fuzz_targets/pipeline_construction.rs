#![no_main]

use arbitrary::Arbitrary;
use axon::config::{OutputType, StageConfig, StageObservability};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
enum FuzzStage {
    Validate { dims: Vec<u8> },
    Normalize { mean: f32, std: f32 },
    Clip { min: f32, max: f32 },
    Impute { default_value: f32 },
    Infer,
    Postprocess { threshold: f32, output_type: u8 },
}

fn obs() -> StageObservability {
    StageObservability {
        timed: None,
        instrumented: None,
        retries: None,
        deadline_ms: None,
    }
}

impl FuzzStage {
    fn into_stage_config(self) -> StageConfig {
        match self {
            FuzzStage::Validate { dims } => {
                // Ensure at least one dimension so expected_shape is non-empty.
                let shape: Vec<i64> = if dims.is_empty() {
                    vec![1]
                } else {
                    dims.iter().map(|&d| (d as i64).max(1)).collect()
                };
                StageConfig::Validate {
                    expected_shape: shape,
                    observability: obs(),
                }
            }
            FuzzStage::Normalize { mean, std } => StageConfig::Normalize {
                mean,
                std,
                observability: obs(),
            },
            FuzzStage::Clip { min, max } => StageConfig::Clip {
                min,
                max,
                observability: obs(),
            },
            FuzzStage::Impute { default_value } => StageConfig::Impute {
                default_value,
                observability: obs(),
            },
            FuzzStage::Infer => StageConfig::Infer { observability: obs() },
            FuzzStage::Postprocess {
                threshold,
                output_type,
            } => StageConfig::Postprocess {
                threshold,
                output_type: match output_type % 3 {
                    0 => OutputType::Binary,
                    1 => OutputType::Probability,
                    _ => OutputType::Raw,
                },
                observability: obs(),
            },
        }
    }
}

fuzz_target!(|stages: Vec<FuzzStage>| {
    let stage_configs: Vec<StageConfig> = stages
        .into_iter()
        .map(FuzzStage::into_stage_config)
        .collect();
    let _ = axon::pipeline::build::validate_ordering(&stage_configs);
});
