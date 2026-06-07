//! Generates Triton Inference Server config.pbtxt from the model schema.

use std::fmt::Write as _;

use crate::config::ModelSchemaConfig;

/// Generates a Triton config.pbtxt string from the model name and schema.
///
/// The caller is responsible for writing the returned string to the correct
/// path inside the Triton model repository (e.g. `models/<name>/config.pbtxt`).
#[allow(clippy::unwrap_used)] // writeln! to String is infallible; Write::write_str always returns Ok
pub fn generate_triton_config(
    model_name: &str,
    schema: &ModelSchemaConfig,
) -> anyhow::Result<String> {
    // Validate all dtypes up front so we never produce a partial config.
    let input_dtypes = schema
        .inputs
        .iter()
        .map(|t| triton_dtype(&t.dtype))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let output_dtypes = schema
        .outputs
        .iter()
        .map(|t| triton_dtype(&t.dtype))
        .collect::<anyhow::Result<Vec<_>>>()?;

    let mut out = String::new();

    writeln!(out, "name: \"{model_name}\"").unwrap();
    writeln!(out, "backend: \"onnxruntime\"").unwrap();
    writeln!(out, "max_batch_size: 0").unwrap();

    write_tensor_block(&mut out, "input", &schema.inputs, &input_dtypes);
    write_tensor_block(&mut out, "output", &schema.outputs, &output_dtypes);

    Ok(out)
}

/// Writes a Triton `input [...]` or `output [...]` block for a set of tensors.
#[allow(clippy::unwrap_used)] // writeln! to String is infallible; Write::write_str always returns Ok
fn write_tensor_block(
    out: &mut String,
    field: &str,
    tensors: &[crate::config::TensorSpec],
    dtypes: &[&str],
) {
    writeln!(out, "\n{field} [").unwrap();
    for (i, (tensor, dtype)) in tensors.iter().zip(dtypes).enumerate() {
        let mut dims = String::new();
        for (i, d) in tensor.shape.iter().enumerate() {
            if i > 0 {
                dims.push_str(", ");
            }
            dims.push_str(&d.to_string());
        }
        writeln!(out, "  {{").unwrap();
        writeln!(out, "    name: \"{}\"", tensor.name).unwrap();
        writeln!(out, "    data_type: {dtype}").unwrap();
        writeln!(out, "    dims: [ {dims} ]").unwrap();
        if i < tensors.len() - 1 {
            writeln!(out, "  }},").unwrap();
        } else {
            writeln!(out, "  }}").unwrap();
        }
    }
    writeln!(out, "]").unwrap();
}

/// Maps a dtype string from the model schema to a Triton data type token.
fn triton_dtype(dtype: &str) -> anyhow::Result<&'static str> {
    match dtype {
        "float32" => Ok("TYPE_FP32"),
        "float64" => Ok("TYPE_FP64"),
        "int8" => Ok("TYPE_INT8"),
        "int16" => Ok("TYPE_INT16"),
        "int32" => Ok("TYPE_INT32"),
        "int64" => Ok("TYPE_INT64"),
        "uint8" => Ok("TYPE_UINT8"),
        "uint16" => Ok("TYPE_UINT16"),
        "uint32" => Ok("TYPE_UINT32"),
        "uint64" => Ok("TYPE_UINT64"),
        "bool" => Ok("TYPE_BOOL"),
        "string" => Ok("TYPE_STRING"),
        other => anyhow::bail!("packaging: unsupported dtype '{other}'"),
    }
}
