#!/usr/bin/env python3
"""
Trains a logistic regression fraud-detection model, registers it to MLflow,
and seeds Redis with synthetic feature vectors.

Run this once before `axon init` to set up a working demo environment.

Requirements:
    pip install numpy scikit-learn onnx mlflow redis protobuf
    protoc on PATH
    a checkout of cortex-contract (default: ../cortex-contract relative to this repo)
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import ModuleType

import mlflow
import mlflow.onnx
import numpy as np
import onnx
import redis
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
from onnx import TensorProto, helper
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

N_FEATURES = 30
N_ENTITIES = 50


def load_feature_record_module(contract_dir: Path) -> ModuleType:
    """Compiles cortex-contract's feature_record.proto to a Python module.

    cortex-contract isn't published to PyPI, so (matching the contract repo's
    own scripts/check_roundtrip.py) this generates the Python type on the fly
    from the same .proto Rust compiles via prost — one wire definition, no
    hand-written mirror to drift out of sync.
    """
    if shutil.which("protoc") is None:
        sys.exit(
            "error: 'protoc' not found on PATH (required to read cortex-contract's proto/)"
        )

    proto_dir = contract_dir / "proto"
    proto_file = proto_dir / "cortex" / "contract" / "v1" / "feature_record.proto"
    if not proto_file.exists():
        sys.exit(
            f"error: {proto_file} not found\n"
            f"  hint: pass --contract-dir pointing at a cortex-contract checkout"
        )

    tmp_dir = tempfile.mkdtemp(prefix="axon-seed-demo-")
    subprocess.run(
        ["protoc", f"--python_out={tmp_dir}", "-I", str(proto_dir), str(proto_file)],
        check=True,
    )
    # protoc mirrors the proto's package path into the output dir, so the
    # generated module lands under <tmp_dir>/cortex/contract/v1/.
    sys.path.insert(0, str(Path(tmp_dir) / "cortex" / "contract" / "v1"))
    import feature_record_pb2  # noqa: PLC0415

    return feature_record_pb2


def build_onnx_model(coef: np.ndarray, intercept: np.ndarray) -> onnx.ModelProto:
    """Wraps logistic regression weights as an ONNX Gemm + Sigmoid graph.

    Input:  'features'  float32  [batch, N_FEATURES]
    Output: 'score'     float32  [batch, 1]

    Using sigmoid over raw logits gives a probability in [0, 1] that works
    directly with axon's binary postprocess stage and a 0.5 threshold.
    """
    W = coef.astype(np.float32)  # [1, N_FEATURES]
    b = intercept.astype(np.float32)  # [1]

    x_info = helper.make_tensor_value_info(
        "features", TensorProto.FLOAT, [None, N_FEATURES]
    )
    y_info = helper.make_tensor_value_info("score", TensorProto.FLOAT, [None, 1])

    W_init = helper.make_tensor("W", TensorProto.FLOAT, W.shape, W.flatten().tolist())
    b_init = helper.make_tensor("b", TensorProto.FLOAT, b.shape, b.flatten().tolist())

    gemm = helper.make_node("Gemm", ["features", "W", "b"], ["logit"], transB=1)
    sigmoid = helper.make_node("Sigmoid", ["logit"], ["score"])

    graph = helper.make_graph(
        [gemm, sigmoid],
        "fraud_demo",
        [x_info],
        [y_info],
        initializer=[W_init, b_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the axon demo environment")
    parser.add_argument("--mlflow-uri", default="http://localhost:5000")
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", default=6379, type=int)
    parser.add_argument("--key-prefix", default="features")
    parser.add_argument("--model-name", default="fraud_demo")
    parser.add_argument(
        "--contract-dir",
        default=Path(__file__).resolve().parent.parent.parent / "cortex-contract",
        type=Path,
        help="path to a cortex-contract checkout (default: ../cortex-contract)",
    )
    args = parser.parse_args()

    fr_pb2 = load_feature_record_module(args.contract_dir)

    # Synthetic tabular data: 3% fraud rate, similar to real transaction data.
    X, y = make_classification(
        n_samples=1000,
        n_features=N_FEATURES,
        n_informative=15,
        n_redundant=5,
        weights=[0.97, 0.03],
        random_state=42,
    )
    X = X.astype(np.float32)

    mean = float(round(float(X.mean()), 4))
    std = float(round(float(X.std()), 4))

    clf = LogisticRegression(max_iter=300, class_weight="balanced")
    clf.fit(X, y)

    onnx_model = build_onnx_model(clf.coef_, clf.intercept_)

    # Log to MLflow. artifact_path="model" means the model version's artifact
    # root will be {run}/artifacts/model/, so axon can find model.onnx and
    # MLmodel at the paths it expects.
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment("axon-demo")

    # Use batch size 1 (not -1) — axon always serves single entities, and
    # build_scratchpad casts shape dims to usize; -1 would wrap to usize::MAX.
    signature = ModelSignature(
        inputs=Schema([TensorSpec(np.dtype("float32"), (1, N_FEATURES), "features")]),
        outputs=Schema([TensorSpec(np.dtype("float32"), (1, 1), "score")]),
    )

    with mlflow.start_run() as run:
        # axon init reads these params to pre-fill config.toml.
        mlflow.log_params(
            {
                "mean": mean,
                "std": std,
                "clip_min": -3.0,
                "clip_max": 3.0,
                "threshold": 0.5,
            }
        )
        mlflow.onnx.log_model(onnx_model, artifact_path="model", signature=signature)
        run_id = run.info.run_id

    version = mlflow.register_model(f"runs:/{run_id}/model", args.model_name)
    print(
        f"Registered '{args.model_name}' v{version.version} to MLflow at {args.mlflow_uri}"
    )

    # Seed Redis with protobuf(FeatureRecord)-encoded vectors under features:{entity_id},
    # the same wire format cortex-contract's Rust codec decodes on Axon's read path.
    # schema_version is a demo placeholder; Phase B is what makes Axon check it against
    # the model's trained version.
    r = redis.Redis(host=args.redis_host, port=args.redis_port, decode_responses=False)
    now_ms = int(time.time() * 1000)
    for i in range(N_ENTITIES):
        key = f"{args.key_prefix}:entity_{i:04d}"
        record = fr_pb2.FeatureRecord(
            schema_version=1,
            event_time_ms=now_ms,
            features=X[i % len(X)].tolist(),
        )
        r.set(key, record.SerializeToString())
    print(
        f"Seeded {N_ENTITIES} entities in Redis at {args.redis_host}:{args.redis_port}"
    )

    print(f"""
Run next:
  docker run --rm --network host -v $(pwd):/output \\
    axon init {args.model_name} {version.version} \\
    --registry-uri {args.mlflow_uri} \\
    --output /output/config.toml

Then open config.toml and set registry.uri to {args.mlflow_uri}.
All other values are pre-filled from the run params logged above.
""")


if __name__ == "__main__":
    main()
