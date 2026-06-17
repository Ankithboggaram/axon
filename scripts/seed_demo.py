#!/usr/bin/env python3
"""
Trains a logistic regression fraud-detection model, registers it to MLflow,
and seeds Redis with synthetic feature vectors.

Run this once before `axon init` to set up a working demo environment.

Requirements:
    pip install numpy scikit-learn onnx mlflow redis msgpack-python
"""

import argparse

import mlflow
import mlflow.onnx
import msgpack
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
    args = parser.parse_args()

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

    # Seed Redis with MessagePack-encoded float32 vectors under features:{entity_id}.
    r = redis.Redis(host=args.redis_host, port=args.redis_port, decode_responses=False)
    for i in range(N_ENTITIES):
        key = f"{args.key_prefix}:entity_{i:04d}"
        features = X[i % len(X)].tolist()
        # use_single_float=True encodes as float32; axon deserializes Vec<f32>
        # and rmp_serde does not coerce float64 → float32 automatically.
        r.set(key, msgpack.packb(features, use_single_float=True))
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
