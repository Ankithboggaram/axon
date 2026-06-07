"""Minimal unary inference client.

Sends one Predict call with inline features (no feature store required)
and prints the response.

Usage:
    pip install grpcio grpcio-tools
    python -m grpc_tools.protoc \
        -I proto \
        --python_out=. \
        --grpc_python_out=. \
        proto/axon/inference/v1/inference.proto

    python examples/client.py
    python examples/client.py localhost:50052
"""

import sys

import grpc
from axon.inference.v1 import inference_pb2 as pb
from axon.inference.v1 import inference_pb2_grpc as stub

addr = sys.argv[1] if len(sys.argv) > 1 else "localhost:50051"

try:
    with grpc.insecure_channel(addr) as channel:
        client = stub.InferenceServiceStub(channel)
        response = client.Predict(
            pb.PredictRequest(
                entity_id="example-entity",
                features=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            )
        )
        print(f"entity_id:    {response.entity_id}")
        print(f"timestamp_ms: {response.timestamp_ms}")
        for output in response.outputs:
            print(
                f"output '{output.name}': {list(output.values)}  shape={list(output.shape)}"
            )
except grpc.RpcError as e:
    print(f"error: {e.details()}")
    print("hint: is the server running?  try: axon serve --config config.toml")
    sys.exit(1)
