"""Minimal streaming inference client.

Subscribes to continuous predictions for an entity and prints each response
as it arrives. Runs until Ctrl-C.

Usage:
    pip install grpcio grpcio-tools
    python -m grpc_tools.protoc \
        -I proto \
        --python_out=. \
        --grpc_python_out=. \
        proto/axon/inference/v1/inference.proto

    python examples/streaming_client.py
    python examples/streaming_client.py localhost:50052
"""

import sys

import grpc
from axon.inference.v1 import inference_pb2 as pb
from axon.inference.v1 import inference_pb2_grpc as stub

addr = sys.argv[1] if len(sys.argv) > 1 else "localhost:50051"

try:
    with grpc.insecure_channel(addr) as channel:
        client = stub.InferenceServiceStub(channel)
        for response in client.PredictStream(
            pb.PredictStreamRequest(entity_id="example-entity")
        ):
            for output in response.outputs:
                print(f"[{response.timestamp_ms}] {output.name}: {list(output.values)}")
except KeyboardInterrupt:
    pass
except grpc.RpcError as e:
    print(f"error: {e.details()}")
    print("hint: is the server running?  try: axon serve --config config.toml")
    sys.exit(1)
