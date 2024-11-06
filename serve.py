import sys

sys.path.append("./src")
import time
import threading
from concurrent import futures
import grpc
from app_pb2 import DummyResponse  # type: ignore # pylint:disable=E0611
from app_pb2_grpc import (
    DashServiceServicer,
    add_DashServiceServicer_to_server,
)  # Import gRPC generated classes
from src.app import app  # Your Dash app


class DashService(DashServiceServicer):
    def RunDashApp(self, request, context):
        return DummyResponse(response="This is a dummy response.")


def serve_grpc():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_DashServiceServicer_to_server(DashService(), server)
    server.add_insecure_port("[::]:8061")
    server.start()
    print("gRPC server running on port 8061")
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)


def run_dash():
    app.run_server(host="0.0.0.0", port=8062)  # Run Dash app on port 8062


if __name__ == "__main__":
    grpc_thread = threading.Thread(target=serve_grpc)
    grpc_thread.start()
    run_dash()
