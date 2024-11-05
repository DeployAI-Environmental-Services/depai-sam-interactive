import sys

sys.path.append("./src")
from concurrent import futures
import time
import threading
import grpc
from src.app import app


class MyServiceServicer:
    pass


def serve_grpc():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
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
