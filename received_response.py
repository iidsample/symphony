import os
import sys
import json
import queue
import grpc
import argparse
from concurrent import futures


sys.path.append(os.path.join(os.path.dirname(__file__), "rpc"))
sys.path.append(os.path.join(os.path.dirname(__file__), "rpc", "grpc_stubs"))
print("Sys Path {}".format(sys.path))
from rpc.grpc_stubs import datagen_pb2_grpc
from rpc.grpc_stubs import datagen_pb2


class WorkloadReciever(datagen_pb2_grpc.DataServerServicer):
    """
    Recieve response
    """

    def __init__(self):
        self.response_queue = queue.Queue()  # store responses here

    def PostResponse(self, request, context):
        """
        Get a request from llm server
        """
        recieved_info = json.loads(request.response)
        self.response_queue.put(recieved_info)
        return_request = datagen_pb2.JsonResponse()
        return_request.response = json.dumps({"recieved": True})

        return return_request

    def poll_response(self):
        """
        Return all response back to the data generator
        """
        all_responses = []
        while True:
            try:
                task = self.response_queue.get(block=False)
                # task = json.loads(task.response)
                all_responses.append(task)
            except queue.Empty:
                break
        return all_responses


def parse_args(parser):
    """
    Parse Arguments
    """
    parser.add_argument(
        "--loadgen-rpc-port", type=str, default=50050, help="Load Generation RPC Port"
    )

    parser.add_argument("--num-clients", type=str, default=64, help="Number of Clients")
    args = parser.parse_args()
    return args


def launch_server(args) -> grpc.Server:
    """
    Launches the server
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    datagen_pb2_grpc.add_DataServerServicer_to_server(
        WorkloadReciever(num_clients=args.num_clients), server
    )
    server.add_insecure_port(f"[::]:{args.loadgen_rpc_port}")
    server.start()
    print("Workload Server Started")
    return server


def start_server(workload_rcvr: WorkloadReciever, reciever_port: str):
    """
    Start Server
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    datagen_pb2_grpc.add_DataServerServicer_to_server(workload_rcvr, server)
    server.add_insecure_port(f"[::]:{reciever_port}")
    server.start()
    return server


if __name__ == "__main__":
    args = parse_args(
        argparse.ArgumentParser(description="Arguments for Data Generator")
    )

    try:
        server = launch_server(args)
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(0)
