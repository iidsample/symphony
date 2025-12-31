# this should be called global scheduler frontend
import os
import sys
import json
import time
import queue
import grpc
import argparse
from typing import Tuple
from concurrent import futures


sys.path.append(os.path.join(os.path.dirname(__file__), "rpc"))
sys.path.append(os.path.join(os.path.dirname(__file__), "rpc", "grpc_stubs"))
from rpc.grpc_stubs import scheduler_pb2_grpc
from rpc.grpc_stubs import datagen_pb2, datagen_pb2_grpc


class SchedulerServer(scheduler_pb2_grpc.SchedulerServerServicer):
    """
    Recieve LLM Inference Server
    """

    def __init__(self, workload_gen_ip, workload_gen_port):
        self.input_request_queue = queue.Queue()  # store responses here
        self.finished_request_queue = queue.Queue()  # store finished request queue
        self.workload_gen_port = (
            workload_gen_port  # the port of workload gen server, assume localhost
        )
        self.workload_gen_ip = workload_gen_ip
        # RPC setup - this is a bit dirty, put this in a reusable class
        self.channel = grpc.insecure_channel(
            f"{self.workload_gen_ip}:{self.workload_gen_port}"
        )
        self.workload_gen_stub = datagen_pb2_grpc.DataServerStub(self.channel)
        return None

    def InferRequest(self, request, context):
        """
        Get Inference Request
        """
        recieved_request = json.loads(request.response)
        self.input_request_queue.put(recieved_request)
        return_request = datagen_pb2.JsonResponse()
        return_request.response = json.dumps({"recieved": True})
        return return_request

    def FinishedRequest(self, request, context):
        """
        Finished Request respond to workload gen by real worker
        """
        return_request = datagen_pb2.JsonResponse()
        return_request.response = json.dumps(return_request)
        self.finished_request_queue.put(return_request)
        self.workload_gen_stub.PostResponse(return_request)  # forward this to scheduler
        return_request = datagen_pb2.JsonResponse()
        return_request.response = json.dumps({"return": True})
        return return_request

    def get_all_requests(self):
        """
        Return all response back to the data generator
        """

        all_requests = []
        while True:
            try:
                task = self.input_request_queue.get(block=False)
                all_requests.append(task)
            except queue.Empty:
                break
        return all_requests

    def get_all_finished_requests(self):
        """
        Return all finished requests to the data generator
        """

        all_requests = []
        while True:
            try:
                task = self.finished_request_queue.get(block=False)
                all_requests.append(task)
            except queue.Empty:
                break
        return all_requests


def start_server(scheduler_server: SchedulerServer, reciever_port: str):
    """
    Start Server
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    scheduler_pb2_grpc.add_SchedulerServerServicer_to_server(scheduler_server, server)
    server.add_insecure_port(f"[::]:{reciever_port}")
    server.start()
    return server
