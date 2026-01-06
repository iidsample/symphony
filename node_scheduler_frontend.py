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
from rpc.grpc_stubs import datagen_pb2
from rpc.grpc_stubs import scheduler_pb2_grpc
from rpc.grpc_stubs import node_scheduler_pb2_grpc


class NodeSchedulerFrontend(node_scheduler_pb2_grpc.RecieveDecodeRequestServicer):
    """
    Recieve Requests to run
    """

    def __init__(self, global_scheduler_ip, global_scheduler_port):
        self.input_request_queue = queue.Queue()
        self.request_id = 0
        self.request_id_to_session_id_mapping = dict()
        self.global_scheduler_port = global_scheduler_port
        self.global_scheduler_ip = global_scheduler_ip
        self.channel = grpc.insecure_channel(
            f"{self.global_scheduler_ip}:{self.global_scheduler_port}"
        )
        self.global_scheduler_stub = scheduler_pb2_grpc.SchedulerServerStub(
            self.channel
        )
        return None

    def AcceptRequest(self, request, context):
        """
        Accept Request which was send by Global Scheduler
        Can be either actual inference request or advisory (predictive)
        """
        recieved_request = json.loads(request.response)
        session_id_rcv = recieved_request["session_id"]
        
        # Check if this is an advisory request
        if recieved_request.get("request_type") == "advisory":
            # Handle advisory - user has started typing
            eta = recieved_request.get("estimated_time_to_request", 0)
            print(f"Node received advisory for session {session_id_rcv}, ETA: {eta:.2f}s")
            # TODO: Implement predictive actions:
            # - Warm up KV cache if evicted
            # - Reserve GPU memory
            # - Pre-schedule in queue
            return_request = datagen_pb2.JsonResponse()
            return_request.response = json.dumps({"recieved": True, "type": "advisory"})
            return return_request
        
        # Normal inference request handling
        recieved_request["req_id"] = self.request_id
        self.request_id_to_session_id_mapping[self.request_id] = session_id_rcv
        self.request_id += 1
        self.input_request_queue.put(recieved_request)
        return_request = datagen_pb2.JsonResponse()
        return_request.response = json.dumps({"recieved": True})
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

    def finished_request(self, return_dict):
        """
        Return requests which have been finished
        """
        req_id = return_dict["req_id"]
        session_id = self.request_id_to_session_id_mapping[req_id]
        return_dict["session_id"] = session_id
        return_request = datagen_pb2.JsonResponse()
        return_request.response = json.dumps(return_dict)
        self.global_scheduler_stub.FinishedRequest(return_request)
        return None

    def GetStatus(self, request, context):
        """
        Return current status of this node scheduler
        Called by global scheduler to get running requests and resource info
        """
        # Get list of running session IDs
        running_sessions = list(self.request_id_to_session_id_mapping.values())
        
        # For now, return basic status - can be enhanced with actual vLLM metrics
        status_dict = {
            "running_requests": running_sessions,
            "tokens_remaining": {},  # Can be populated from vLLM engine metrics
            "free_memory_gb": 66.85,  # Placeholder - should query actual GPU memory
            "completed_requests": []  # List of recently completed requests
        }
        
        return_response = datagen_pb2.JsonResponse()
        return_response.response = json.dumps(status_dict)
        return return_response


def start_server(scheduler_server: NodeSchedulerFrontend, reciever_port: str):
    """
    Start Server
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    node_scheduler_pb2_grpc.add_RecieveDecodeRequestServicer_to_server(
        scheduler_server, server
    )
    server.add_insecure_port(f"[::]:{reciever_port}")
    return server
