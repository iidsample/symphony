# simulation for calculating the queue on each server
import os
import sys
import json
import time
import queue
import random
import grpc
import argparse
import requests
import threading
from typing import Tuple, Dict, Any
from collections import defaultdict
from concurrent import futures

import llm_frontend


sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "rpc"))
sys.path.append(os.path.join(os.path.dirname(__file__), "rpc", "grpc_stubs"))
from rpc.grpc_stubs import datagen_pb2_grpc, datagen_pb2


class LLMScheduler(object):
    def __init__(
        self, scheduler_port: str, workload_gen_ip: str, workload_gen_port: str
    ):
        """
        Perform scheduling
        """
        self.server, self.scheduler_class = launch_server(
            scheduler_port, workload_gen_ip, workload_gen_port
        )
        self.workload_gen_port = (
            workload_gen_port  # the port of workload gen server, assume localhost
        )
        self.workload_gen_ip = workload_gen_ip
        # RPC setup - this is a bit dirty, put this in a reusable class
        self.channel = grpc.insecure_channel(
            f"{self.workload_gen_ip}:{self.workload_gen_port}"
        )
        self.workload_gen_stub = datagen_pb2_grpc.DataServerStub(self.channel)
        self.total_number_of_reqs = 0

        # bookkeeping dicts
        self.active_session_lengths = defaultdict(
            int
        )  # active session ids -> and prompt + gen

        self.active_request_dict = dict()  # map of all active requests -> recieved dict
        self.out_time_file = open(f"./queue_length_per_server_{time.time()}.json", "w")
        self.out_current_requests_file = open(
            f"./running_requests_length_per_server_{time.time()}.json", "w"
        )
        self.server_req_dict = {
            i: [] for i in range(8)
        }  # server id -> list of requests running
        self.server_req_dict_tokens_remaining = {
            i: [] for i in range(8)
        }  # server id -> tokens remaining
        self.server_free_memory_available = {
            i: 66.85 for i in range(8)
        }  # memory remaining after loading the model

        self.queued_requests = {i: [] for i in range(8)}

        self.kv_cache_map = {}  # session id -> gpu map
        
        # vLLM server configuration
        self.vllm_host = "localhost"
        self.vllm_port = 8000
        self.vllm_base_url = f"http://{self.vllm_host}:{self.vllm_port}/v1"
        
        # Track active vLLM requests
        self.active_vllm_requests = {}  # session_id -> {thread, start_time, server_id}
        self.completed_vllm_requests = []  # List of completed session_ids
        self.vllm_lock = threading.Lock()

    def get_requests_and_schedule(self):
        """
        Poll
        """
        try:
            while True:
                print("Wait Queue {}".format(self.queued_requests))
                print("Run Request {}".format(self.server_req_dict))
                recvd_requests = self.scheduler_class.get_all_requests()
                # if len(recvd_requests) > 0:
                # print("All rcvd request {}".format(recvd_requests))
                start_scheduling_time = time.time()
                running_requests = [
                    len(self.server_req_dict[k]) for k in self.server_req_dict
                ]
                queued_requests = [
                    len(self.queued_requests[k]) for k in self.server_req_dict
                ]
                print("Running Request {}".format(sum(running_requests)))
                print("Waiting Requests {}".format(sum(queued_requests)))
                print(
                    "Total Request in system {}".format(
                        sum(running_requests) + sum(queued_requests)
                    )
                )

                for req in recvd_requests:
                    # print(req)
                    session_id = req["session_id"]
                    # TODO: add additional logic for performing memory managment and
                    self.total_number_of_reqs += 1
                    self.active_session_lengths[session_id] += len(req["prompt"]) + len(
                        req["response"]
                    )
                    self.active_request_dict[session_id] = (
                        req  # maintaining request in the dict
                    )

                    # schedule request on a machine
                    memory_required = (self.active_session_lengths[session_id]) * (
                        1.1 / 1000
                    )  # Gb
                    # find if we have a preference for this request
                    if session_id in self.kv_cache_map:
                        # already seen the request
                        server_id_to_schedule = self.kv_cache_map[session_id]
                        # check if memory is free
                        free_memory_available = self.server_free_memory_available[
                            server_id_to_schedule
                        ]
                        if free_memory_available > memory_required:
                            # schedule it for run - make actual vLLM call
                            self.server_req_dict[server_id_to_schedule].append(
                                session_id
                            )
                            self.server_req_dict_tokens_remaining[
                                server_id_to_schedule
                            ].append(self.active_session_lengths[session_id])
                            # added tokens to process
                            self.server_free_memory_available[
                                server_id_to_schedule
                            ] -= memory_required
                            
                            # Make actual vLLM API call in background thread
                            self._make_vllm_request_async(session_id, req, server_id_to_schedule)

                        else:
                            # can't schedule put in wait queue
                            self.queued_requests[server_id_to_schedule].append(
                                session_id
                            )

                    else:
                        # first time seeing this session id
                        server_found = False
                        for server_id in self.server_free_memory_available:
                            # import ipdb

                            # ipdb.set_trace()
                            server_id = random.sample(
                                list(self.server_free_memory_available.keys()), 1
                            )[0]
                            free_memory_available = self.server_free_memory_available[
                                server_id
                            ]

                            if free_memory_available > memory_required:
                                self.server_req_dict[server_id].append(session_id)
                                self.server_req_dict_tokens_remaining[server_id].append(
                                    self.active_session_lengths[session_id]
                                )

                                self.server_free_memory_available[
                                    server_id
                                ] -= memory_required

                                self.kv_cache_map[session_id] = server_id
                                
                                # Make actual vLLM API call in background thread
                                self._make_vllm_request_async(session_id, req, server_id)
                                
                                server_found = True
                                break
                        if not server_found:
                            # need to schedule on one with max free memory
                            # TODO: Experiment with this
                            server_id_max_free_memory = max(
                                self.server_free_memory_available,
                                key=self.server_free_memory_available.get,
                            )
                            self.queued_requests[server_id_max_free_memory].append(
                                session_id
                            )
                            self.kv_cache_map[session_id] = server_id_max_free_memory
                # At the end of this
                # either a req is running on a gpu updating k,v cache map
                # req dict update and the tokens needed to process
                time.sleep(0.1)  # Short sleep to check for completed requests
                
                # Check for completed vLLM requests
                session_id_to_respond_back = []
                with self.vllm_lock:
                    if self.completed_vllm_requests:
                        session_id_to_respond_back = self.completed_vllm_requests.copy()
                        self.completed_vllm_requests.clear()
                
                # Process completed requests
                for session_id in session_id_to_respond_back:
                    if session_id in self.active_vllm_requests:
                        server_id = self.active_vllm_requests[session_id]['server_id']
                        
                        # Free memory
                        num_tokens_freed_memory = self.active_session_lengths[session_id]
                        self.server_free_memory_available[server_id] += (
                            num_tokens_freed_memory * (1.1 / 1000)
                        )
                        
                        # Remove from active requests
                        if session_id in self.server_req_dict[server_id]:
                            self.server_req_dict[server_id].remove(session_id)
                        if session_id in self.active_vllm_requests:
                            del self.active_vllm_requests[session_id]
                        
                        # Remove from tokens remaining tracking
                        # Find and remove from server_req_dict_tokens_remaining
                        try:
                            idx = [i for i, sid in enumerate(self.server_req_dict[server_id]) 
                                   if sid == session_id]
                            if idx:
                                del self.server_req_dict_tokens_remaining[server_id][idx[0]]
                        except (ValueError, IndexError):
                            pass
                # print(
                # "Tokens Remaining after {}".format(
                # self.server_req_dict_tokens_remaining
                # )
                # )
                # print("Server Status after {}".format(self.server_req_dict))

                json.dump(self.queued_requests, self.out_time_file)
                self.out_time_file.write("\n")
                json.dump(self.server_req_dict, self.out_current_requests_file)
                self.out_current_requests_file.write("\n")
                # Schedule from wait queues to actual servers after free memory

                for server_id in self.queued_requests:
                    queued_requests_per_server = self.queued_requests[server_id]
                    scheduled_requests = []
                    for sreq in queued_requests_per_server:
                        memory_required = (
                            (self.active_session_lengths[sreq]) * 1.1 / 1000
                        )
                        if (
                            self.server_free_memory_available[server_id]
                            > memory_required
                        ):
                            self.server_free_memory_available[
                                server_id
                            ] -= memory_required
                            scheduled_requests.append(sreq)
                            self.server_req_dict[server_id].append(sreq)
                            self.server_req_dict_tokens_remaining[server_id].append(
                                self.active_session_lengths[sreq]
                            )
                            
                            # Make actual vLLM API call for the queued request
                            req_info = self.active_request_dict[sreq]
                            self._make_vllm_request_async(sreq, req_info, server_id)
                        else:
                            break  # server not having memory, # this maintains fifo order
                    # remove scheduled requests from wait queue
                    for sched_req in scheduled_requests:
                        self.queued_requests[server_id].remove(sched_req)

                # time to respond back finished requests
                # print(
                # "-------Session ids sent {}----------------------------".format(
                # session_id_to_respond_back
                # )
                # )
                for resp_back in session_id_to_respond_back:
                    req_info = self.active_request_dict[resp_back]
                    is_last = self.active_request_dict[resp_back]["is_last"]
                    del self.active_request_dict[resp_back]
                    # only if it is not last
                    return_request = datagen_pb2.JsonResponse()
                    return_request.response = json.dumps({"session_id": resp_back})
                    self.workload_gen_stub.PostResponse(return_request)
                    if is_last:
                        del self.active_session_lengths[resp_back]
                        print(f"{resp_back} Last Request")

        except KeyboardInterrupt:
            print("Exiting Server")
            self.channel.close()
            self.server.stop(0)

    def _make_vllm_request_async(self, session_id: int, req: Dict[str, Any], server_id: int):
        """
        Make async vLLM API request in a background thread
        """
        thread = threading.Thread(
            target=self._call_vllm_api,
            args=(session_id, req, server_id),
            daemon=True
        )
        
        with self.vllm_lock:
            self.active_vllm_requests[session_id] = {
                'thread': thread,
                'start_time': time.time(),
                'server_id': server_id
            }
        
        thread.start()
    
    def _call_vllm_api(self, session_id: int, req: Dict[str, Any], server_id: int):
        """
        Make actual HTTP call to vLLM server
        """
        try:
            prompt = req.get('prompt', '')
            
            # Prepare the request for vLLM OpenAI-compatible API
            payload = {
                "model": "default",  # vLLM will use the loaded model
                "prompt": prompt,
                "max_tokens": 2048,
                "temperature": 0.7,
                "stream": False
            }
            
            # Make the request to vLLM
            response = requests.post(
                f"{self.vllm_base_url}/completions",
                json=payload,
                timeout=300  # 5 minute timeout
            )
            
            if response.status_code == 200:
                print(f"vLLM request completed for session {session_id}")
                # Mark as completed
                with self.vllm_lock:
                    self.completed_vllm_requests.append(session_id)
            else:
                print(f"vLLM request failed for session {session_id}: {response.status_code}")
                # Still mark as completed to free resources
                with self.vllm_lock:
                    self.completed_vllm_requests.append(session_id)
                    
        except requests.exceptions.Timeout:
            print(f"vLLM request timeout for session {session_id}")
            with self.vllm_lock:
                self.completed_vllm_requests.append(session_id)
        except Exception as e:
            print(f"vLLM request error for session {session_id}: {str(e)}")
            with self.vllm_lock:
                self.completed_vllm_requests.append(session_id)


def parse_args(parser):
    """
    Parse Arguments
    """
    parser.add_argument(
        "--scheduler-port", type=str, default=50051, help="Scheduler Port"
    )
    parser.add_argument(
        "--workload-gen-ip",
        type=str,
        default="localhost",
        help=" Workload Generator IP",
    )
    parser.add_argument(
        "--workload-gen-port", type=str, default=50050, help="Workload Gen Port"
    )
    args = parser.parse_args()
    return args


def launch_server(
    scheduler_port: str, workload_gen_ip: str, workload_gen_port: str
) -> Tuple[grpc.Server, llm_frontend.SchedulerServer]:
    """
    Launch Server
    """
    scheduler_server = llm_frontend.SchedulerServer(workload_gen_ip, workload_gen_port)
    server = llm_frontend.start_server(scheduler_server, scheduler_port)
    print("Server Started")
    return server, scheduler_server


if __name__ == "__main__":
    args = parse_args(
        argparse.ArgumentParser(description="Arguments for Top-Level Scheduler")
    )
    workload_gen = LLMScheduler(
        args.scheduler_port, args.workload_gen_ip, args.workload_gen_port
    )
    workload_gen.get_requests_and_schedule()
