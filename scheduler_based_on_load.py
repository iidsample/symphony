import os
import sys
import json
import time
import queue
import grpc
import argparse
from typing import Tuple, Dict, Any
from collections import defaultdict
from concurrent import futures

import llm_frontend


sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "rpc"))
sys.path.append(os.path.join(os.path.dirname(__file__), "rpc", "grpc_stubs"))
from rpc.grpc_stubs import datagen_pb2_grpc, datagen_pb2, node_scheduler_pb2_grpc, scheduler_pb2_grpc


class LLMScheduler(object):
    def __init__(
        self, scheduler_port: str, workload_gen_ip: str, workload_gen_port: str
    ):
        """
        Perform scheduling
        """
        self.server, self.scheduler_class = launch_server(scheduler_port)
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
        
        # Node manager gRPC configuration - maps server_id to node manager info
        self.node_managers = {
            0: {"ip": "localhost", "port": 9000},
            1: {"ip": "localhost", "port": 9001},
            2: {"ip": "localhost", "port": 9002},
            3: {"ip": "localhost", "port": 9003},
            4: {"ip": "localhost", "port": 9004},
            5: {"ip": "localhost", "port": 9005},
            6: {"ip": "localhost", "port": 9006},
            7: {"ip": "localhost", "port": 9007},
        }
        
        # Create gRPC channels and stubs for each node manager
        self.node_manager_channels = {}
        self.node_manager_stubs = {}
        for server_id, node_info in self.node_managers.items():
            channel = grpc.insecure_channel(f"{node_info['ip']}:{node_info['port']}")
            self.node_manager_channels[server_id] = channel
            self.node_manager_stubs[server_id] = node_scheduler_pb2_grpc.RecieveDecodeRequestStub(channel)
        
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
        self.per_server_throughput = 1738  # num tokens/sec
        
        # Track when we last polled each node manager
        self.last_status_poll = {i: 0 for i in range(8)}
        self.status_poll_interval = 1.0  # Poll every 1 second
        
        # Track advisory requests (predictive scheduling)
        self.advisory_requests = {}  # session_id -> {eta, target_server, timestamp}

    def get_requests_and_schedule(self):
        """
        Poll
        """
        try:
            while True:
                print("Wait Queue {}".format(self.queued_requests))
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
                    
                    # Check if this is an advisory request
                    if req.get("request_type") == "advisory":
                        print(f"Advisory request received for session {session_id}")
                        self._handle_advisory_request(req)
                        continue  # Don't schedule advisory as actual request
                    
                    # Normal request handling
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
                            # schedule it for run - send to node manager
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
                            
                            # Send request to node manager via gRPC
                            self._send_request_to_node_manager(server_id_to_schedule, session_id, req)

                        else:
                            # can't schedule put in wait queue
                            self.queued_requests[server_id_to_schedule].append(
                                session_id
                            )

                    else:
                        # first time seeing this session id
                        server_found = False
                        for server_id in self.server_free_memory_available:
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
                                
                                # Send request to node manager via gRPC
                                self._send_request_to_node_manager(server_id, session_id, req)
                                
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
                
                # Poll node managers for status updates
                current_time = time.time()
                for server_id in range(8):
                    if current_time - self.last_status_poll[server_id] >= self.status_poll_interval:
                        self.last_status_poll[server_id] = current_time
                        self._poll_node_manager_status(server_id)
                
                time.sleep(1)  # sleep x amount time
                end_scheduling_time = time.time()
                sec_to_schedule_to_all_receieved_req = (
                    end_scheduling_time - start_scheduling_time
                )
                start_scheduling_time = time.time()
                # import ipdb

                # ipdb.set_trace()
                session_id_to_respond_back = list()
                # the loop here equally divides the throughput
                # removes finised request and counts them
                # print(
                # "Tokens remaining to process {}".format(
                # self.server_req_dict_tokens_remaining
                # )
                # )
                # print(
                # "Tokens Remaining before {}".format(
                # self.server_req_dict_tokens_remaining
                # )
                # )
                # print("Server Status before {}".format(self.server_req_dict))

                for server_id in self.server_req_dict_tokens_remaining:
                    all_tokens = self.server_req_dict_tokens_remaining[server_id]
                    if len(all_tokens) == 0:
                        continue
                    num_tokens_processed_per_request = (
                        self.per_server_throughput / len(all_tokens)
                    ) * sec_to_schedule_to_all_receieved_req  # multiply
                    new_token_list = []
                    request_to_remove = []
                    for idx, tokens_remaining in enumerate(all_tokens):
                        new_remaining_tokens = (
                            tokens_remaining - num_tokens_processed_per_request
                        )
                        if new_remaining_tokens < 0:
                            # request finished
                            # marked of request to respond back
                            # print("Removing index {}".format(idx))
                            # import ipdb

                            # ipdb.set_trace()
                            # print(
                            # "Removing {}".format(
                            # self.server_req_dict[server_id][idx]
                            # )
                            # )
                            # import ipdb

                            # ipdb.set_trace()
                            session_id_to_respond_back.append(
                                self.server_req_dict[server_id][idx]
                            )  # finished
                            # free memory for finished token
                            num_tokens_freed_memory = self.active_session_lengths[idx]
                            self.server_free_memory_available[server_id] += (
                                num_tokens_freed_memory
                            ) * (1.1 / 1000)

                            # remove the session id from running list
                            request_to_remove.append(idx)

                        else:
                            new_token_list.append(new_remaining_tokens)
                    print("Request to remove {}".format(request_to_remove))
                    for req_remove in sorted(request_to_remove, reverse=True):
                        # import ipdb

                        # ipdb.set_trace()
                        del self.server_req_dict[server_id][req_remove]
                    # update remaining tokens
                    self.server_req_dict_tokens_remaining[server_id] = new_token_list
                # print(
                # "Tokens Remaining after {}".format(
                # self.server_req_dict_tokens_remaining
                # )
                # )
                # print("Server Status after {}".format(self.server_req_dict))

                json.dump(self.queued_requests, self.out_time_file)
                self.out_time_file.write("\n")
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
                            
                            # Send queued request to node manager via gRPC
                            req_info = self.active_request_dict[sreq]
                            self._send_request_to_node_manager(server_id, sreq, req_info)
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
            # Close all node manager channels
            for channel in self.node_manager_channels.values():
                channel.close()
            self.server.stop(0)

    def _send_request_to_node_manager(self, server_id: int, session_id: int, req: Dict[str, Any]):
        """
        Send inference request to the node manager via gRPC
        """
        try:
            stub = self.node_manager_stubs[server_id]
            
            # Prepare the request payload
            request_dict = {
                "session_id": session_id,
                "prompt": req.get("prompt", ""),
                "response": req.get("response", ""),
                "is_last": req.get("is_last", False)
            }
            
            # Create gRPC request
            grpc_request = datagen_pb2.JsonResponse()
            grpc_request.response = json.dumps(request_dict)
            
            # Send via gRPC AcceptRequest
            response = stub.AcceptRequest(grpc_request)
            response_data = json.loads(response.response)
            
            if response_data.get("recieved"):
                print(f"Request {session_id} sent to node manager {server_id}")
            else:
                print(f"Node manager {server_id} did not acknowledge request {session_id}")
                
        except grpc.RpcError as e:
            print(f"gRPC error sending request {session_id} to node manager {server_id}: {e.code()} - {e.details()}")
        except Exception as e:
            print(f"Error sending request {session_id} to node manager {server_id}: {str(e)}")

    def _poll_node_manager_status(self, server_id: int):
        """
        Poll node manager for current status and update local metrics
        """
        try:
            stub = self.node_manager_stubs[server_id]
            
            # Create empty request for GetStatus
            empty_request = datagen_pb2.JsonResponse()
            empty_request.response = json.dumps({})
            
            # Call GetStatus RPC
            response = stub.GetStatus(empty_request)
            status_data = json.loads(response.response)
            
            # Update server_req_dict with running requests
            if "running_requests" in status_data:
                running = status_data["running_requests"]
                if running is not None:
                    self.server_req_dict[server_id] = running
            
            # Update free memory
            if "free_memory_gb" in status_data:
                free_mem = status_data["free_memory_gb"]
                if free_mem is not None:
                    self.server_free_memory_available[server_id] = free_mem
            
            # Update tokens remaining (if provided)
            if "tokens_remaining" in status_data:
                tokens_dict = status_data["tokens_remaining"]
                if tokens_dict:
                    tokens_list = []
                    for session_id in self.server_req_dict[server_id]:
                        tokens_list.append(tokens_dict.get(str(session_id), 0))
                    self.server_req_dict_tokens_remaining[server_id] = tokens_list
                    
        except grpc.RpcError as e:
            print(f"gRPC error polling node manager {server_id}: {e.code()}")
        except Exception as e:
            print(f"Error polling node manager {server_id}: {str(e)}")

    def _handle_advisory_request(self, req: Dict[str, Any]):
        """
        Handle advisory request indicating user has started typing
        Forwards to appropriate node manager for predictive scheduling
        """
        try:
            session_id = req["session_id"]
            eta = req.get("estimated_time_to_request", 0)
            
            # Determine target server for this advisory
            if session_id in self.kv_cache_map:
                # Session has affinity to a specific server
                target_server = self.kv_cache_map[session_id]
                print(f"Advisory for session {session_id} -> server {target_server} (affinity), ETA: {eta:.2f}s")
            else:
                # New session, use load balancing to select server
                # Use small memory requirement for advisory (doesn't allocate yet)
                target_server = self._select_server_load_balanced(memory_required=0.1)
                if target_server == -1:
                    # No server available, pick one with max free memory
                    target_server = max(
                        self.server_free_memory_available,
                        key=self.server_free_memory_available.get
                    )
                print(f"Advisory for session {session_id} -> server {target_server} (load balanced), ETA: {eta:.2f}s")
            
            # Track advisory for monitoring
            self.advisory_requests[session_id] = {
                "eta": eta,
                "target_server": target_server,
                "timestamp": time.time()
            }
            
            # Forward advisory to node manager
            self._send_advisory_to_node_manager(target_server, session_id, req)
            
        except Exception as e:
            print(f"Error handling advisory request: {str(e)}")

    def _send_advisory_to_node_manager(self, server_id: int, session_id: int, req: Dict[str, Any]):
        """
        Send advisory request to node manager via gRPC
        """
        try:
            stub = self.node_manager_stubs[server_id]
            
            # Create advisory payload
            advisory_dict = {
                "session_id": session_id,
                "request_type": "advisory",
                "estimated_time_to_request": req.get("estimated_time_to_request", 0),
                "prompt": "",
                "response": "",
                "is_last": False
            }
            
            # Create gRPC request
            grpc_request = datagen_pb2.JsonResponse()
            grpc_request.response = json.dumps(advisory_dict)
            
            # Send via gRPC AcceptRequest (node manager will recognize advisory type)
            response = stub.AcceptRequest(grpc_request)
            response_data = json.loads(response.response)
            
            if response_data.get("recieved"):
                print(f"Advisory forwarded to node manager {server_id} for session {session_id}")
            
        except grpc.RpcError as e:
            print(f"gRPC error sending advisory to node manager {server_id}: {e.code()}")
        except Exception as e:
            print(f"Error sending advisory to node manager {server_id}: {str(e)}")

    def _select_server_load_balanced(self, memory_required: float) -> int:
        """
        Select best server using load balancing algorithm
        Considers both current load (running + queued requests) and free memory
        
        Returns server_id or -1 if no server available
        """
        best_server = -1
        best_score = float('inf')
        
        for server_id in range(8):
            # Skip if not enough memory
            if self.server_free_memory_available[server_id] < memory_required:
                continue
            
            # Calculate load score (lower is better)
            running_count = len(self.server_req_dict[server_id])
            queued_count = len(self.queued_requests[server_id])
            total_requests = running_count + queued_count
            
            # Normalize memory (0-1, higher free memory = lower score)
            memory_score = 1.0 - (self.server_free_memory_available[server_id] / 66.85)
            
            # Combined score: weighted sum of request count and memory
            # Weight request count more heavily (0.7) than memory (0.3)
            score = (0.7 * total_requests) + (0.3 * memory_score * 100)
            
            if score < best_score:
                best_score = score
                best_server = server_id
        
        return best_server



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
    scheduler_port: str,
) -> Tuple[grpc.Server, llm_frontend.SchedulerServer]:
    """
    Launch Server
    """
    scheduler_server = llm_frontend.SchedulerServer()
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
