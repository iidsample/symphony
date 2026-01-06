# this is basically gpu scheduler runs on vllm, launch this with cuda visible devices
import os
import sys
import json
import time
import queue
import grpc
import argparse
from typing import Tuple
from collections import defaultdict
from concurrent import futures

import node_scheduler_frontend


sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "rpc"))
sys.path.append(os.path.join(os.path.dirname(__file__), "rpc", "grpc_stubs"))
from rpc.grpc_stubs import datagen_pb2_grpc, datagen_pb2

from typing import List, Tuple

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.utils import FlexibleArgumentParser


class NodeScheduler(object):
    def __init__(
        self,
        args,
        node_scheduler_port: str,
        global_scheduler_ip: str,
        global_scheduler_port: str,
        memory_threshold: float = 0.8,  # Free cache when memory usage exceeds 80%
        cache_eviction_layers: int = 2,  # Number of early layers to evict
    ):
        """
        Perform Node Scheduling
        """

        self.server, self.node_scheduler_class = launch_server(
            node_scheduler_port, global_scheduler_ip, global_scheduler_port
        )
        # self.global_scheduler_port = global_scheduler_port
        # self.global_scheduler_ip = global_scheduler_ip
        # self.channel = grpc.insecure_channel(
        # f"{self.global_scheduler_ip}:{self.global_scheduler_port}"
        # )
        self.waiting_request = list()  # this iteration is not using this
        self.engine_args = EngineArgs.from_cli_args(args)
        print(self.engine_args)
        self.engine = LLMEngine.from_engine_args(self.engine_args)
        
        # Memory management thresholds
        self.memory_threshold = memory_threshold
        self.cache_eviction_layers = cache_eviction_layers
        self.last_memory_check = time.time()
        self.memory_check_interval = 1.0  # Check memory every 1 second

    def _check_and_manage_memory(self):
        """
        Cooperative memory management: Check GPU memory usage and instruct vLLM 
        to free early layers of K,V cache when memory exceeds threshold.
        """
        current_time = time.time()
        
        # Only check memory at specified intervals to avoid overhead
        if current_time - self.last_memory_check < self.memory_check_interval:
            return
        
        self.last_memory_check = current_time
        
        try:
            import torch
            if torch.cuda.is_available():
                # Get memory stats for the current GPU
                device = torch.cuda.current_device()
                memory_allocated = torch.cuda.memory_allocated(device)
                memory_reserved = torch.cuda.memory_reserved(device)
                total_memory = torch.cuda.get_device_properties(device).total_memory
                
                # Calculate memory usage ratio
                memory_usage_ratio = memory_reserved / total_memory
                
                if memory_usage_ratio > self.memory_threshold:
                    print(f"[MEMORY] High memory usage detected: {memory_usage_ratio*100:.1f}% "
                          f"(allocated: {memory_allocated/1e9:.2f}GB, "
                          f"reserved: {memory_reserved/1e9:.2f}GB, "
                          f"total: {total_memory/1e9:.2f}GB)")
                    
                    if hasattr(self.engine, 'free_kv_cache_layers'):
                        print(f"[MEMORY] Instructing vLLM to free K,V cache for "
                              f"first {self.cache_eviction_layers} layers")
                        self.engine.free_kv_cache_layers(num_layers=self.cache_eviction_layers)
                    else:
                        print(f"[MEMORY] Warning: vLLM doesn't support layer-specific cache eviction")
                        if hasattr(self.engine, 'clear_cache'):
                            self.engine.clear_cache()
                        
        except ImportError:
            pass
        except Exception as e:
            print(f"[MEMORY] Error checking memory: {e}")

    def get_request_and_run(self):
        """
        Poll for requests and process with vLLM engine
        Handles both advisory (predictive) and actual inference requests
        """
        try:
            while True:
                # Cooperative memory management: Check and free K,V cache if needed
                self._check_and_manage_memory()
                
                # Get all requests from the frontend
                recv_requests = self.node_scheduler_class.get_all_requests()
                
                if len(recv_requests) > 0:
                    print(f"Received {len(recv_requests)} request(s)")
                
                # Process each request
                for req in recv_requests:
                    session_id = req.get("session_id")
                    request_type = req.get("request_type", "inference")
                    
                    # Handle advisory requests
                    if request_type == "advisory":
                        eta = req.get("estimated_time_to_request", 0)
                        print(f"[ADVISORY] Session {session_id} will send request in {eta:.2f}s")
                        
                        # Send advisory to vLLM with special tag
                        # Modified vLLM will recognize this and prepare resources without full inference
                        advisory_id = f"advisory_{session_id}_{int(time.time()*1000)}"
                        
                        # Create sampling params with minimal tokens (advisory doesn't need generation)
                        advisory_sampling_params = SamplingParams(
                            temperature=0.0,
                            max_tokens=1,  # Minimal, just for resource prep
                        )
                        
                        # Add advisory request to vLLM with special ID prefix
                        # Modified vLLM checks if request_id starts with "advisory_" for special handling
                        print(f"[ADVISORY->vLLM] Sending advisory {advisory_id} to vLLM for resource prep")
                        self.engine.add_request(
                            advisory_id,
                            f"[ADVISORY] Session {session_id}, ETA: {eta}s",  # Special prompt format
                            advisory_sampling_params
                        )
                        continue  # Skip normal inference handling
                    
                    # Handle actual inference requests
                    req_prompt = req.get("prompt", "")
                    req_id = req.get("req_id")
                    
                    # Determine generation length
                    # Can be from request or use default
                    gen_length = req.get("gen_length", 256)  # Default 256 tokens
                    
                    # Create sampling parameters
                    sampling_params = SamplingParams(
                        temperature=0.7,
                        max_tokens=gen_length,
                    )
                    
                    # Add request to vLLM engine
                    print(f"[INFERENCE] Adding request {req_id} (session {session_id}) to vLLM")
                    self.engine.add_request(str(req_id), req_prompt, sampling_params)
                
                # Step the vLLM engine to process requests
                request_outputs = self.engine.step()
                
                # Handle completed requests
                for output in request_outputs:
                    if output.finished:
                        # Collect metrics
                        return_dict = {
                            "req_id": int(output.request_id),
                            "finish_time": output.metrics.finished_time,
                            "arrival_time": output.metrics.arrival_time,
                            "last_token_time": output.metrics.last_token_time,
                            "first_scheduled_time": output.metrics.first_scheduled_time,
                            "first_token_time": output.metrics.first_token_time,
                            "time_in_queue": output.metrics.time_in_queue,
                        }
                        
                        print(f"[COMPLETE] Request {output.request_id} finished")
                        
                        # Notify global scheduler
                        self.node_scheduler_class.finished_request(return_dict)

        except KeyboardInterrupt:
            print("Exiting GPU Server")
            # self.channel.close()
            self.server.stop(0)


# def parse_args(parser):
# """
# Parse Arguments
# """
# parser.add_argument(
# "--scheduler-port", type=str, default=50051, help="Scheduler Port"
# )
# parser.add_argument(
# "--workload-gen-ip",
# type=str,
# default="localhost",
# help=" Workload Generator IP",
# )
# parser.add_argument(
# "--workload-gen-port", type=str, default=50050, help="Workload Gen Port"
# )
# args = parser.parse_args()
# return args


def launch_server(
    scheduler_port: str,
    global_scheduler_ip: str,
    global_scheduler_port: str,
) -> Tuple[grpc.Server, node_scheduler_frontend.NodeSchedulerFrontend]:
    """
    Launch Server
    """
    # scheduler_server = llm_frontend.SchedulerServer()
    scheduler_server = node_scheduler_frontend.NodeSchedulerFrontend(
        global_scheduler_ip, global_scheduler_port
    )
    server = node_scheduler_frontend.start_server(scheduler_server, scheduler_port)
    print("Node Scheduler Started")
    return server, scheduler_server


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Using LLM Engine")
    parser = EngineArgs.add_cli_args(parser)
    parser.add_argument(
        "--node-scheduler-port", type=str, default=40051, help="Scheduler Port"
    )
    parser.add_argument(
        "--global-scheduler-ip",
        type=str,
        default="localhost",
        help=" Global Scheduler IP",
    )
    parser.add_argument(
        "--global-scheduler-port", type=str, default=50050, help="Workload Gen Port"
    )
    parser.add_argument(
        "--memory-threshold", 
        type=float, 
        default=0.8, 
        help="Memory usage threshold (0.0-1.0) above which K,V cache is freed"
    )
    parser.add_argument(
        "--cache-eviction-layers", 
        type=int, 
        default=2, 
        help="Number of early layers to evict from K,V cache during memory pressure"
    )
    args = parser.parse_args()
    node_scheduler = NodeScheduler(
        args,
        args.node_scheduler_port,
        args.global_scheduler_ip,
        args.global_scheduler_port,
        memory_threshold=args.memory_threshold,
        cache_eviction_layers=args.cache_eviction_layers,
    )

    node_scheduler.get_request_and_run()
