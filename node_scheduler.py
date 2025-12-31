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

    def get_request_and_run(self):
        """
        Poll
        """
        try:
            while True:
                # get all requests in the process
                # print("Wait Queue {}".format(self.waiting_request))
                recv_requests = self.node_scheduler_class.get_all_requests()
                print("Received request {}".format(recv_requests))
                for req in recv_requests:
                    req_prompt = recv_requests["prompt"]
                    gen_length = recv_requests["gen_length"]
                    request_id = recv_requests["req_id"]
                    sampling_params = SamplingParams(
                        temperature=0.0, max_tokens=gen_length, min_tokens=gen_length
                    )
                    self.engine.add_request(str(request_id), prompt, sampling_params)
                # add all scheduled request
                request_output = self.engine.step()
                for req in request_output:
                    if req.finished:
                        temp_dict = {}
                        temp_dict["req_id"] = req.request_id
                        temp_dict["finish_time"] = req.metrics.finished_time
                        temp_dict["arrival_time"] = req.metrics.arrival_time
                        temp_dict["last_token_time"] = req.metrics.last_token_time
                        temp_dict["first_scheduled_time"] = (
                            req.metrics.first_scheduled_time
                        )
                        temp_dict["first_token_time"] = req.metrics.first_token_time
                        temp_dict["time_in_queue"] = req.metrics.time_in_queue
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
    args = parser.parse_args()
    node_scheduler = NodeScheduler(
        args,
        args.node_scheduler_port,
        args.global_scheduler_ip,
        args.global_scheduler_port,
    )

    node_scheduler.get_request_and_run()
