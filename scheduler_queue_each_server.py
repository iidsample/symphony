# simulation for calculating the queue on each server
import os
import sys
import json
import time
import queue
import random
import grpc
import argparse
from typing import Tuple
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
        self.per_server_throughput = 35  # num tokens/sec

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
                            # schedule it for run
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
                            num_tokens_freed_memory = self.active_session_lengths[
                                self.server_req_dict[server_id][idx]
                            ]
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
