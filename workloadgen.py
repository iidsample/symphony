import os
import sys
import json
import time
import queue
import grpc
import argparse
from typing import Tuple
from concurrent import futures

import numpy as np
import received_response


sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "rpc"))
sys.path.append(os.path.join(os.path.dirname(__file__), "rpc", "grpc_stubs"))
from rpc.grpc_stubs import scheduler_pb2_grpc, scheduler_pb2, datagen_pb2


class WorkloadGen(object):
    def __init__(
        self,
        reciever_port: str,
        scheduler_ip: str,
        scheduler_port: str,
        num_clients: int,
    ):
        """
        Generate Workload
        """
        self.server, self.workload_recvr = launch_server(reciever_port)
        self.num_clients = num_clients
        self.dataset_path = (
            "/home/saurabh/Work/memory_managment_llm_inference/sg_90k_part1.json"
        )
        with open(self.dataset_path, "r") as fopen:
            self.open_data = json.load(fopen)  # list
        self.active_sessions = dict()
        self.session_ids = []
        self.session_id_counter = 0
        self.request_list_to_send = list()
        self.send_list_id = dict()

        self.future_request_to_send = (
            dict()
        )  # session ids -> next-request to send mapping

        self.future_request_recieved = (
            dict()
        )  # session ids -> next-request response based on dataset

        self.future_request_time_to_send = (
            dict()
        )  # session ids -> next-request time to send mapping
        
        # Separate tracking for read and type times
        self.future_request_read_time = dict()  # session ids -> read time
        self.future_request_type_time = dict()  # session ids -> type time
        self.advisory_sent = dict()  # session ids -> whether advisory was sent
        
        # remove session ids which have sent last request
        self.last_request_dict = dict()  # session ids -> session ids

        # Distribution parameters for read and write speeds
        self.read_speed_mean = 25  # characters per second
        self.read_speed_std = 10
        self.type_speed_mean = 6  # characters per second
        self.type_speed_std = 2
        
        # Per-session speeds (sampled from distributions)
        self.session_read_speeds = dict()  # session_id -> read speed
        self.session_type_speeds = dict()  # session_id -> type speed
        # RPC channel stuff
        self.scheduler_ip = scheduler_ip
        self.scheduler_port = scheduler_port
        self.channel = grpc.insecure_channel(
            f"{self.scheduler_ip}:{self.scheduler_port}"
        )
        self.send_request_stub = scheduler_pb2_grpc.SchedulerServerStub(self.channel)

    def send_data(self):
        """
        Send data
        """
        # if len(self.active_sessions) < self.num_clients
        try:
            while True:
                # Maintaining given number of clients
                new_session_ids_ready_to_send = []
                new_clients_needed = self.num_clients - len(self.active_sessions)
                print("Length of active_sessions {}".format(len(self.active_sessions)))
                print("New clients needed {}".format(new_clients_needed))
                for _ in range(new_clients_needed):
                    new_data = self.open_data.pop(0)["conversations"]
                    if len(new_data) == 1:
                        # hack to avoid dataset with just one line
                        # leads to little bot trouble not maintaining number of clients
                        # but will fix automatically in next round
                        continue
                    self.active_sessions[self.session_id_counter] = new_data
                    # Sample read and type speeds for this session
                    self.session_read_speeds[self.session_id_counter] = max(
                        1.0, np.random.normal(self.read_speed_mean, self.read_speed_std)
                    )
                    self.session_type_speeds[self.session_id_counter] = max(
                        1.0, np.random.normal(self.type_speed_mean, self.type_speed_std)
                    )
                    new_session_ids_ready_to_send.append((self.session_id_counter, 1))
                    self.session_id_counter += 1
                    # all new clients are ready to send
                returned_response = self.workload_recvr.poll_response()
                print("Returned Response {}".format(returned_response))
                for response in returned_response:
                    sess_id_reponse = response["session_id"]
                    if sess_id_reponse in self.last_request_dict:
                        del self.active_sessions[sess_id_reponse]
                        del self.last_request_dict[sess_id_reponse]
                        # since last request the response recieved will not be updated
                        del self.future_request_recieved[sess_id_reponse]
                        # Clean up session speeds
                        del self.session_read_speeds[sess_id_reponse]
                        del self.session_type_speeds[sess_id_reponse]
                    else:
                        new_session_ids_ready_to_send.append(
                            (response["session_id"], 2)
                        )
                # We now have all possible request to send
                self.update_time_to_next_req(new_session_ids_ready_to_send)
                
                # Check if any advisory requests need to be sent
                # Advisory is sent after read time but before type time
                current_time = time.time()
                for sids in list(self.future_request_read_time.keys()):
                    if sids not in self.advisory_sent and self.future_request_read_time[sids] <= 0:
                        # Read time complete, send advisory request
                        self._send_advisory_request(sids)
                        self.advisory_sent[sids] = True
                
                # find min time client for actual request
                min_time_client = min(
                    self.future_request_time_to_send,
                    key=self.future_request_time_to_send.get,
                )
                min_time = self.future_request_time_to_send[min_time_client]
                time.sleep(min_time)
                data_to_send = datagen_pb2.JsonResponse()
                if min_time_client in self.last_request_dict:
                    data_to_send.response = json.dumps(
                        {
                            "session_id": min_time_client,
                            "prompt": self.future_request_to_send[min_time_client],
                            "response": self.future_request_recieved[min_time_client],
                            "is_last": True,
                        }
                    )
                else:
                    data_to_send.response = json.dumps(
                        {
                            "session_id": min_time_client,
                            "prompt": self.future_request_to_send[min_time_client],
                            "response": self.future_request_recieved[min_time_client],
                            "is_last": False,
                        }
                    )
                response = self.send_request_stub.InferRequest(data_to_send)
                # subtract sleep time from all elements
                self.subtract_time_dict(min_time)
                # remove the sent element
                del self.future_request_to_send[min_time_client]
                del self.future_request_time_to_send[min_time_client]
                if min_time_client in self.future_request_read_time:
                    del self.future_request_read_time[min_time_client]
                if min_time_client in self.future_request_type_time:
                    del self.future_request_type_time[min_time_client]
                if min_time_client in self.advisory_sent:
                    del self.advisory_sent[min_time_client]

        except KeyboardInterrupt:
            print("Exiting Server")
            self.server.stop(0)

    def _send_advisory_request(self, session_id):
        """
        Send advisory request to scheduler indicating user has started typing
        """
        try:
            type_time_remaining = self.future_request_type_time.get(session_id, 0)
            advisory_data = datagen_pb2.JsonResponse()
            advisory_data.response = json.dumps({
                "session_id": session_id,
                "request_type": "advisory",
                "estimated_time_to_request": type_time_remaining,
                "prompt": "",  # No prompt yet, user is still typing
                "response": "",
                "is_last": False
            })
            response = self.send_request_stub.InferRequest(advisory_data)
            print(f"Advisory request sent for session {session_id}, ETA: {type_time_remaining:.2f}s")
        except Exception as e:
            print(f"Error sending advisory request for session {session_id}: {str(e)}")

    def subtract_time_dict(self, min_time):
        """
        Modify dictionary time
        """
        for key in self.future_request_time_to_send:
            self.future_request_time_to_send[key] -= min_time
        
        # Also subtract from read times for advisory tracking
        for key in list(self.future_request_read_time.keys()):
            self.future_request_read_time[key] -= min_time
            if self.future_request_read_time[key] <= 0:
                # Read time complete, can be removed after advisory sent
                if key in self.advisory_sent:
                    del self.future_request_read_time[key]

    def update_time_to_next_req(self, new_session_ids_ready_to_send):
        """
        Decide time to send for next request
        """
        for sids, req_type in new_session_ids_ready_to_send:
            # there can be cases where dataset is not structured like this and not everything is even
            if req_type == 1:
                # first request
                req_sent_to_infer = self.active_sessions[sids].pop(0)["value"]
                response_rcvd_back = self.active_sessions[sids].pop(0)["value"]
                time_to_type = len(req_sent_to_infer) / self.session_type_speeds[sids]
                time_to_read_response = 0  # nothing to read first request sent

            if req_type == 2:
                # second request
                req_sent_to_infer = self.active_sessions[sids].pop(0)["value"]
                response_rcvd_back = self.active_sessions[sids].pop(0)["value"]
                time_to_type = len(req_sent_to_infer) / self.session_type_speeds[sids]
                time_to_read_response = (
                    len(self.future_request_recieved[sids]) / self.session_read_speeds[sids]
                )
            if (
                len(self.active_sessions[sids]) == 0
                or len(self.active_sessions[sids]) == 1
            ):
                # 1 to deal with bad cases
                self.last_request_dict[sids] = sids

            total_time_to_next_req = time_to_type + time_to_read_response
            # request to send
            self.future_request_to_send[sids] = req_sent_to_infer
            # response which will be recieved
            self.future_request_recieved[sids] = response_rcvd_back
            # future request time
            self.future_request_time_to_send[sids] = total_time_to_next_req
            # Track read and type times separately for advisory
            self.future_request_read_time[sids] = time_to_read_response
            self.future_request_type_time[sids] = time_to_type
            if sids in self.advisory_sent:
                del self.advisory_sent[sids]  # Reset for next request
            print("Time to next rq {}".format(self.future_request_time_to_send))

        if False:
            for sids, req_type in new_session_ids_ready_to_send:
                is_last = False
                if req_type == 1:
                    # first request
                    req_sent_to_infer = self.active_sessions[sids].pop(0)["value"]
                    time_to_type = len(req_sent_to_infer) / self.session_type_speeds[sids]
                    time_to_read_response = 0  # nothing to read all request sent
                if req_type == 2:
                    # continuing session
                    response_rcvd_back = self.active_sessions[sids].pop(0)["value"]
                    if len(self.active_sessions[sids]) != 0:
                        req_sent_to_infer = self.active_sessions[sids].pop(0)["value"]
                    else:
                        req_sent_to_infer = ""
                    time_to_type = len(req_sent_to_infer) / self.session_type_speeds[sids]
                    time_to_read_response = (
                        len(response_rcvd_back) / self.session_read_speeds[sids]
                    )

                if len(self.active_sessions[sids]) == 0:
                    # no more elements left
                    self.last_request_dict[sids] = sids
                total_time_to_next_req = time_to_type + time_to_read_response
                self.future_request_to_send[sids] = req_sent_to_infer
                self.future_request_time_to_send[sids] = total_time_to_next_req


def parse_args(parser):
    """
    Parse Arguments
    """
    parser.add_argument(
        "--reciever-port", type=str, default=50050, help="Load Generation RPC Port"
    )
    parser.add_argument(
        "--scheduler-ip", type=str, default="localhost", help="IP address of scheduler"
    )
    parser.add_argument(
        "--scheduler-port",
        type=str,
        default="50051",
        help="IP address of scheduler",
    )
    parser.add_argument("--num-clients", type=int, default=256, help="Load Generation")
    args = parser.parse_args()
    return args


def launch_server(
    reciever_port: str,
) -> Tuple[grpc.Server, received_response.WorkloadReciever]:
    """
    Launch Server
    """
    workload_rcv_server = received_response.WorkloadReciever()
    server = received_response.start_server(workload_rcv_server, reciever_port)
    print("Server Started")
    return server, workload_rcv_server


if __name__ == "__main__":
    args = parse_args(argparse.ArgumentParser(description="Arguments for Workload Gen"))
    workload_gen = WorkloadGen(
        args.reciever_port, args.scheduler_ip, args.scheduler_port, args.num_clients
    )
    workload_gen.send_data()
