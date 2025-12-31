import os
import sys
import json
import time
import queue
import grpc
import argparse
from typing import Tuple
from concurrent import futures

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
        # remove session ids which have sent last request
        self.last_request_dict = dict()  # session ids -> session ids

        self.read_speed_char_ps = (
            25  # read speed per second #TODO: Change to distribution
        )
        self.type_speed_char_ps = 6  # type speed #TODO: Change to distribution
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
                    else:
                        new_session_ids_ready_to_send.append(
                            (response["session_id"], 2)
                        )
                # We now have all possible request to send
                self.update_time_to_next_req(new_session_ids_ready_to_send)
                # find min time client
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

        except KeyboardInterrupt:
            print("Exiting Server")
            self.server.stop(0)

    def subtract_time_dict(self, min_time):
        """
        Modify dictionary time
        """

        for key in self.future_request_time_to_send:
            self.future_request_time_to_send[key] -= min_time

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
                time_to_type = len(req_sent_to_infer) / self.type_speed_char_ps
                time_to_read_response = 0  # nothing to read first request sent

            if req_type == 2:
                # second request
                req_sent_to_infer = self.active_sessions[sids].pop(0)["value"]
                response_rcvd_back = self.active_sessions[sids].pop(0)["value"]
                time_to_type = len(req_sent_to_infer) / self.type_speed_char_ps
                time_to_read_response = (
                    len(self.future_request_recieved[sids]) / self.read_speed_char_ps
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
            print("Time to next rq {}".format(self.future_request_time_to_send))

        if False:
            for sids, req_type in new_session_ids_ready_to_send:
                is_last = False
                if req_type == 1:
                    # first request
                    req_sent_to_infer = self.active_sessions[sids].pop(0)["value"]
                    time_to_type = len(req_sent_to_infer) / self.type_speed_char_ps
                    time_to_read_response = 0  # nothing to read all request sent
                if req_type == 2:
                    # continuing session
                    response_rcvd_back = self.active_sessions[sids].pop(0)["value"]
                    if len(self.active_sessions[sids]) != 0:
                        req_sent_to_infer = self.active_sessions[sids].pop(0)["value"]
                    else:
                        req_sent_to_infer = ""
                    time_to_type = len(req_sent_to_infer) / self.type_speed_char_ps
                    time_to_read_response = (
                        len(response_rcvd_back) / self.read_speed_char_ps
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
