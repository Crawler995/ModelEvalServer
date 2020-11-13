from concurrent import futures
import grpc
import time
import os
import torch

from server.rpc_package.eval_model_in_edge_pb2_grpc import add_EvalModelInEdgeServiceServicer_to_server, \
    EvalModelInEdgeServiceServicer
from server.rpc_package.eval_model_in_edge_pb2 import GetModelMetricsRequest, ModelMetricsReply


import sys
sys.path.append('.')
from eval.parse_config_and_eval_model import parse_config_and_eval_model


def remove(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


class EvalModelInEdge(EvalModelInEdgeServiceServicer):
    def save_bytes_to_file(self, bytes_data, file_path):
        with open(file_path, 'wb') as f:
            f.write(bytes_data)

    def get_files_tmp_path(self):
        cur_dir_path = os.path.dirname(os.path.abspath(__file__))
        time_str = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
        model_file_tmp_path = os.path.join(cur_dir_path, './model_{}.jit.zip'.format(time_str))
        eval_config_file_tmp_path = os.path.join(cur_dir_path, './eval_config_{}.yaml'.format(time_str))
        metrics_file_tmp_path = os.path.join(cur_dir_path, './metrics_{}'.format(time_str))

        return model_file_tmp_path, eval_config_file_tmp_path, metrics_file_tmp_path

    def get_model_metrics_under_cgroup(self, model_file_path, eval_config_file_path, metrics_file_path):
        cur_dir_path = os.path.dirname(os.path.abspath(__file__))
        run_shell_py_path = os.path.join(cur_dir_path, './run_shell.py')

        print('running {} to eval model under cgroup...'.format(run_shell_py_path))
        os.system('python {} {} {} {}'.format(run_shell_py_path, model_file_path, eval_config_file_path, metrics_file_path))

        metrics = torch.load(metrics_file_path)
        return metrics

    def get_model_metrics(self, request, context):
        model_bytes_data, config_bytes_data = request.model_file, request.config_file
        model_file_tmp_path, eval_config_file_tmp_path, metrics_file_tmp_path = self.get_files_tmp_path()
        self.save_bytes_to_file(model_bytes_data, model_file_tmp_path)
        self.save_bytes_to_file(config_bytes_data, eval_config_file_tmp_path)

        print('---------------------------------------')
        metrics = self.get_model_metrics_under_cgroup(model_file_tmp_path, eval_config_file_tmp_path, metrics_file_tmp_path)

        remove(model_file_tmp_path)
        remove(eval_config_file_tmp_path)
        remove(metrics_file_tmp_path)
        print('---------------------------------------')

        return ModelMetricsReply(**metrics)

        
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2), options=[
        ('grpc.max_receive_message_length', 256 * 1024 * 1024),
    ])
    add_EvalModelInEdgeServiceServicer_to_server(EvalModelInEdge(), server)

    server.add_insecure_port('[::]:50000')
    server.start()
    try:
        while True:
            time.sleep(60 * 60 * 24)
    except KeyboardInterrupt:
        server.stop(0)
