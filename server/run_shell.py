import os
import sys
sys.path.append('.')

from eval.eval_model import eval_model
from eval.parse_config import parse_config


def run_shell(model_file_path, eval_config_file_path, metrics_file_path):
    limit_config, _ = parse_config(eval_config_file_path)

    memory_size = limit_config['memory_size']
    swap_memory_size = limit_config['swap_memory_size']

    cur_dir_path = os.path.dirname(os.path.abspath(__file__))
    shell_path = os.path.join(cur_dir_path, './run.sh')

    # TODO: if leverage existing cgroup?
    os.system('sh {} {} {} {} {} {}'.format(shell_path, memory_size, swap_memory_size, \
        model_file_path, eval_config_file_path, metrics_file_path))


if __name__ == '__main__':
    model_file_path, eval_config_file_path, metrics_file_path = sys.argv[1], sys.argv[2], sys.argv[3]
    run_shell(model_file_path, eval_config_file_path, metrics_file_path)
