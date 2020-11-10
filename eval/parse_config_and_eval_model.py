import sys
sys.path.append('.')
from eval.eval_model import eval_model
from eval.parse_config import parse_config


def parse_config_and_eval_model(model_file_path, eval_config_file_path, metrics_file_path):
    limit_config, infer_config = parse_config(eval_config_file_path)

    device = limit_config['device']
    max_inference_time = limit_config['inference_time']
    max_model_load_time = limit_config['model_load_time']

    test_sample_num = infer_config['test_sample_num']
    input_shape = infer_config['input_shape']

    eval_model(model_file_path, metrics_file_path, input_shape, test_sample_num, max_model_load_time, max_inference_time, device)


if __name__ == '__main__':
    model_file_path, eval_config_file_path, metrics_file_path = sys.argv[1], sys.argv[2], sys.argv[3]
    parse_config_and_eval_model(model_file_path, eval_config_file_path, metrics_file_path)
