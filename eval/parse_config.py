import yaml


def parse_config(yaml_file_path):
    config_file = open(yaml_file_path, 'r')
    config = yaml.load(config_file, Loader=yaml.Loader)

    limit_config, infer_config = config['limitation'], config['inference']

    return limit_config, infer_config
