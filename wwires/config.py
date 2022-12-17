import yaml
from pathlib import Path

class Configuration:
    def __init__(self, config_name, path_head = 'config'):
        self.config_name = config_name
        try:
            with open(Path(path_head + '/' + config_name + '.yml').resolve(), 'r') as f:
                self.config_dict = yaml.load(f, Loader = yaml.FullLoader)
        except FileNotFoundError:
            print(f'ERROR: {config_name} not found in path {path_head}')
            exit()

    def __getitem__(self, name):
        return self.config_dict[name]