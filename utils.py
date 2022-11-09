import json

class Config(dict): 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            config = json.loads(f.read())
            return Config(config)
    
    @classmethod
    def save(cls, config_dict, file):
        with open(file, 'w') as f:
            config = json.dump(config_dict, f, indent=4)
            print('config file is saved on {}'.format(file))