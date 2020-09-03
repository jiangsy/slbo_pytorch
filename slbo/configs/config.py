import os

import yaml
from munch import DefaultMunch
from yaml import Loader
import collections

try:
    from slbo.misc import logger
except ImportError:
    from stable_baselines import logger


def flatten(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, str(v)))
    return dict(items)


class Config:
    def __new__(cls, config_path='config.yaml'):
        if not config_path.startswith('/'):
            config_path = os.path.join(os.path.dirname(__file__), config_path)
        logger.log('Load configs from {}.'.format(config_path))
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.load(f, Loader=Loader)
        config = DefaultMunch.fromDict(config_dict, object())
        return config, flatten(config_dict)
