import os

import yaml
from munch import DefaultMunch
from yaml import Loader

try:
    from slbo.misc import logger
except ImportError:
    from stable_baselines import logger


class Config:
    def __new__(cls, config_path='config.yaml'):
        if not config_path.startswith('/'):
            config_path = os.path.join(os.path.dirname(__file__), config_path)
        logger.log('Load configs from {}.'.format(config_path))
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.load(f, Loader=Loader)
        config = DefaultMunch.fromDict(config_dict, object())
        return config
