import os

import argparse
import yaml
from munch import DefaultMunch
from yaml import Loader
import collections

from slbo.misc import logger


def flatten(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, str(v)))
    return dict(items)


def change_dict_value_recursive(obj, keys, value):
    try:
        for key in keys[:-1]:
            obj = obj[key]
        obj[keys[-1]] = value
    except KeyError:
        raise KeyError('Incorrect key sequences')


class Config:
    def __new__(cls, config_paths=['config.yaml']):
        parser = argparse.ArgumentParser(description='Stochastic Lower Bound Optimization')
        parser.add_argument('-c', '--configs', type=str, help='configuration file (YAML)', nargs='+', action='append')
        parser.add_argument('-s', '--set', type=str, help='additional options', nargs='*', action='append')

        args, unknown = parser.parse_known_args()
        config_dict = {}

        if args.configs:
            config_paths = args.configs

        if isinstance(config_paths, str):
            config_paths = [config_paths]

        for config_path in config_paths:
            logger.info('Loading configs from {}.'.format(config_path))
            if not config_path.startswith('/'):
                config_path = os.path.join(os.path.dirname(__file__), config_path)

            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict.update(yaml.load(f, Loader=Loader))

        if args.set:
            for instruction in sum(args.set, []):
                path, value = instruction.split('=')
                change_dict_value_recursive(config_dict, path.split('.'), eval(value))
                logger.info('Hyperparams {} is reset to {}'.format(path, value))

        config = DefaultMunch.fromDict(config_dict, object())
        config_dict = flatten(config_dict)
        logged_config_dict = {}

        for key, value in config_dict.items():
            if key.find('.') >= 0:
                logged_config_dict[key] = value
        return config, logged_config_dict


