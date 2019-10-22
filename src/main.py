import numpy as np
import os
import collections
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
import yaml
import datetime

from run import run

import pdb



SETTINGS['CAPTURE_MODE'] = "fd" # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


@ex.main
def my_main(_run, _config, _log, env_args):
    # Setting the random seed throughout the modules
    np.random.seed(_config["seed"])
    th.manual_seed(_config["seed"])
    env_args['seed'] = _config["seed"]
    # th.cuda.manual_seed(_config["seed"])
    
#     np.random.seed(12345)
#     th.manual_seed(12345)
#     env_args['seed'] =12345
#     th.cuda.manual_seed(12345)
    

    # run the framework
    run(_run, _config, _log)


def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


if __name__ == '__main__':
    params = deepcopy(sys.argv)

    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")
    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)
    
    # specify the map for experiment
    # CUDA_VISIBLE_DEVICES=3 python3 src/main.py --config=coma_smac --env-config=sc2 --map=3m
    map_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == "--map":
            map_name = _v.split("=")[1]
            del params[_i]
            break
    if map_name:
        config_dict['env_args']['map_name'] = map_name
    else: 
        map_name = config_dict['env_args']['map_name'] 
        
    # now add all the config to sacred
    ex.add_config(config_dict)

    # Save to disk by default for sacred
    unique_token = "{}_{}_{}".format(config_dict['name'], map_name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    logger.info("Saving to FileStorageObserver in results/sacred/{}/.".format(unique_token))  # qmix_smac_3m
    file_obs_path = os.path.join(results_path, "sacred", unique_token)
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    ex.run_commandline(params)

