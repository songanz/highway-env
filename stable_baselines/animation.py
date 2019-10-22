import sys
import re
import multiprocessing
import os.path as osp
import os
import gym
from collections import defaultdict
import tensorflow as tf
import numpy as np
import json
from collections import deque
import scipy.io as sio
from importlib import import_module

from stable_baselines import logger
from stable_baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from stable_baselines.common.misc_util import set_global_seeds
from stable_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.common.vec_env.base_vec_env import VecEnv
from stable_baselines.bench.monitor import Monitor
from stable_baselines.common.cmd_util import make_atari_env

from stable_baselines.run import *


def animation(args):
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)  # unknown_args to a dict

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        configure_logger(args.log_path)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        configure_logger(args.log_path, format_strs=[])

    env_type, env_id = get_env_type(args)
    env = build_env(args)

    load_path = extra_args['load_path']
    extra_args.pop("load_path", None)
    extra_args.pop("animation", None)

    alg_kwargs = {}
    alg_kwargs.update(extra_args)
    alg_kwargs['network'] = args.network

    model = get_alg_module(args, env, alg_kwargs)

    model.load(load_path, env=env)  # give env, otherwise no lode

    obs = env.reset()
    while True:
        act, _ = model.predict(obs)
        obs, rew, done, _ = env.step(act)
        print("Action: ", env.envs[0].env.ACTIONS[act[0]], "\t",
              "Closest car xy: ",
              "{:.2f}".format(env.envs[0].env.observation.reverse_normalize(obs[0]).x[1]), " ",
              "{:.2f}".format(env.envs[0].env.observation.reverse_normalize(obs[0]).y[1]), "\t",
              "Velocity: ",
              "{:.2f}".format(env.envs[0].env.observation.reverse_normalize(obs[0]).vx[0]), "\t",
              "Lane: ",
              "{:.2f}".format(env.envs[0].env.observation.reverse_normalize(obs[0]).y[0]), "\t",
              "reward: ", rew[0])
        env.render()
        if done:
            obs = env.reset()

if __name__ == '__main__':
    animation(sys.argv)
