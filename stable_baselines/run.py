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


try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

try:
    import roboschool
except ImportError:
    roboschool = None

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    env_type = env.entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

# reading benchmark names directly from retro requires
# importing retro here, and for some reason that crashes tensorflow
# in ubuntu
_game_envs['retro'] = {
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
}

def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def common_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='Reacher-v2')
    parser.add_argument('--env_type', help='type of environment, used when the environment type cannot be automatically determined', type=str)
    parser.add_argument('--env_json', help='Path to environment config json file', default=None, type=str)
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--alg', help='Algorithm', type=str, default='ppo2')
    parser.add_argument('--num_timesteps', type=float, default=1e6),
    parser.add_argument('--network', help='network type (mlp, cnn, lstm, cnn_lstm, conv_only)', default=None)
    parser.add_argument('--gamestate', help='game state to load (so far only used in retro games)', default=None)
    parser.add_argument('--num_env', help='Number of environment copies being run in parallel. When not specified, set to number of cpus for Atari, and to 1 for Mujoco', default=None, type=int)
    parser.add_argument('--reward_scale', help='Reward scale factor. Default: 1.0', default=1.0, type=float)
    parser.add_argument('--save_path', help='Path to save trained model to', default=None, type=str)
    parser.add_argument('--save_video_interval', help='Save video every x steps (0 = disabled)', default=0, type=int)
    parser.add_argument('--save_video_length', help='Length of recorded video. Default: 200', default=200, type=int)
    parser.add_argument('--log_path', help='Directory to save learning curve data.', default=None, type=str)
    parser.add_argument('--CVAE_path', help='Directory import CVAE model.', default=None, type=str)
    parser.add_argument('--surprise', type=bool, default=False, help='whether use surprise-based intrinsic reward for exploration')
    parser.add_argument('--play', default=False, action='store_true')
    return parser

def parse_unknown_args(args):
    """
    Parse arguments not consumed by arg parser into a dictionary
    """
    retval = {}
    preceded_by_key = False
    for arg in args:
        if arg.startswith('--'):
            if '=' in arg:
                key = arg.split('=')[0][2:]
                value = arg.split('=')[1]
                retval[key] = value
            else:
                key = arg[2:]
                preceded_by_key = True
        elif preceded_by_key:
            retval[key] = arg
            preceded_by_key = False

    return retval

def parse(v):

    assert isinstance(v, str)
    try:
        return eval(v)
    except (NameError, SyntaxError):
        return v

def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    '''

    return {k: parse(v) for k,v in parse_unknown_args(args).items()}

def configure_logger(log_path, **kwargs):
    if log_path is not None:
        logger.configure(log_path)
    else:
        logger.configure(**kwargs)

def get_env_type(args):
    env_id = args.env

    if args.env_type is not None:
        return args.env_type, env_id

    # Re-parse the gym registry, since we could have new envs since last time.
    for env in gym.envs.registry.all():
        env_type = env.entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)  # This is a set so add is idempotent

    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        if ':' in env_id:
            env_type = re.sub(r':.*', '', env_id)
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    return env_type, env_id

def get_alg_module(args, _env, alg_kwargs, submodule=None):
    submodule = submodule or args.alg
    if not args.surprise:
        alg_module = import_module('.'.join(['stable_baselines', args.alg]))
        policy = alg_kwargs.pop('network', None)
        alg_class = getattr(alg_module, submodule.upper())(policy, _env, policy_kwargs=alg_kwargs)
    else:
        alg_module = import_module('.'.join(['stable_baselines.surprise_off_po', args.alg]))  # the alg from sur folder
        policy = alg_kwargs.pop('network', None)
        CVAE_save_path = args.save_path[:-6]
        """ for surprise based intrinsic reward method, send in surprise=True """
        alg_class = getattr(alg_module, submodule.upper())(policy, _env, policy_kwargs=alg_kwargs,
                                                           surprise=True, CVAE_save_path=CVAE_save_path)

    return alg_class

def get_learn_function(args, _env, alg_kwargs):
    return get_alg_module(args, _env, alg_kwargs).learn

def make_vec_env(env_id, num_env, seed,
                 env_kwargs=None, start_index=0,
                 force_dummy=False, initializer=None):
    env_kwargs = env_kwargs or {}
    mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    seed = seed + 10000 * mpi_rank if seed is not None else None
    logger_dir = logger.get_dir()

    def make_thunk(rank, initializer=None):
        return lambda: make_env(
            env_id=env_id,
            mpi_rank=mpi_rank,
            subrank=rank,
            seed=seed,
            env_kwargs=env_kwargs,
            logger_dir=logger_dir,
            initializer=initializer
        )

    set_global_seeds(seed)
    if not force_dummy and num_env > 1:
        return SubprocVecEnv([make_thunk(i + start_index, initializer=initializer) for i in range(num_env)])
    else:
        return DummyVecEnv([make_thunk(i + start_index, initializer=None) for i in range(num_env)])

def make_env(env_id,
            mpi_rank=0,
            subrank=0,
            seed=None,
            env_kwargs=None,
            logger_dir=None,
            initializer=None):
    if initializer is not None:
        initializer(mpi_rank=mpi_rank, subrank=subrank)

    env_kwargs = env_kwargs or {}

    env = gym.make(env_id, **env_kwargs)

    env.seed(seed + subrank if seed is not None else None)
    env = Monitor(env,
                  logger_dir and os.path.join(logger_dir, str(mpi_rank) + '.' + str(subrank)),
                  allow_early_resets=True)

    return env


def train(args, extra_args):
    env_type, env_id = get_env_type(args)
    print('env_type: {}'.format(env_type))

    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    env = build_env(args)
    if args.save_video_interval != 0:
        env = VecVideoRecorder(env, osp.join(logger.get_dir(), "videos"),
                               record_video_trigger=lambda x: x % args.save_video_interval == 0,
                               video_length=args.save_video_length)

    alg_kwargs = {}
    alg_kwargs.update(extra_args)
    alg_kwargs['network'] = args.network

    learn = get_learn_function(args, env, alg_kwargs)

    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))

    model = learn(total_timesteps, seed=seed)

    return model, env

def build_env(args):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu
    alg = args.alg
    seed = args.seed

    env_type, env_id = get_env_type(args)

    if args.env_json:
        with open(args.env_json) as f:
            env_kwargs = json.loads(f.read())  # need to corresponding to env.__init__ arguments
        # for surprise based model
        if args.CVAE_path:
            env_kwargs["config"]["CVAE_path"] = args.CVAE_path
        env = make_vec_env(env_id, args.num_env or 1, seed, env_kwargs=env_kwargs)
    else:
        env = make_vec_env(env_id, args.num_env or 1, seed)

    return env

def main(args):
    # configure logger, disable logging in child MPI processes (with rank > 0)

    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)  # unknown_args to a dict

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        configure_logger(args.log_path)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        configure_logger(args.log_path, format_strs=[])

    model, env = train(args, extra_args)

    if args.save_path is not None and rank == 0:
        save_path = osp.expanduser(args.save_path)
        model.save(save_path)

    if args.play:
        obs = env.reset()

        state = model.initial_state if hasattr(model, 'initial_state') else None
        dones = np.zeros((1,))

        episode_rew = 0
        while True:
            if state is not None:
                actions, _, state, _ = model.step(obs,S=state, M=dones)
            else:
                actions, _, _, _ = model.step(obs)

            obs, rew, done, _ = env.step(actions)
            episode_rew += rew[0] if isinstance(env, VecEnv) else rew
            env.render()
            done = done.any() if isinstance(done, np.ndarray) else done
            if done:
                print('episode_rew={}'.format(episode_rew))
                episode_rew = 0
                obs = env.reset()

    env.close()

    return model

if __name__ == '__main__':
    main(sys.argv)
