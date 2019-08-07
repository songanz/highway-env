import os
import baselines.run as run
import highway_env  # don't remove, for registration the new game
import sys


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # for remove TF warning

cwd = os.getcwd()

env_json_path = os.path.abspath(cwd + '/scripts/config/Aggressive.json')
save_eval_path = os.path.abspath(cwd + '/models/evaluation/temp')
load_path = os.path.abspath(cwd + '/trails/temp/latest')
env = "highway-continuous-v0"

DEFAULT_ARGUMENTS = [
    "--env=" + env,
    "--alg=trpo_mpi",
    "--num_timesteps=5e5",  # episode * steps = num_timesteps = 1e6

    # policy net parameter
    "--network=mlp",
    "--num_layers=3",
    "--num_hidden=124",
    "--activation=tf.tanh",

    "--num_env=0",  # >1 for mpi, disabled for online learning
    "--save_video_interval=0",

    "--save_eval_path=" + save_eval_path,
    "--env_json=" + env_json_path,
    "--log_path=" + save_eval_path + "_log",
    "--load_path=" + load_path,

    "--play"
]

def get_dict_from_list(l):
    d = {}
    temp = [s.split('=')[0] for s in l]
    for x in range(len(temp)):
        try:
            d[temp[x]] = [a.split('=') for a in l][x][1]
        except IndexError:
            continue
    return d

if __name__ == "__main__":
    args = sys.argv
    
    args_dic = get_dict_from_list(args)
    default_args_dic = get_dict_from_list(DEFAULT_ARGUMENTS)

    for i in [s for s in default_args_dic.keys() if s not in args_dic.keys()]:
        args.append(i + '=' + default_args_dic[i])

    run.animation(args)  # for animation
