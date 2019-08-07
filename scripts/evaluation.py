import os
import baselines.run as run
import highway_env  # don't remove, for registration the new game
import sys


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # for remove TF warning

cwd = os.getcwd()

env_json_path = os.path.abspath(cwd + '/scripts/config/Aggressive.json')
save_eval_path = os.path.abspath(cwd + '/models/evaluation/00')
load_path = os.path.abspath(cwd + '/trails/02/latest')
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
    "--load_path=" + load_path,
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

    # User should not specify the log_path, the log will in the same directory as the save_path
    try:
        # remove accidentally added log_path
        log_itm = [s for s in args if "--log_path" in s][0]
        args.remove(log_itm)
    except IndexError:
        pass

    t = int(0)
    # Will not raise any error, sine the save_eval_path is in DEFAULT_ARGUMENTS
    itm = [s for s in args if "--save_eval_path" in s][0]
    fullpath = itm.split("=")[1]

    if os.path.exists(fullpath):
        args.remove(itm)
        new_path = fullpath
        while os.path.exists(new_path):
            t += 1
            new_path = fullpath[:-1] + str(t)
        args.append("--save_eval_path=" + new_path)
        args.append("--log_path=" + new_path + '_log')
    else:
        args.append("--log_path=" + fullpath + '_log')

    args.append('--play')  # for showing the animation
    run.animation(args)  # for evaluation
