import os
import stable_baselines.run as run
import highway_env  # don't remove, for registration the new game
import sys


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # for remove TF warning

cwd = os.getcwd()
env_json_path = os.path.abspath(cwd + '/scripts/config/IDM.json')
save_path = os.path.abspath(cwd + '/trails/00/latest')
# env = "highway-continuous-intrinsic-rew-v0"
env = "highway-discrete-adversarial-v0"

DEFAULT_ARGUMENTS = [
    "--env=" + env,
    "--alg=dqn",
    "--num_timesteps=5e5",  # episode * steps = num_timesteps = 1e6

    # policy net parameter
    "--network=MlpPolicy",
    "--act_fun=tf.tanh",
    "--layers=[124,124,124]",

    "--num_env=0",  # >1 for mpi, disabled for online learning
    "--save_video_interval=0",

    "--save_path=" + save_path,
    "--env_json=" + env_json_path,

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
        args.append(i+'='+default_args_dic[i])

    # User should not specify the log_path, the log will in the same directory as the save_path
    try:
        # remove accidentally added log_path
        log_itm = [s for s in args if "--log_path" in s][0]
        args.remove(log_itm)
    except IndexError:
        pass

    t = int(0)
    # Will not raise any error, sine the save_path is in DEFAULT_ARGUMENTS
    itm = [s for s in args if "--save_path" in s][0]
    fullpath = itm.split("=")[1]
    directory = os.path.dirname(fullpath)
    filename = fullpath.split(directory)[1]
    if os.path.exists(directory):
        args.remove(itm)
        new_dir = directory
        while os.path.exists(new_dir):
            t += 1
            new_dir = directory[:-1] + str(t)
        args.append("--save_path=" + new_dir + filename)
        args.append("--log_path=" + new_dir + filename + '_log')
    else:
        args.append("--log_path=" + directory + filename + '_log')

    if 'intrinsic-rew' in [s for s in args if "--env=" in s][0] and\
            "trpo" not in [s for s in args if "--alg=" in s][0]:  # only for off-policy method
        item = [s for s in args if "--env=" in s][0]
        ind = args.index(item)
        args[ind] = args[ind].replace('intrinsic-rew-', '')
        args.append("--surprise=True")

    run.main(args)  # for training
