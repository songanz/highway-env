import os
import baselines.run as run
import highway_env  # don't remove, for registration the new game
import sys


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # for remove TF warning

cwd = os.getcwd()

env_json_path = os.path.abspath(cwd + '/scripts/config/Aggressive.json')
# last save name must be 'latest', otherwise check the trpo_mpi file
save_path = os.path.abspath(cwd + '/models/Surprise_dis_aggressive/latest')
load_path = os.path.abspath(cwd + '/models/baseline/latest')
env = "highway-discrete-intrinsic-rew-v0"

# f = open(cwd + "/models/test.out", 'w')
# sys.stdout = f

DEFAULT_ARGUMENTS = [
    "--alg=trpo_mpi",
    "--num_timesteps=1e6",  # episode * steps = num_timesteps = 1e6

    # policy net parameter
    "--network=mlp",
    "--num_layers=3",
    "--num_hidden=124",
    "--activation=tf.tanh",

    "--num_env=0",  # >1 for mpi, disabled for online learning
    "--save_video_interval=0",
    "--play"
]

if __name__ == "__main__":
    args = sys.argv
    if len(args) <= 1:
        args = DEFAULT_ARGUMENTS
    else:
        DEFAULT_ARGUMENTS.extend(args)
        args = DEFAULT_ARGUMENTS

    if not [s for s in args if "--save_path=" in s]:
        args.append("--save_path=" + save_path)

    if not [s for s in args if "--env=" in s]:
        args.append("--env=" + env)

    if not [s for s in args if "--env_json=" in s]:
        args.append("--env_json=" + env_json_path)

    if not [s for s in args if "--log_path=" in s]:
        log_path = [s for s in args if "--save_path=" in s][0][12:]
        args.append("--log_path=" + log_path + "_log")

    run.main(args)  # for training
    # run.animation(args)  # for animation
    # f.close()