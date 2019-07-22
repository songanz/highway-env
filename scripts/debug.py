import os
import baselines.run as run
import highway_env  # don't remove, for registration the new game
import sys


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # for remove TF warning

cwd = os.getcwd()
env_json_path = os.path.abspath(cwd + '/scripts/config/im_IDM.json')
# env_json_path = os.path.abspath(cwd + '/scripts/config/IDM.json')
save_path = os.path.abspath(cwd + '/models/debug/latest')
load_path = os.path.abspath(cwd + '/models/Surprise/latest')

# f = open(cwd + "/models/test.out", 'w')
# sys.stdout = f

DEFAULT_ARGUMENTS = [
    # "--env=highway-v0",
    # "--env=highway-continuous-v0",
    # "--env=highway-continuous-intrinsic-rew-v0",
    # "--env=highway-continuous-imagine-v0",
    # "--env=highway-discrete-v0",
    # "--env=highway-discrete-intrinsic-rew-v0",
    "--env=highway-discrete-imagine-v0",


    "--alg=trpo_mpi",
    "--num_timesteps=1e6",  # episode * steps = num_timesteps = 1e6


    "--network=mlp",
    "--num_layers=3",
    "--num_hidden=124",
    "--activation=tf.tanh",


    "--num_env=0",  # >1 for mpi, disabled for online learning
    "--env_json=" + env_json_path,


    "--save_path=" + save_path,
    "--save_video_interval=0",
    "--play"
]

if __name__ == "__main__":
    args = sys.argv
    if len(args) <= 1:
        args = DEFAULT_ARGUMENTS
    run.main(args)  # for training
    # run.animation(args)  # for animation
    # f.close()