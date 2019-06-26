import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # for remove TF warning
import baselines.run as run

import highway_env  # don't remove, for registration the new game

import sys
# f = open("../models/test.out", 'w')
# sys.stdout = f

DEFAULT_ARGUMENTS = [
    # "--env=highway-continuous-v0",
    "--env=highway-continuous-intrinsic-rew-v0",
    "--alg=trpo_mpi",
    # "--num_timesteps=1e7",  # episode * steps = num_timesteps = 1e7
    "--num_timesteps=6e6",  # episode * steps = num_timesteps = 6e6
    # "--num_timesteps=1e3",  # testing

    # policy net parameter
    "--network=mlp",
    "--num_layers=3",
    "--num_hidden=124",
    "--activation=tf.tanh",

    "--num_env=0",  # >1 for mpi, disabled for online learning
    "--save_path=C:/Users/szhan117/Documents/git_repo/highway-env/models/Surprise/latest",
    # "--load_path=C:/Users/szhan117/Documents/git_repo/highway-env/models/crazy_driver/latest",
    # "--load_path=C:/Users/szhan117/Documents/git_repo/highway-env/models/latest",
    # "--load_path=~/models/latest",
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
