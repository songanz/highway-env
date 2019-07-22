import os
import baselines.run as run
import highway_env  # don't remove, for registration the new game
import sys


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # for remove TF warning

cwd = os.getcwd()
env_json_path = os.path.abspath(cwd + '/scripts/config/IDM.json')
save_eval_path = os.path.abspath(cwd + '/models/evaluation/evalHist_IDM2IDM')
load_path = os.path.abspath(cwd + '/models/baseline_con_00/latest')

# f = open(cwd + "/models/test.out", 'w')
# sys.stdout = f

DEFAULT_ARGUMENTS = [
    "--env=highway-continuous-v0",  # if use other env. the CVAE model will update!!!
    "--alg=trpo_mpi",
    "--num_timesteps=1e6",  # episode * steps = num_timesteps = 1e6

    # policy net parameter
    "--network=mlp",
    "--num_layers=3",
    "--num_hidden=124",
    "--activation=tf.tanh",

    "--num_env=0",  # >1 for mpi, disabled for online learning
    "--env_json=" + env_json_path,
    "--save_eval_path=" + save_eval_path,
    "--load_path=" + load_path,
    "--save_video_interval=0",
    "--play"
]

if __name__ == "__main__":
    args = sys.argv
    if len(args) <= 1:
        args = DEFAULT_ARGUMENTS
    # run.main(args)  # for training
    run.animation(args)  # for animation
    # f.close()