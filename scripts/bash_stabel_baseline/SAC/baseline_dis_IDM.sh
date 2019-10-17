#!/bin/bash
cd /home/songan/Documents/git_repo/highway-env/
export PYTHONPATH=/home/songan/Documents/git_repo/highway-env/
python3 ./scripts/train.py \
--env=highway-discrete-v0 \
--save_path=/home/songan/Documents/git_repo/highway-env/trails/sac/baseline_dis_IDM/00/latest \
--env_json=/home/songan/Documents/git_repo/highway-env/scripts/config/IDM.json
--alg=sac
