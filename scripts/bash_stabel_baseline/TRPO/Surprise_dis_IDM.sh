#!/bin/bash
cd /home/songanz/Documents/Git_repo/highway-env/
export PYTHONPATH=/home/songanz/Documents/Git_repo/highway-env/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/extras/CUPTI/lib64
python3 ./scripts/train.py \
--env=highway-discrete-intrinsic-rew-v0 \
--save_path=/home/songanz/Documents/Git_repo/highway-env/trails/trpo/Surprise_dis/00/latest \
--env_json=/home/songanz/Documents/Git_repo/highway-env/scripts/config/IDM.json \
--alg=trpo_mpi
