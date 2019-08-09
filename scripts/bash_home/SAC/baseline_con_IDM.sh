#!/bin/bash
cd /home/songanz/Documents/Git_repo/highway-env/
export PYTHONPATH=/home/songanz/Documents/Git_repo/highway-env/
python3 ./scripts/train_stable_baselines.py \
--env=highway-continuous-v0 \
--save_path=/home/songanz/Documents/Git_repo/highway-env/models/sac/baseline_con/00/latest \
--env_json=/home/songanz/Documents/Git_repo/highway-env/scripts/config/IDM.json \
--alg=sac
