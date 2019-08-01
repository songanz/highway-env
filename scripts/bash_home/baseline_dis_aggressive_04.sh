#!/bin/bash
cd /home/songanz/Documents/Git_repo/highway-env/
export PYTHONPATH=/home/songanz/Documents/Git_repo/highway-env/
python3 ./scripts/train_baselines.py \
--env=highway-discrete-v0 \
--save_path=/home/songanz/Documents/Git_repo/highway-env/models/baseline_dis_aggressive/00/latest \
--env_json=/home/songanz/Documents/Git_repo/highway-env/scripts/config/Aggressive.json
