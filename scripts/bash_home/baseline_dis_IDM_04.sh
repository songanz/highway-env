#!/bin/bash
cd /home/songanz/Documents/Git_repo/highway-env/
export PYTHONPATH=/home/songanz/Documents/Git_repo/highway-env/
python3 ./scripts/train.py \
--env=highway-discrete-v0 \
--save_path=/home/songanz/Documents/Git_repo/highway-env/models/baseline_dis/04/latest \
--env_json=/home/songanz/Documents/Git_repo/highway-env/scripts/config/IDM.json
