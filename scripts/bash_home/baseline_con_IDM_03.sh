#!/bin/bash
cd /home/songanz/Documents/Git_repo/highway-env/
export PYTHONPATH=/home/songanz/Documents/Git_repo/highway-env/
python3 ./scripts/train.py \
--env=highway-continuous-v0 \
--save_path=/home/songanz/Documents/Git_repo/highway-env/models/baseline_con/03/latest \
--env_json=/home/songanz/Documents/Git_repo/highway-env/scripts/config/IDM.json
