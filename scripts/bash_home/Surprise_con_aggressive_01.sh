#!/bin/bash
cd /home/songanz/Documents/Git_repo/highway-env/
export PYTHONPATH=/home/songanz/Documents/Git_repo/highway-env/
python3 ./scripts/train.py \
--env=highway-continuous-intrinsic-rew-v0 \
--save_path=/home/songanz/Documents/Git_repo/highway-env/models/Surprise_con_aggressive/01/latest \
--env_json=/home/songanz/Documents/Git_repo/highway-env/scripts/config/Aggressive.json
