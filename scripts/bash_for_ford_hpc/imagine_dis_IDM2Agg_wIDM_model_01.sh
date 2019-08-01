#!/bin/bash
cd /home/songanz/Documents/Git_repo/highway-env/
export PYTHONPATH=/home/songanz/Documents/Git_repo/highway-env/
python3 ./scripts/train_imagine.py \
--env=highway-discrete-imagine-v0 \
--save_path=/home/songanz/Documents/Git_repo/highway-env/models/Imagine_dis_IDM2Agg_wIDM_model/01/latest \
--env_json=/home/songanz/Documents/Git_repo/highway-env/scripts/config/im_Aggressive.json \
--load_path=/home/songanz/Documents/Git_repo/highway-env/models/baseline_dis/00/latest
