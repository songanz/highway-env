#!/bin/bash
cd /home/songanz/Documents/Git_repo/highway-env/
export PYTHONPATH=/home/songanz/Documents/Git_repo/highway-env/
python3 ./scripts/train_imagine.py \
--env=highway-continuous-imagine-v0 \
--save_path=/home/songanz/Documents/Git_repo/highway-env/models/Imagine_con_IDM2Agg_wIDM_model/00/latest \
--env_json=/home/songanz/Documents/Git_repo/highway-env/scripts/config/Aggressive.json \
--load_path=/home/songanz/Documents/Git_repo/highway-env/models/baseline_con/00/latest \
--CVAE_path=/models/CVAE/Environment_model_IDMVehicle1_00.pth.tar
