#!/bin/bash
cd /s/szhan117/highway-env/
export PYTHONPATH=/s/szhan117/highway-env/
python3 ./scripts/train.py \
--env=highway-discrete-imagine-v0 \
--save_path=/s/szhan117/highway-env/models/Imagine_dis_IDM2Agg_wIDM_model/00/latest \
--env_json=/s/szhan117/highway-env/scripts/config/Aggressive.json \
--load_path=/s/szhan117/highway-env/models/baseline_dis/00/latest \
--CVAE_path=/models/CVAE/Environment_model_IDMVehicle1_discrete_00.pth.tar
