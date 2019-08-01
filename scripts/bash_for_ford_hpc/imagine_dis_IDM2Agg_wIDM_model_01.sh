#!/bin/bash
cd /s/szhan117/highway-env/
export PYTHONPATH=/s/szhan117/highway-env/
python3 ./scripts/train_imagine.py \
--env=highway-discrete-imagine-v0 \
--save_path=/s/szhan117/highway-env/models/Imagine_dis_IDM2Agg_wIDM_model/01/latest \
--env_json=/s/szhan117/highway-env/scripts/config/im_Aggressive.json \
--load_path=/s/szhan117/highway-env/models/baseline_dis/00/latest
