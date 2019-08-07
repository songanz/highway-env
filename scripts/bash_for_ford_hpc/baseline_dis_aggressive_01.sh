#!/bin/bash
cd /s/szhan117/highway-env/
export PYTHONPATH=/s/szhan117/highway-env/
python3 ./scripts/train.py \
--env=highway-discrete-v0 \
--save_path=/s/szhan117/highway-env/models/baseline_dis_aggressive/01/latest \
--env_json=/s/szhan117/highway-env/scripts/config/Aggressive.json
