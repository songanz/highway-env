#!/bin/bash
cd /s/szhan117/highway-env/
export PYTHONPATH=/s/szhan117/highway-env/
python3 ./scripts/train.py \
--env=highway-discrete-v0 \
--save_path=/s/szhan117/highway-env/models/baseline_dis/03/latest \
--env_json=/s/szhan117/highway-env/scripts/config/IDM.json
