#!/bin/bash
cd /s/szhan117/highway-env/
export PYTHONPATH=/s/szhan117/highway-env/
python3 ./scripts/train_surprise.py \
--env=highway-discrete-intrinsic-rew-v0 \
--save_path=/s/szhan117/highway-env/models/Surprise_dis/01/latest \
--env_json=/s/szhan117/highway-env/scripts/config/IDM.json
