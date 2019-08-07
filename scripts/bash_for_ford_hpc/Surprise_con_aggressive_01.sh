#!/bin/bash
cd /s/szhan117/highway-env/
export PYTHONPATH=/s/szhan117/highway-env/
python3 ./scripts/train.py \
--env=highway-continuous-intrinsic-rew-v0 \
--save_path=/s/szhan117/highway-env/models/Surprise_con_aggressive/01/latest \
--env_json=/s/szhan117/highway-env/scripts/config/Aggressive.json
