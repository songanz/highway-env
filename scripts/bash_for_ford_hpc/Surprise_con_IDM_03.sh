#!/bin/bash
cd /s/szhan117/highway-env/
export PYTHONPATH=/s/szhan117/highway-env/
python3 ./scripts/train_surprise.py \
--env=highway-continuous-intrinsic-rew-v0 \
--save_path=/s/szhan117/highway-env/models/Surprise_con/03/latest \
--env_json=/s/szhan117/highway-env/scripts/config/IDM.json