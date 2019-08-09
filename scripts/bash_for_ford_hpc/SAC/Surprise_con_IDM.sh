#!/bin/bash
cd /s/szhan117/highway-env/
export PYTHONPATH=/s/szhan117/highway-env/
python3 ./scripts/train_stable_baselines.py \
--env=highway-continuous-intrinsic-rew-v0 \
--save_path=/s/szhan117/highway-env/models/sac/Surprise_con/00/latest \
--env_json=/s/szhan117/highway-env/scripts/config/IDM.json \
--alg=sac
