#!/bin/bash
cd /s/szhan117/highway-env/
export PYTHONPATH=/s/szhan117/highway-env/
python3 ./scripts/train.py \
--env=highway-discrete-intrinsic-rew-v0 \
--save_path=/s/szhan117/highway-env/models/dqn/Surprise_dis_aggressive/00/latest \
--env_json=/s/szhan117/highway-env/scripts/config/Aggressive.json \
--alg=deepq
