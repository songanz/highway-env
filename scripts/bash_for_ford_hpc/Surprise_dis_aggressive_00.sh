#!/bin/bash
cd /s/szhan117/highway-env/
export PYTHONPATH=/s/szhan117/highway-env/
python3 ./scripts/train_surprise.py --env=highway-discrete-intrinsic-rew-v0 --save_path=/s/szhan117/highway-env/models/Surprise_dis_aggressive/00/latest --env_json=/s/szhan117/highway-env/scripts/config/Aggressive.json
