#!/bin/bash
cd /s/szhan117/highway-env/
export PYTHONPATH=/s/szhan117/highway-env/
python3 ./scripts/train_surprise.py --env=highway-discrete-v0 --save_path=/s/szhan117/highway-env/models/Surprise_dis_aggressive/00 --env_json=/s/szhan117/highway-env/scripts/config/Aggressive.json
