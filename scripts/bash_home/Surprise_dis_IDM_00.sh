#!/bin/bash
cd /home/songanz/Documents/Git_repo/highway-env/
export PYTHONPATH=/home/songanz/Documents/Git_repo/highway-env/
python3 ./scripts/train_surprise.py --env=highway-discrete-intrinsic-rew-v0 --save_path=/home/songanz/Documents/Git_repo/highway-env/models/Surprise_dis/00/latest --env_json=/home/songanz/Documents/Git_repo/highway-env/scripts/config/IDM.json
