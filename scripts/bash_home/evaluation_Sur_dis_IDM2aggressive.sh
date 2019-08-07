#!/bin/bash
cd /home/songanz/Documents/Git_repo/highway-env/
export PYTHONPATH=/home/songanz/Documents/Git_repo/highway-env/
python3 ./scripts/evaluation.py \
--env=highway-discrete-v0 \
--save_eval_path=/home/songanz/Documents/Git_repo/highway-env/models/evaluation/evalHist_Surprise_dis_IDM2Aggressive \
--env_json=/home/songanz/Documents/Git_repo/highway-env/scripts/config/Aggressive.json \
--load_path=/home/songanz/Documents/Git_repo/highway-env/models/Surprise_dis/00/latest
