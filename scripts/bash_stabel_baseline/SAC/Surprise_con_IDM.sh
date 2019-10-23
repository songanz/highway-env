#!/bin/bash
cd /home/songanz/Documents/Git_repo/highway-env/
export PYTHONPATH=/home/songanz/Documents/Git_repo/highway-env/
python3 ./scripts/train.py \
--env=highway-continuous-v0 \
--surprise=True \
--save_path=/home/songanz/Documents/Git_repo/highway-env/trails/sac/Surprise_con/00/latest \
--env_json=/home/songanz/Documents/Git_repo/highway-env/scripts/config/IDM.json \
--alg=sac
