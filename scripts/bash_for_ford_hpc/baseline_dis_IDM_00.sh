cd /s/szhan117/highway-env/
export PYTHONPATH=/s/szhan117/highway-env/
python3 ./scripts/train_baselines.py --env=highway-discrete-v0 --save_path=/s/szhan117/highway-env/models/baseline_dis/00 --env_json=/s/szhan117/highway-env/scripts/config/IDM.json
