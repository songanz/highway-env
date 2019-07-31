cd /s/szhan117/highway-env/
export PYTHONPATH=/s/szhan117/highway-env/
python3 ./scripts/evaluation.py --env=highway-discrete-v0 --save_eval_path=/s/szhan117/highway-env/models/evaluation/evalHist_Surprise_con_IDM2Aggressive --env_json_path=/s/szhan117/highway-env/scripts/config/Aggressive.json --load_path=/s/szhan117/highway-env/models/Surprise_dis_aggressive/Surprise_dis_aggressive_00/latest
