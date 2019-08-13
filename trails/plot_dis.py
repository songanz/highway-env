import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


ddqn_dis_IDM = []
ddqn_time_steps = []
trpo_dis_IDM = []
trpo_time_steps = []

# load data
for i in range(1):
    deepq_dis_IDM_path = os.path.abspath(os.getcwd() + '/trails/dynamic_dv_5_ddqn/baseline_dis/0' + str(i) + '/latest_log/progress.csv')
    trpo_dis_IDM_path = os.path.abspath(os.getcwd() + '/trails/dynamic_dv_5_trpo/baseline_dis/0' + str(i) + '/latest_log/progress.csv')

    ddqn_dis_IDM.append(np.array(pd.read_csv(deepq_dis_IDM_path)['mean 100 episode reward per step']))
    ddqn_time_steps.append(np.array(pd.read_csv(deepq_dis_IDM_path)['steps']))

    trpo_dis_IDM.append(np.array(pd.read_csv(trpo_dis_IDM_path)['EpRew/step']))
    trpo_time_steps.append(np.array(pd.read_csv(trpo_dis_IDM_path)['TimestepsSoFar']))

trpo_dis_IDM = np.array(trpo_dis_IDM)
trpo_dis_IDM_m = np.mean(trpo_dis_IDM, axis=0)
trpo_dis_IDM_s = np.std(trpo_dis_IDM, axis=0)
trpo_time_steps = np.mean(trpo_time_steps, axis=0)

ddqn_dis_IDM = np.array(ddqn_dis_IDM)
deepq_dis_IDM_m = np.mean(ddqn_dis_IDM, axis=0)
deepq_dis_IDM_s = np.std(ddqn_dis_IDM, axis=0)
ddqn_time_steps = np.mean(ddqn_time_steps, axis=0)

# Average episode
# for i in range(total_ep):
#     surprise[i] = np.mean(surprise[i:i+50], axis=0)
#     baseline[i] = np.mean(baseline[i:i+50], axis=0)

fig = plt.gcf()
fig.set_size_inches(7,4)

# Draw lines
plt.plot(ddqn_time_steps, deepq_dis_IDM_m, color="#111111", label="DDQN")
plt.plot(trpo_time_steps, trpo_dis_IDM_m, color="#004ec9", label="Baseline TRPO")

# Draw bands
plt.fill_between(ddqn_time_steps, deepq_dis_IDM_m - deepq_dis_IDM_s,
                 deepq_dis_IDM_m + deepq_dis_IDM_s, color="#DDDDDD", alpha=0.5)
plt.fill_between(trpo_time_steps, trpo_dis_IDM_m - trpo_dis_IDM_s,
                 trpo_dis_IDM_m + trpo_dis_IDM_s, color="#7997c6", alpha=0.5)

# Create plot
plt.xlabel("Training episode"), plt.ylabel("Average reward")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()