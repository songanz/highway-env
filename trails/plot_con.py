import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


ddpg_con_IDM = []
ddpg_time_steps = []
trpo_con_IDM = []
trpo_time_steps = []
sac_con_IDM = []
sac_time_steps = []

# load data
for i in range(1):
    ddpg_con_IDM_path = os.path.abspath(os.getcwd() + '/trails/ddpg/baseline_con/0' + str(i) + '/latest_log/progress.csv')
    sac_con_IDM_path = os.path.abspath(os.getcwd() + '/trails/sac/baseline_con/0' + str(i) + '/latest_log/progress.csv')
    trpo_con_IDM_path = os.path.abspath(os.getcwd() + '/trails/trpo/baseline_con/0' + str(i) + '/latest_log/progress.csv')

    ddpg_con_IDM.append(np.array(pd.read_csv(ddpg_con_IDM_path)['rollout/return_per_step_mean']))
    ddpg_time_steps.append(np.array(pd.read_csv(ddpg_con_IDM_path)['total/steps']))

    sac_con_IDM.append(np.array(pd.read_csv(sac_con_IDM_path)['mean 100 episode reward per step']))
    sac_time_steps.append(np.array(pd.read_csv(sac_con_IDM_path)['total timesteps']))

    trpo_con_IDM.append(np.array(pd.read_csv(trpo_con_IDM_path)['EpRew/step']))
    trpo_time_steps.append(np.array(pd.read_csv(trpo_con_IDM_path)['TimestepsSoFar']))

trpo_con_IDM = np.array(trpo_con_IDM)
trpo_con_IDM_m = np.mean(trpo_con_IDM, axis=0)
trpo_con_IDM_s = np.std(trpo_con_IDM, axis=0)
trpo_time_steps = np.mean(trpo_time_steps, axis=0)

ddpg_con_IDM = np.array(ddpg_con_IDM)
ddpg_con_IDM_m = np.mean(ddpg_con_IDM, axis=0)
ddpg_con_IDM_s = np.std(ddpg_con_IDM, axis=0)
ddpg_time_steps = np.mean(ddpg_time_steps, axis=0)

sac_con_IDM = np.array(sac_con_IDM)
sac_con_IDM_m = np.mean(sac_con_IDM, axis=0)
sac_con_IDM_s = np.std(sac_con_IDM, axis=0)
sac_time_steps = np.mean(sac_time_steps, axis=0)

# Average episode
# for i in range(total_ep):
#     surprise[i] = np.mean(surprise[i:i+50], axis=0)
#     baseline[i] = np.mean(baseline[i:i+50], axis=0)

fig = plt.gcf()
fig.set_size_inches(7,4)

# Draw lines
plt.plot(ddpg_time_steps, ddpg_con_IDM_m, color="#111111", label="DDPG")
plt.plot(sac_time_steps, sac_con_IDM_m, color="#FF3333", label="SAC")
plt.plot(trpo_time_steps, trpo_con_IDM_m, color="#004ec9", label="TRPO")

# Draw bands
plt.fill_between(ddpg_time_steps, ddpg_con_IDM_m - ddpg_con_IDM_s,
                 ddpg_con_IDM_m + ddpg_con_IDM_s, color="#DDDDDD", alpha=0.5)
plt.fill_between(sac_time_steps, sac_con_IDM_m - sac_con_IDM_s,
                 sac_con_IDM_m + sac_con_IDM_s, color="#FFBDBD", alpha=0.5)
plt.fill_between(trpo_time_steps, trpo_con_IDM_m - trpo_con_IDM_s,
                 trpo_con_IDM_m + trpo_con_IDM_s, color="#7997c6", alpha=0.5)

# Create plot
plt.xlabel("Training episode"), plt.ylabel("Average reward")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()