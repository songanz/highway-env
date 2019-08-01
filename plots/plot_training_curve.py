import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


total_ep = 300

episode = [i for i in range(total_ep)]  # x axis
surprise_dis_IDM = []
baseline_dis_IDM = []

# load data
for i in range(2):
    surprise_dis_IDM_path = os.path.abspath(os.getcwd() + '/models/Surprise_dis/0' + str(i) + '/latest_log/progress.csv')
    baseline_dis_IDM_path = os.path.abspath(os.getcwd() + '/models/baseline_dis/0' + str(i) + '/latest_log/progress.csv')
    surprise_dis_IDM.append(np.array(pd.read_csv(surprise_dis_IDM_path)['EpRew/step'])[:total_ep])
    baseline_dis_IDM.append(np.array(pd.read_csv(baseline_dis_IDM_path)['EpRew/step'])[:total_ep])

baseline_dis_IDM = np.array(baseline_dis_IDM)
baseline_dis_IDM_m = np.mean(baseline_dis_IDM, axis=0)
baseline_dis_IDM_s = np.std(baseline_dis_IDM, axis=0)

surprise_dis_IDM = np.array(surprise_dis_IDM)
surprise_dis_IDM_m = np.mean(surprise_dis_IDM, axis=0)
surprise_dis_IDM_s = np.std(surprise_dis_IDM, axis=0)

# Average episode
# for i in range(total_ep):
#     surprise[i] = np.mean(surprise[i:i+50], axis=0)
#     baseline[i] = np.mean(baseline[i:i+50], axis=0)

fig = plt.gcf()
fig.set_size_inches(7,4)

# Draw lines
plt.plot(episode, surprise_dis_IDM_m, color="#111111", label="Model-based exploration")
plt.plot(episode, baseline_dis_IDM_m, color="#004ec9", label="Baseline TRPO")

# Draw bands
plt.fill_between(episode, surprise_dis_IDM_m - surprise_dis_IDM_s,
                 surprise_dis_IDM_m + surprise_dis_IDM_s, color="#DDDDDD", alpha=0.5)
plt.fill_between(episode, baseline_dis_IDM_m - baseline_dis_IDM_s,
                 baseline_dis_IDM_m + baseline_dis_IDM_s, color="#7997c6", alpha=0.5)

# Create plot
plt.xlabel("Training episode"), plt.ylabel("Average reward")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()