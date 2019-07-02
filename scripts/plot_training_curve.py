import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


path = 'C:/Users/szhan117/Documents/git_repo/highway-env/models/'
total_ep = 3000

episode = [i for i in range(total_ep)]  # x axis

# data
intr_rew = sio.loadmat(path + 'Surprise/trainHist.mat')['intrinsic_reward'][0, :total_ep+100]
surprise = sio.loadmat(path + 'Surprise/trainHist.mat')['mean_reward'][0, :total_ep+100]
baseline = sio.loadmat(path + 'trainHist.mat')['mean_reward'][0, :total_ep+100]

for i in range(total_ep):
    surprise[i] = np.mean(surprise[i:i+50], axis=0)
    baseline[i] = np.mean(baseline[i:i+50], axis=0)


fig = plt.gcf()
fig.set_size_inches(7,4)
# Draw lines
plt.plot(episode, surprise[:total_ep], color="#111111", label="Model-based exploration")
plt.plot(episode, baseline[:total_ep], color="#004ec9", label="TRPO")

# Create plot
plt.xlabel("Training episode"), plt.ylabel("Average reward")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()