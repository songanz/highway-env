import torch as tr
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from collections import deque
import random

def log_likelihood(xs, dist_info):
    means = dist_info["mean"]
    log_stds = dist_info["log_std"]
    zs = (xs - means) / np.exp(log_stds)
    return - np.sum(log_stds, axis=-1) - \
           0.5 * np.sum(np.square(zs), axis=-1) - \
           0.5 * means.shape[-1] * np.log(2 * np.pi)

class Memory:
    def __init__(self, MemLen=10000):
        self.memory = deque(maxlen=MemLen)

    def getMiniBatch(self, batch_size=16):
        if batch_size < len(self.memory):
            return random.sample(self.memory, batch_size)
        else:
            return random.sample(self.memory, len(self.memory))

    def remember(self, state, action, reward, next_state, coli):
        self.memory.append((state, action, reward, next_state, coli))

def getSAT(Buf, device, num_CVAE=0):
    ST_IDX = 0
    ACT_IDX = 1
    REW_IDX = 2
    NXTST_IDX = 3
    DN_IDX = 4
    # there is enough data in memory hence
    if not num_CVAE == 0:
        Batch = Buf.getMiniBatch(batch_size=num_CVAE)
    else:
        Batch = Buf.getMiniBatch(batch_size=16)

    miniBatch = Batch

    stateBuf_array = np.array([each[ST_IDX] for each in miniBatch])
    actionBuf_array = np.array([each[ACT_IDX] for each in miniBatch])
    rewards_array = np.array([each[REW_IDX] for each in miniBatch])
    nextStateBuf_array = np.array([each[NXTST_IDX] for each in miniBatch])
    doneBuf_array = np.array([not each[DN_IDX] for each in miniBatch], dtype=int)

    stateBuf = Variable(tr.from_numpy(stateBuf_array).to(device), requires_grad=False).float()
    actionBuf = Variable(tr.from_numpy(actionBuf_array.astype(int)).to(device), requires_grad=False).float()
    rewards = Variable(tr.from_numpy(rewards_array).to(device), requires_grad=False).float()
    nextStateBuf = Variable(tr.from_numpy(nextStateBuf_array).to(device), requires_grad=False).float()
    doneBuf = Variable(tr.from_numpy(doneBuf_array).to(device), requires_grad=False).float()

    return stateBuf, actionBuf, rewards, nextStateBuf, doneBuf, \
           stateBuf_array, actionBuf_array, rewards_array, nextStateBuf_array, doneBuf_array

class Encoder(nn.Module):
    def __init__(self, hidden_size_IP=100, hidden_size_rest=100, alpha=0.01,
                 state_size=2, latent_size=6,
                 condi_size=2):
        super(Encoder, self). __init__()
        self.in_size = state_size + condi_size
        self.alpha = alpha
        self.fc1 = nn.Linear(self.in_size, hidden_size_IP)
        self.bn1 = nn.BatchNorm1d(hidden_size_IP)

        self.fc2 = nn.Linear(hidden_size_IP, hidden_size_rest)
        self.bn2 = nn.BatchNorm1d(hidden_size_rest)

        self.z_mean = nn.Linear(hidden_size_rest, latent_size)
        self.z_log_var = nn.Linear(hidden_size_rest, latent_size)

    def forward(self, x, c):
        try:
            x = tr.cat((x, c), dim=1)  # batch setup
        except (IndexError, RuntimeError):
            x = tr.cat((tr.squeeze(x), tr.squeeze(c)), dim=0)

        try:
            x = nn.LeakyReLU(self.alpha)(self.bn1(self.fc1(x)))
            x = nn.LeakyReLU(self.alpha)(self.bn2(self.fc2(x)))
        except ValueError:
            x = nn.LeakyReLU(self.alpha)(self.bn1(self.fc1(x[None])))
            x = nn.LeakyReLU(self.alpha)(self.bn2(self.fc2(x)))

        means = nn.Tanh()(self.z_mean(x))
        log_vars = nn.Tanh()(self.z_log_var(x))

        return means, log_vars


class Decoder(nn.Module):
    def __init__(self, hidden_size_IP=100, hidden_size_rest=100, alpha=0.01,
                 state_size=2, latent_size=5,
                 condi_size=2+2):
        super(Decoder, self). __init__()
        self.in_size = latent_size + condi_size
        self.alpha = alpha
        self.fc1 = nn.Linear(self.in_size, hidden_size_IP)
        self.bn1 = nn.BatchNorm1d(hidden_size_IP)

        self.fc2 = nn.Linear(hidden_size_IP, hidden_size_rest)
        self.bn2 = nn.BatchNorm1d(hidden_size_rest)

        self.output = nn.Linear(hidden_size_rest, state_size)

    def forward(self, z, c):
        try:
            x = tr.cat((z, c), dim=1)  # batch setup
        except (IndexError, RuntimeError):
            x = tr.cat((tr.squeeze(z), tr.squeeze(c)), dim=0)

        try:
            x = nn.LeakyReLU(self.alpha)(self.bn1(self.fc1(x)))
            x = nn.LeakyReLU(self.alpha)(self.bn2(self.fc2(x)))
        except ValueError:
            x = nn.LeakyReLU(self.alpha)(self.bn1(self.fc1(x[None])))
            x = nn.LeakyReLU(self.alpha)(self.bn2(self.fc2(x)))

        x = nn.Tanh()(self.output(x))

        return x


class CVAE(nn.Module):  # in our case, condi_size should be state_size + action_size

    ETA = 2

    def __init__(self, name='network', hidden_size_IP=100, hidden_size_rest=100, alpha=0.01,
                 state_size=2, action_size=2, learning_rate=1e-4,
                 latent_size=2, condi_size=2,
                 bonus_trade_off = ETA):
        super(CVAE, self).__init__()
        self.name = name
        self.alpha = alpha
        self.lr = learning_rate
        self.latent_size = latent_size
        self.condi_size = condi_size
        self.eta = bonus_trade_off

        self.encoder = Encoder(hidden_size_IP, hidden_size_rest, alpha, state_size, latent_size, condi_size)
        self.decoder = Decoder(hidden_size_IP, hidden_size_rest, alpha, state_size, latent_size, condi_size)
        self.opt = tr.optim.Adam(self.parameters(), lr=self.lr)
        self.done = False

    def forward(self, x, c, device):
        # x: input, c: condition
        # here x: next state, c: current state
        latent_mean, latent_log_var = self.encoder(x, c)
        std = tr.exp(0.5 * latent_log_var).to(device)
        eps = Variable(tr.randn(std.size()).to(device), requires_grad=False)

        z = latent_mean + eps*std
        next_state = self.decoder(z, c)

        return next_state, latent_mean, latent_log_var

    def loss_fun(self, mean, log_var, nexs_state, recon_next_state):
        MSE = nn.MSELoss(reduction='elementwise_mean')(recon_next_state, nexs_state)
        KL_D = -0.5 * tr.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return MSE + KL_D, MSE, KL_D

    def sample_next_s(self, ego_action_, state_, env):
        self.encoder.eval()
        self.decoder.eval()

        c = Variable(tr.from_numpy(np.hstack((state_, ego_action_))), requires_grad=False).float()
        c = tr.squeeze(c)
        if len(c.shape) == 1:
            batch_size = 1
        else:
            batch_size = c.shape[0]
        z = Variable(tr.randn((batch_size, self.latent_size)), requires_grad=False)
        next_state = self.decoder(z, c)
        next_state = next_state.data.numpy()
        rew = self.calculate_rew(env, state_)
        return rew, next_state

    # def bonus_reward_each_state(self, ego_action_, state_, next_state_, rewards, device):
    #     self.encoder.eval()
    #     self.decoder.eval()
    #
    #     x = Variable(tr.from_numpy(next_state_).to(device), requires_grad=False).float()
    #     c = Variable(tr.from_numpy(np.hstack((state_, ego_action_))).to(device), requires_grad=False).float()
    #     z_means,log_var = self.encoder(x,c)
    #
    #     z_means_numpy = z_means.cpu().detach().numpy()
    #     standar_normal = {"mean": np.zeros(z_means_numpy.shape),
    #                       "log_std": np.zeros(z_means_numpy.shape)}
    #     KL_D = log_likelihood(z_means_numpy, standar_normal) - log_likelihood(np.zeros(z_means_numpy.shape), standar_normal)
    #     # KL_D = -0.5 * tr.sum(1 + log_var - z_means.pow(2) - log_var.exp())
    #     # KL_D = KL_D.cpu().detach().numpy()
    #     eta = self.eta/max((1, np.mean(rewards)))
    #     bonus = -KL_D*eta
    #     return bonus, KL_D, z_means_numpy

    def bonus_reward_each_state(self, ego_action_, state_, next_state_, rewards, device):
        self.encoder.eval()
        self.decoder.eval()

        x = Variable(tr.from_numpy(next_state_).to(device), requires_grad=False).float()
        c = Variable(tr.from_numpy(np.hstack((state_, ego_action_))).to(device), requires_grad=False).float()

        recon_next_state, mean, log_var = self.forward(x, c, device)
        loss, _, KL_D = self.loss_fun(mean, log_var, x, tr.squeeze(recon_next_state))
        loss_for_bonus_rew = loss.data.numpy()*100

        eta = self.eta/max((1, np.mean(loss_for_bonus_rew)))

        bonus = loss_for_bonus_rew*eta
        return bonus, KL_D, mean

    def train_step(self, ego_action_, state_, next_state_, device):
        self.encoder.train()
        self.decoder.train()

        x = Variable(tr.from_numpy(next_state_).to(device), requires_grad=False).float()
        c = Variable(tr.from_numpy(np.hstack((state_, ego_action_))).to(device), requires_grad=False).float()

        recon_next_state, mean, log_var = self.forward(x, c, device)

        self.opt.zero_grad()
        loss, MSE, KL_D = self.loss_fun(mean, log_var, x, recon_next_state)
        loss.backward()
        self.opt.step()

        return loss, MSE, KL_D

    def calculate_rew(self, env, state_):
        true_state = env.observation.reverse_normalize(state_)  # dataframe

        y = true_state['y'][0]
        vx = true_state['vx'][0]
        vy = true_state['vy'][0]

        lane_index = env.road.network.get_closest_lane_index(env.vehicle.position)
        lane_width = env.road.network.get_lane(lane_index).width
        lane_num = len(env.road.network.lanes_list())
        lanes_center = [lane_width * i for i in range(lane_num)]

        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return idx, array[idx]

        def find_nearest_front(array, value):
            array = np.asarray(array)
            array -= value
            array = list(array)
            m = min(i for i in array if i > 0)
            return array.index(m), m + value

        def check_collision(df):
            p = np.array([df['x'][0], df['y'][0]])
            for i in range(1, df.shape[0]):
                other_p = np.array([df['x'][i], df['y'][i]])
                if np.linalg.norm(other_p - p) < np.linalg.norm([env.vehicle.LENGTH, env.vehicle.WIDTH]):
                    return True
            return False

        # find dy
        which_lane, lane_cen = find_nearest(lanes_center, y)
        dy = np.abs(lane_cen - y)

        # find dx
        temp_x = []
        temp_v = []
        for i in range(1, true_state.shape[0]):
            if find_nearest(lanes_center, true_state['y'][i])[0] == which_lane:
                temp_x.append(true_state['x'][i])
                temp_v.append(true_state['vx'][i])

        if not [i for i in temp_x if i > 0]:
            # no car in the same lane or no car in front in the same lane
            dx = env.M_ACL_DIST
            front_veh_vx = env.SPEED_MAX
        else:
            j, dx = find_nearest_front(temp_x, 0)  # becuase the position of host car is 0
            front_veh_vx = temp_v[j]

        # rew_x
        # keep safe distance
        sfDist = (env.NOM_DIST * env.LEN_SCL) + (vx - front_veh_vx) * env.NO_COLI_TIME  # calculate safe distance
        rew_x = 0
        if dx < sfDist * env.SAFE_FACTOR:
            rew_x = np.exp(-(dx - sfDist * env.SAFE_FACTOR) ** 2 / (2 * env.NOM_DIST ** 2)) - 1
        # rew_y
        # in the center of lane
        rew_y = np.exp(-dy ** 2 / (0.1 * lane_width ** 2)) - 1
        # rew_v
        # run as quick as possible but not speeding
        rew_v = np.exp(-(vx - env.SPEED_MAX) ** 2 / (2 * 2 * (6 * env.ACCELERATION_RANGE) ** 2)) - 1

        state_reward = (rew_v + rew_y + rew_x) / 3

        if check_collision(true_state):
            self.done = True
            return env.config["collision_reward"] * env.config["duration"] * env.POLICY_FREQUENCY

        # outside road
        lane_bound_1 = (lane_num - 1) * lane_width + lane_width / 2  # max y location in lane
        lane_bound_2 = 0 - lane_width / 2  # min y location in lane

        if y > lane_bound_1 + lane_width / 2 or y < lane_bound_2 - lane_width / 2:
            self.done = True
            return env.config["collision_reward"] * env.config["duration"] * env.POLICY_FREQUENCY

        if y > lane_bound_1:
            out_lane_punish = env.config["collision_reward"] * 2 * abs(y - lane_bound_1) - 5
            return out_lane_punish
        elif y < lane_bound_2:
            out_lane_punish = env.config["collision_reward"] * 2 * abs(y - lane_bound_2) - 5
            return out_lane_punish

        # running in the oppsite direction
        if vx < 0 or abs(vy / vx) > 1:
            self.done = True
            velocity_heading_punish = env.config["collision_reward"] * env.config["duration"] * env.POLICY_FREQUENCY
            return velocity_heading_punish

        return state_reward
