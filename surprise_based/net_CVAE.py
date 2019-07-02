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
            x = tr.cat((x, c), dim=0)

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
            x = tr.cat((z, c), dim=0)

        try:
            x = nn.LeakyReLU(self.alpha)(self.bn1(self.fc1(x)))
            x = nn.LeakyReLU(self.alpha)(self.bn2(self.fc2(x)))
        except ValueError:
            x = nn.LeakyReLU(self.alpha)(self.bn1(self.fc1(x[None])))
            x = nn.LeakyReLU(self.alpha)(self.bn2(self.fc2(x)))

        x = nn.Tanh()(self.output(x))

        return x


class CVAE(nn.Module):  # in our case, condi_size should be state_size + action_size

    ETA = 1000

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
        MSE = nn.MSELoss(size_average=True)(recon_next_state, nexs_state)
        KL_D = -0.5 * tr.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return MSE + KL_D, MSE, KL_D

    def sample_x(self, ego_action_, state_):
        c = Variable(tr.from_numpy(np.hstack((state_, ego_action_[:, np.newaxis]))), requires_grad=False).float()
        batch_size = c.shape[0]
        z = Variable(tr.randn((batch_size, self.latent_size)), requires_grad=False)
        recon_x = self.decoder(z, c)
        next_state = self.dynamic(recon_x, c)

        return next_state

    def bonus_reward_each_state(self, ego_action_, state_, next_state_, rewards, device):
        self.encoder.eval()
        self.decoder.eval()

        x = Variable(tr.from_numpy(next_state_).to(device), requires_grad=False).float()
        c = Variable(tr.from_numpy(np.hstack((state_, ego_action_))).to(device), requires_grad=False).float()
        z_means,log_var = self.encoder(x,c)

        z_means_numpy = z_means.cpu().detach().numpy()
        standar_normal = {"mean": np.zeros(z_means_numpy.shape),
                          "log_std": np.zeros(z_means_numpy.shape)}
        KL_D = log_likelihood(z_means_numpy, standar_normal) - log_likelihood(np.zeros(z_means_numpy.shape), standar_normal)
        # KL_D = -0.5 * tr.sum(1 + log_var - z_means.pow(2) - log_var.exp())
        # KL_D = KL_D.cpu().detach().numpy()
        eta = self.eta/max((1, np.mean(rewards)))
        bonus = -KL_D*eta
        return bonus, KL_D, z_means_numpy

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