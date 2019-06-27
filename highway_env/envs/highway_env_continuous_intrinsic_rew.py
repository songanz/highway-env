from __future__ import division, print_function, absolute_import
import numpy as np
from gym import spaces

from highway_env import utils
from highway_env.envs.highway_env_continuous import HighwayEnvCon
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.dynamics import Vehicle
from surprise_based.net_CVAE import *


class HighwayEnvCon_intrinsic_rew(HighwayEnvCon):
    """Only reward is different, it will consider intrinsic reward"""
    def __init__(self, config=None):  # create CVAE model
        self.device = 'cpu'
        self.terminal_num = 0
        if not config:
            super(HighwayEnvCon_intrinsic_rew, self).__init__()
        else:
            super(HighwayEnvCon_intrinsic_rew, self).__init__(config)
        state_size = self.observation_space.shape[0]
        action_size = self.action_space.shape[0]
        latent_size = (self.observation.vehicles_count-1)*action_size
        condi_size = state_size + action_size
        self.env_model = CVAE(name='Environment_model',
                              state_size=state_size,
                              action_size=action_size,
                              latent_size=latent_size,
                              condi_size=condi_size)
        self.Buf = Memory(MemLen=int(1e5))

    def step(self, action, fear=False):
        old_s = self.observation.observe()
        obs, rew_env, terminal, _ = super(HighwayEnvCon_intrinsic_rew, self).step(action)

        if terminal:
            self.terminal_num += 1

        if self.terminal_num % 20 == 1 and len(self.Buf.memory) > 16:
            self.update_env_model()
            self.terminal_num += 1
        # calculate intrinsic reward
        rew_intrinsic = self.intrinsic_rew(old_s, obs, action, rew_env)
        # update buffer
        self.Buf.remember(old_s, action, rew_env, obs, terminal)

        if fear:  # for runing old policy in new environment
            state_reward = rew_env - rew_intrinsic
        else:  # for exploration
            state_reward = rew_env + rew_intrinsic
        print("state_reward: %8.4f;  rew_env: %8.4f;  rew_i: %8.4f" % (state_reward, rew_env, rew_intrinsic))

        info = {'r_e': rew_env, "r_i": rew_intrinsic}

        return obs, state_reward, terminal, info

    def intrinsic_rew(self, old_s, s, a, r):
        """
        Calculate intrinsic reward
        :param old_s: current state
        :param s: state
        :param a: host vehicle action correspond to old_s
        :param r: environment reward
        :return: rew_i: intrinsic reward
        """
        rew_i, _, _ = self.env_model.bonus_reward_each_state(a, old_s, s, r, self.device)
        return rew_i

    def update_env_model(self):
        for k in range(20):
            _, _, _, _, _, s_array, a_array, r_array, nextS_array, doneBuf_array \
                = getSAT(self.Buf, self.device, num_CVAE=int(32))
            for w in range(5):
                loss, MSE, KLD = self.env_model.train_step(a_array, s_array, nextS_array, self.device)
                if k == 19 and w == 4:
                    print("CVAE finish training, loss: %8.4f;  MSE: %8.4f;  KLD: %8.4f"
                          % (loss, MSE, KLD))
