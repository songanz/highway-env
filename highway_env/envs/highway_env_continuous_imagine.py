from __future__ import division, print_function, absolute_import
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from baselines.surprise_based.net_CVAE import *
from highway_env.envs.highway_env_continuous import HighwayEnvCon


class HighwayEnvCon_imagine(HighwayEnvCon):
    """Step part is different, will build imagination rollouts"""
    def __init__(self, config=None):  # create CVAE model
        self.device = 'cpu'
        self.terminal_num = 0
        if not config:
            super(HighwayEnvCon_imagine, self).__init__()
        else:
            try:
                self.CVAE_path = os.path.abspath(os.getcwd()+config.pop('CVAE_path'))
            except KeyError:
                cwd = os.getcwd()
                CVAEdir = os.path.abspath(cwd + '/models/CVAE/')
                filename = 'Environment_model_' + self.other_vehicles_type + self.spacing + '_00.pth.tar'
                self.CVAE_path = os.path.join(CVAEdir, filename)
            super(HighwayEnvCon_imagine, self).__init__(config)
        state_size = self.observation_space.shape[0]
        action_size = self.action_space.shape[0]
        latent_size = (self.observation.vehicles_count-1)*action_size
        condi_size = state_size + action_size
        self.env_model = CVAE(name='Environment_model' + self.other_vehicles_type + self.spacing,
                              state_size=state_size,
                              action_size=action_size,
                              latent_size=latent_size,
                              condi_size=condi_size)
        self.Buf = Memory(MemLen=int(1e5))

        # for imagination rollouts
        self.load_CVAE()
        self.imagine = True
        self.imagine_step = 64
        self.im_counter = 0
        self.im_old_ob = self.observation.observe()
        self.vpred_im_mc = 0
        self.im_path_num = 0
        self.im_length = 16
        self.im_ep = 5
        # todo delete debug plot
        self.im = []

    def step(self, action, fear=False):
        gamma = 0.99
        while self.im_path_num < self.im_ep:
            while self.imagine:
                if self.im_counter == 0:
                    self.im_old_ob = self.observation.observe()
                    self.vpred_im_mc = 0
                    # todo delete debug plot
                    self.im = [self.observation.reverse_normalize(self.im_old_ob)]

                if self.env_model.done or self.im_counter > self.im_length:
                    self.env_model.done = False
                    self.im_counter = 0
                    self.im_path_num += 1
                    break

                ob_next_im, rew_im, _= self.imagine_(action, self.im_old_ob)

                self.vpred_im_mc += np.power(gamma, self.im_counter)*rew_im

                self.im_counter += 1
                self.im_old_ob = np.squeeze(ob_next_im)

                if self.im_counter > self.im_length:
                    self.env_model.done = True

                info = {'imagine': self.im_counter,
                        'vpred_im_mc': self.vpred_im_mc,
                        'imagine_done': self.env_model.done}

                # todo delete debug plot
                # self.im.append(self.observation.reverse_normalize(self.im_old_ob))
                # self.check_im(self.im)

                return ob_next_im, rew_im, False, info

        self.im_path_num = 0
        old_s = self.observation.observe()

        obs, rew_env, terminal, _ = super(HighwayEnvCon_imagine, self).step(action)

        if terminal:
            self.terminal_num += 1

        if self.terminal_num % 20 == 1 and len(self.Buf.memory) > 256:
            self.update_env_model()
            self.terminal_num += 1

        # update buffer
        self.Buf.remember(old_s, action, rew_env, obs, terminal)

        # print("state_reward: %8.4f;  rew_env: %8.4f;  rew_i: %8.4f" % (state_reward, rew_env, rew_intrinsic))
        info = {'r_e': rew_env, "r_i": 0}

        return obs, rew_env, terminal, info

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
                = getSAT(self.Buf, self.device, num_CVAE=int(256))
            for w in range(5):
                loss, MSE, KLD = self.env_model.train_step(a_array, s_array, nextS_array, self.device)
                if k == 19 and w == 4:
                    print("CVAE finish training, loss: %8.4f;  MSE: %8.4f;  KLD: %8.4f"
                          % (loss, MSE, KLD))
                    print('Memory: %8.2d' % len(self.Buf.memory))
        cwd = os.getcwd()  # get father folder of the scripts folder
        CVAEdir = os.path.abspath(cwd + '/models/Imagine_env_model/')
        filename = self.env_model.name + '.pth.tar'
        pathname = os.path.join(CVAEdir, filename)
        tr.save({
                'state_dict': self.env_model.state_dict(),
                'optimizer': self.env_model.opt.state_dict(),
        }, pathname)

    def imagine_(self, ac, obs):
        imagine_env_rew, imagine_next_state = self.env_model.sample_next_s(ac, obs, self)
        return imagine_next_state, imagine_env_rew, self.env_model.done

    def load_CVAE(self):
        ckpt = tr.load(self.CVAE_path)
        self.env_model.load_state_dict(ckpt['state_dict'])
        self.env_model.opt.load_state_dict(ckpt['optimizer'])

    # todo check imagination path by visualization
    def check_im(self, im):
        ori_obs = im[0]
        # Use absolute x position for better visulization
        x = np.array([im[i]['x'][:] + i/self.POLICY_FREQUENCY*im[i]['vx'][0] for i in range(len(im))])
        y = np.array([im[i]['y'][:] for i in range(len(im))])
        numCar = ori_obs.shape[0]
        l = self.vehicle.LENGTH
        w = self.vehicle.WIDTH
        lane_index = self.road.network.get_closest_lane_index(self.vehicle.position)
        lane_width = self.road.network.get_lane(lane_index).width
        lane_num = len(self.road.network.lanes_list())
        st = 0 - lane_width/2

        plt.cla()  # clear current axis
        ax = plt.gca()
        ax.set_xlim(-100, 100)
        ax.set_ylim(-2.3, 12.3)
        ax.set_aspect('equal')

        # plot lane boundary
        plt.plot([-100, 100], [st, st], "-k")  # black: lane boundary
        for j in range(1, lane_num+1):
            plt.plot([-100, 100], [st+j*lane_width, st+j*lane_width], "-k")

        for i in range(numCar):
            if i == 0:
                car = mpatches.Rectangle((ori_obs['x'][i] - l/2, ori_obs['y'][i] - w/2), l, w)
                car.set_fill(True)
            else:
                car = mpatches.Rectangle((ori_obs['x'][i] - l/2, ori_obs['y'][i] - w/2), l, w)
                car.set_fill(False)
            ax.add_patch(car)
            plt.plot(x[:,i], y[:,i])
        plt.grid(False)
        plt.pause(0.001)
