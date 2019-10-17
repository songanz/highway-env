from __future__ import division, print_function, absolute_import
import numpy as np
from gym import spaces

from highway_env import utils
from highway_env.envs.abstract import AbstractEnv
from highway_env.envs.highway_env_discrete import HighwayEnvDis
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.control import MDPVehicle
from highway_env.vehicle.control import ControlledVehicle

# target model
from stable_baselines.run import import_module, parse
import os


class HighwayEnvDisAdv(HighwayEnvDis):

    def __init__(self, config=None):
        super(HighwayEnvDisAdv, self).__init__(config)
        try:
            target_vehicle_model_path = os.path.abspath(os.getcwd()
                                                             + config['target_load_path'])
            alg_module = import_module('.'.join(['stable_baselines', config['target_model']]))
            policy = config['target_network']
            policy_kyw = {k: parse(v) for k,v in config['target_network_kyw'].items()}
            self.target_vehicle_model = getattr(alg_module, config['target_model'].upper())(
                policy, self, policy_kwargs=policy_kyw)
            self.target_vehicle_model.load(target_vehicle_model_path)
            self.tar_alg = config['target_model']
            self.alg = config["alg"]
            print("\t [INFO] target model loaded")
        except KeyError:
            print("\t [ERROR] Must give trained model path")
            exit()

    def _simulate(self, action=None):
        """
        Simulate with the target vehicle learned policy
        :param action:
        :return:
        """
        target_obs = self.observation.observe(vehicle=self.target_vehicle)
        target_act, _ = self.target_vehicle_model.predict(target_obs)
        print("\t [INFO] Target action: ", self.ACTIONS[target_act], "\t", "Action: ", self.ACTIONS[action])
        for k in range(int(self.SIMULATION_FREQUENCY // self.POLICY_FREQUENCY)):
            if action is not None and self.time % int(self.SIMULATION_FREQUENCY // self.POLICY_FREQUENCY) == 0:
                # Forward action to the vehicle
                self.vehicle.act(self.ACTIONS[action])
                self.target_vehicle.act(self.ACTIONS[target_act])

            self.road.act()
            self.road.step(1 / self.SIMULATION_FREQUENCY)
            self.time += 1
            if abs(self.vehicle.lane_distance_to(self.target_vehicle)) > 50:
                self.done = True

            # Automatically render intermediate simulation steps if a viewer has been launched
            self._automatic_rendering()

            # Stop at terminal states
            if self.done or self._is_terminal():
                break
        self.enable_auto_render = False

    def _create_vehicles(self):
        """
            Create some new random vehicles of a given type, and add them on the road.
        """
        self.vehicle = ControlledVehicle.create_random(self.road, spacing=self.config["initial_spacing"])
        # self.vehicle = MDPVehicle.create_random(self.road, spacing=self.config["initial_spacing"])
        self.road.vehicles.append(self.vehicle)

        # add target vehicle
        self.target_vehicle = ControlledVehicle.create_random(self.road, spacing=self.config["initial_spacing"])
        self.target_vehicle.color = (200, 0, 150)  # black
        self.road.vehicles.append(self.target_vehicle)

        vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])  # IDM from the config: can change
        for _ in range(self.config["vehicles_count"]):
            self.road.vehicles.append(vehicles_type.create_random(self.road, spacing=self.config["initial_spacing"]))

    def _reward(self, action):
        """
        Reward build for self.target_vehicle
        The reward is defined to foster driving at high speed, on the center of it's lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        lane_index = self.road.network.get_closest_lane_index(self.target_vehicle.position)
        lane_coords = self.road.network.get_lane(lane_index).local_coordinates(self.target_vehicle.position)
        lane_width = self.road.network.get_lane(lane_index).width
        lane_num = len(self.road.network.lanes_list())

        dy = lane_coords[1]  # distance to lane center
        x = self.target_vehicle.position[0]
        v = self.target_vehicle.velocity
        vx = v * self.target_vehicle.direction[0]
        vy = v * self.target_vehicle.direction[1]

        # get the front and rear vehicle to the target vehicle in the same lane
        front_veh, rear_veh = self.road.neighbour_vehicles(self.target_vehicle, lane_index)

        # run as quick as possible but not speeding
        rew_v = np.exp(-(vx - self.SPEED_MAX)**2/(2*2*(10*self.ACCELERATION_RANGE)**2))-1
        # in the center of lane
        rew_y = np.exp(-dy**2/(0.1*lane_width**2))-1

        state_reward = (rew_v + rew_y) / 2

        # crash for episode
        if self.target_vehicle.crashed:
            # off-policy algorithm
            if self.alg in ['dqn', 'sac', 'ddqg']:
                print('crash rw: %8.2f' % self.config["collision_reward"])
                return self.config["collision_reward"]
            else:
                print('crash rw: %8.2f' % (self.config["collision_reward"]*self.config["duration"]*self.POLICY_FREQUENCY))
                return self.config["collision_reward"]*self.config["duration"]*self.POLICY_FREQUENCY

        # outside road
        lane_bound_1 = (lane_num - 1) * lane_width + lane_width/2  # max y location in lane
        lane_bound_2 = 0 - lane_width/2  # min y location in lane

        if self.vehicle.position[1] > lane_bound_1 + lane_width/2 or\
                self.vehicle.position[1] < lane_bound_2 - lane_width/2:
            self.done = True
            print("vehicle_y: %8.2f;  rew_env: %8.2f"
                  % (self.vehicle.position[1],
                     self.config["collision_reward"] * self.config["duration"] * self.POLICY_FREQUENCY))
            return self.config["collision_reward"] * self.config["duration"] * self.POLICY_FREQUENCY

        if self.vehicle.position[1] > lane_bound_1:
            out_lane_punish = self.config["collision_reward"] * 2 * abs(self.vehicle.position[1]-lane_bound_1) - 5
            print("vehicle_y: %8.2f;  rew_env: %8.2f" % (self.vehicle.position[1], out_lane_punish))
            return out_lane_punish
        elif self.vehicle.position[1] < lane_bound_2:
            out_lane_punish = self.config["collision_reward"] * 2 * abs(self.vehicle.position[1] - lane_bound_2) - 5
            print("vehicle_y: %8.2f;  rew_env: %8.2f" % (self.vehicle.position[1], out_lane_punish))
            return out_lane_punish

        # running in the oppsite direction
        if vx < 0 or abs(vy/vx) > 1:
            velocity_heading_punish = self.config["collision_reward"]*self.config["duration"]*self.POLICY_FREQUENCY
            self.done = True
            print("speed: %8.2f;  rew_env: %8.2f" % (vx, velocity_heading_punish))
            return velocity_heading_punish

        # for debug
        # print('dx: %8.4f;  rew_x: %8.4f;  dy: %8.4f; rew_y: %8.4f;  vx: %8.4f;  rew_v: %8.4f' %
        #       (dx, rew_x, dy, rew_y, vx, rew_v))

        return state_reward
