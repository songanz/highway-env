from __future__ import division, print_function, absolute_import
import numpy as np
from gym import spaces

from highway_env import utils
from highway_env.envs.abstract import AbstractEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.dynamics import Vehicle


class HighwayEnvCon(AbstractEnv):
    """
        A highway driving environment.

        The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high velocity,
        staying on the rightmost lanes and avoiding collisions.
    """

    """ The reward received when colliding with a vehicle."""
    COLLISION_REWARD = -2

    """
        The frequency at which the system dynamics are simulated [Hz]
    """
    SIMULATION_FREQUENCY = 20  # todo: change to 20 and try
    """
        The frequency at which the agent can take actions [Hz]
    """
    POLICY_FREQUENCY = 10

    STEERING_RANGE = np.pi / 4
    ACCELERATION_RANGE = 5.0

    SPEED_MAX = 30  # m/s
    NO_COLI_TIME = 2  # at least for 2 seconds there wont be any collision
    LEN_SCL = 1.5  # at least this times length of car is minimum gap between cars
    VEL_SCL = 1.5  # this can be used for evaluate rear safety
    NOM_DIST = 10  # for keep safe distance parameter
    SAFE_FACTOR = 1.5  # for keep safe distance parameter
    M_ACL_DIST = 100


    # for update config in abstract
    DEFAULT_CONFIG = {
        "observation": {
            "type": "Kinematics",
            "FEATURES": ['x', 'y', 'vx', 'vy'],
            # "FEATURES": ['presence'， 'x', 'y', 'vx', 'vy'],  # dimenstion too high
            "vehicles_count": 7
        },
        "initial_spacing": 2,
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        # "other_vehicles_type": "highway_env.vehicle.behavior.AggressiveVehicle",
        # "other_vehicles_type": "highway_env.vehicle.behavior.DefensiveVehicle",
        "screen_width": 600,
        "screen_height": 150,
        "centering_position": [0.3, 0.5],
        "collision_reward": COLLISION_REWARD
    }

    DIFFICULTY_LEVELS = {
        "EASY": {
            "lanes_count": 2,
            "vehicles_count": 5,
            "duration": 20
        },
        "MEDIUM": {
            "lanes_count": 3,
            "vehicles_count": 20,
            "duration": 30
        },
        "HARD": {
            "lanes_count": 4,
            "vehicles_count": 50,
            "duration": 40
        },
    }

    def __init__(self, config=None):
        if not config:
            config = self.DEFAULT_CONFIG.copy()
        config.update(self.DIFFICULTY_LEVELS["MEDIUM"])
        super(HighwayEnvCon, self).__init__(config)
        self.reset()
        self.action_space = spaces.Box(-1., 1., shape=(2,), dtype=np.float32)
        self.steps = 0

    def reset(self):
        self._create_road()
        self._create_vehicles()
        self.steps = 0
        return super(HighwayEnvCon, self).reset()

    def step(self, action):

        self.steps += 1
        return super(HighwayEnvCon, self).step(action)

    def _simulate(self, action=None):
        """
            Perform several steps of simulation with constant action
        """
        for k in range(int(self.SIMULATION_FREQUENCY // self.POLICY_FREQUENCY)):
            if action is not None and self.time % int(self.SIMULATION_FREQUENCY // self.POLICY_FREQUENCY) == 0:
                # Forward action to the vehicle
                self.vehicle.act({
                    "acceleration": action[0] * self.ACCELERATION_RANGE,
                    "steering": action[1] * self.STEERING_RANGE
                })
                self.vehicle.target_velocity = \
                    self.vehicle.velocity + action[0] * (1 / self.SIMULATION_FREQUENCY)

            self.road.act()
            self.road.step(1 / self.SIMULATION_FREQUENCY)
            self.time += 1

            # Automatically render intermediate simulation steps if a viewer has been launched
            self._automatic_rendering()

            # Stop at terminal states
            if self.done or self._is_terminal():
                break
        self.enable_auto_render = False

    def _create_road(self):
        """
            Create a road composed of straight adjacent lanes.
        """
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"]),
                         np_random=self.np_random)

    def _create_vehicles(self):
        """
            Create some new random vehicles of a given type, and add them on the road.
        """
        self.vehicle = Vehicle.create_random(self.road, 25, spacing=self.config["initial_spacing"])
        self.road.vehicles.append(self.vehicle)

        vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])  # IDM from the config: can change
        for _ in range(self.config["vehicles_count"]):
            self.road.vehicles.append(vehicles_type.create_random(self.road))

    def _reward(self, action):
        """
        The reward is defined to foster driving at high speed, on the center of it's lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        lane_index = self.road.network.get_closest_lane_index(self.vehicle.position)
        lane_coords = self.road.network.get_lane(lane_index).local_coordinates(self.vehicle.position)
        lane_width = self.road.network.get_lane(lane_index).width
        lane_num = len(self.road.network.lanes_list())

        dy = lane_coords[1]  # distance to lane center
        x = self.vehicle.position[0]
        v = self.vehicle.velocity
        vx = v * self.vehicle.direction[0]

        front_veh, rear_veh = self.road.neighbour_vehicles(self.vehicle, lane_index)  # get the front and rear vehicle in the same lane
        try:
            front_veh_vx = front_veh.velocity * front_veh.direction[0]
            dx = front_veh.position[0] - x
        except AttributeError:
            front_veh_vx = self.SPEED_MAX
            dx = self.M_ACL_DIST

        sfDist = (self.NOM_DIST * self.LEN_SCL) + (vx - front_veh_vx) * self.NO_COLI_TIME  # calculate safe distance
        # midDis = sfDist + (self.M_ACL_DIST - sfDist) / self.LEN_SCL

        # keep safe distance
        rew_x = 0
        if dx < sfDist * self.SAFE_FACTOR:
            # print('dx: %8.4f;  sfDist: %8.4f' % (dx, sfDist))
            rew_x = np.exp(-(dx - sfDist*self.SAFE_FACTOR)**2/(2*self.NOM_DIST**2))-1
        # run as quick as possible but not speeding
        rew_v = np.exp(-(v - self.SPEED_MAX)**2/(2*2*self.ACCELERATION_RANGE**2))-1
        # in the center of lane
        rew_y = np.exp(-dy**2/(2*lane_width**2))-1

        state_reward = (rew_v + rew_y + rew_x) / 3

        # crash for episode
        if self.vehicle.crashed:
            print('crash rw: %8.2f' % (self.config["collision_reward"]*self.config["duration"]*self.POLICY_FREQUENCY))
            return self.config["collision_reward"]*self.config["duration"]*self.POLICY_FREQUENCY

        # outside road
        lane_bound_1 = (lane_num - 1) * lane_width + lane_width/2  # max y location in lane
        lane_bound_2 = 0 - lane_width/2  # min y location in lane
        if self.vehicle.position[1] > lane_bound_1:
            out_lane_punish = self.config["collision_reward"] * 2 * abs(self.vehicle.position[1]-lane_bound_1)
            print("vehicle_y: %8.4f;  rw: %8.2f" % (self.vehicle.position[1], out_lane_punish))
            return out_lane_punish
        elif self.vehicle.position[1] < lane_bound_2:
            out_lane_punish = self.config["collision_reward"] * 2 * abs(self.vehicle.position[1] - lane_bound_2)
            print("vehicle_y: %8.4f;  rw: %8.2f" % (self.vehicle.position[1], out_lane_punish))
            return out_lane_punish

        # running in the oppsite direction
        if vx < 0:
            print("speed: %8.4f;  rw: %8.2f" % (vx, self.config["collision_reward"]*abs(vx)**2))
            return self.config["collision_reward"]*abs(vx)**2

        # for debug
        print('rw: %8.4f;  rew_x: %8.4f;  rew_y: %8.4f;  rew_v: %8.4f' % (state_reward, rew_x, rew_y, rew_v))

        return state_reward

    def _is_terminal(self):
        """
            The episode is over if the ego vehicle crashed or the time is out.
        """
        return self.vehicle.crashed or self.steps/self.POLICY_FREQUENCY >= self.config["duration"]

    def _cost(self, action):
        """
            The cost signal is the occurrence of collision
        """
        return float(self.vehicle.crashed)
