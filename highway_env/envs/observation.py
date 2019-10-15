from __future__ import division, print_function, absolute_import
import pandas
from gym import spaces
import numpy as np

from highway_env import utils
from highway_env.envs.finite_mdp import compute_ttc_grid
from highway_env.road.lane import AbstractLane
from highway_env.vehicle.control import MDPVehicle


class ObservationType(object):
    def space(self):
        raise NotImplementedError()

    def observe(self):
        raise NotImplementedError()


class TimeToCollisionObservation(ObservationType):
    def __init__(self, env, horizon=10, **kwargs):
        self.env = env
        self.horizon = horizon

    def space(self):
        try:
            return spaces.Box(shape=self.observe().shape, low=0, high=1, dtype=np.float32)
        except AttributeError:
            return None

    def observe(self):
        grid = compute_ttc_grid(self.env, time_quantization=1/self.env.POLICY_FREQUENCY, horizon=self.horizon)
        padding = np.ones(np.shape(grid))
        padded_grid = np.concatenate([padding, grid, padding], axis=1)
        obs_lanes = 3
        l0 = grid.shape[1] + self.env.vehicle.lane_index[2] - obs_lanes // 2
        lf = grid.shape[1] + self.env.vehicle.lane_index[2] + obs_lanes // 2
        clamped_grid = padded_grid[:, l0:lf+1, :]
        repeats = np.ones(clamped_grid.shape[0])
        repeats[np.array([0, -1])] += clamped_grid.shape[0]
        padded_grid = np.repeat(clamped_grid, repeats.astype(int), axis=0)
        obs_velocities = 3
        v0 = grid.shape[0] + self.env.vehicle.velocity_index - obs_velocities // 2
        vf = grid.shape[0] + self.env.vehicle.velocity_index + obs_velocities // 2
        clamped_grid = padded_grid[v0:vf + 1, :, :]
        return clamped_grid


class KinematicObservation(ObservationType):
    """
        Observe the kinematics of nearby vehicles.
    """
    FEATURES = ['presence', 'x', 'y', 'vx', 'vy']

    def __init__(self, env, features=FEATURES, vehicles_count=5, **kwargs):
        """
        :param env: The environment to observe
        :param features: Names of features used in the observation
        :param vehicles_count: Number of observed vehicles
        """
        self.env = env
        try:
            self.features = kwargs["FEATURES"]
        except KeyError:
            self.features = features
        self.vehicles_count = vehicles_count

    def space(self):
        return spaces.Box(shape=(len(self.features) * self.vehicles_count,), low=-1, high=1, dtype=np.float32)

    def normalize(self, df):
        """
            Normalize the observation values.

            For now, assume that the road is straight along the x axis.
        :param Dataframe df: observation data
        """
        side_lanes = self.env.road.network.all_side_lanes(self.env.vehicle.lane_index)
        x_position_range = 3.0 * MDPVehicle.SPEED_MAX
        y_position_range = AbstractLane.DEFAULT_WIDTH * len(side_lanes)
        y_position_range_0 = 0 - AbstractLane.DEFAULT_WIDTH
        velocity_range = MDPVehicle.SPEED_MAX
        vy_range = 0.4 * MDPVehicle.SPEED_MAX
        df['x'] = utils.remap(df['x'], [-x_position_range, x_position_range], [-1, 1])
        df['y'] = utils.remap(df['y'], [y_position_range_0, y_position_range], [-1, 1])
        df['vx'] = utils.remap(df['vx'], [-velocity_range, velocity_range], [-1, 1])
        df['vy'] = utils.remap(df['vy'], [-vy_range, vy_range], [-1, 1])
        return df

    def reverse_normalize(self, obs):
        # invers the flatten
        obs = obs.reshape(-1,4)

        side_lanes = self.env.road.network.all_side_lanes(self.env.vehicle.lane_index)
        x_position_range = 3.0 * MDPVehicle.SPEED_MAX
        y_position_range = AbstractLane.DEFAULT_WIDTH * len(side_lanes)
        y_position_range_0 = 0 - AbstractLane.DEFAULT_WIDTH
        velocity_range = MDPVehicle.SPEED_MAX
        vy_range = 0.4 * MDPVehicle.SPEED_MAX

        df = pandas.DataFrame({'x':obs[:,0],'y':obs[:,1],'vx':obs[:,2],'vy':obs[:,3]})

        df['x'] = utils.remap(df['x'], [-1, 1], [-x_position_range, x_position_range])
        df['y'] = utils.remap(df['y'], [-1, 1], [y_position_range_0, y_position_range])
        df['vx'] = utils.remap(df['vx'], [-1, 1], [-velocity_range, velocity_range])
        df['vy'] = utils.remap(df['vy'], [-1, 1], [-vy_range, vy_range])

        return df  # true state


    def observe(self, vehicle=None):
        if not vehicle:
            vehicle = self.env.vehicle
        # Add ego-vehicle
        df = pandas.DataFrame.from_records([vehicle.to_dict()])[self.features]
        df['x'][0] = 0  # see explanation in below triple-quoted strings
        # Add nearby traffic
        close_vehicles = self.env.road.closest_vehicles_to(vehicle, self.vehicles_count - 1)
        if close_vehicles:
            '''
            Given the vehicle to v.to_dict() will return relative x.
            Therefore, no need to carry the absolut position
            '''
            df = df.append(pandas.DataFrame.from_records(
                [v.to_dict(vehicle)
                 for v in close_vehicles[-self.vehicles_count + 1:]])[self.features],
                           ignore_index=True)
        # Normalize
        df = self.normalize(df)
        # Fill missing rows
        if df.shape[0] < self.vehicles_count:
            rows = -np.ones((self.vehicles_count - df.shape[0], len(self.features)))
            df = df.append(pandas.DataFrame(data=rows, columns=self.features), ignore_index=True)
        # Reorder
        df = df[self.features]
        # Clip
        obs = np.clip(df.values, -1, 1)
        # Flatten
        obs = np.ravel(obs)  # x, y, vx, vy

        return obs


class KinematicsGoalObservation(KinematicObservation):
    def __init__(self, env, scale, **kwargs):
        self.scale = scale
        super(KinematicsGoalObservation, self).__init__(env, **kwargs)

    def space(self):
        try:
            obs = self.observe()
            return spaces.Dict(dict(
                desired_goal=spaces.Box(-np.inf, np.inf, shape=obs["desired_goal"].shape, dtype=np.float32),
                achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype=np.float32),
                observation=spaces.Box(-np.inf, np.inf, shape=obs["observation"].shape, dtype=np.float32),
            ))
        except AttributeError:
            return None

    def observe(self):
        obs = np.ravel(pandas.DataFrame.from_records([self.env.vehicle.to_dict()])[self.features])
        goal = np.ravel(pandas.DataFrame.from_records([self.env.goal.to_dict()])[self.features])
        obs = {
            "observation": obs / self.scale,
            "achieved_goal": obs / self.scale,
            "desired_goal": goal / self.scale
        }
        return obs


def observation_factory(env, config):
    if config["type"] == "TimeToCollision":
        return TimeToCollisionObservation(env, **config)
    elif config["type"] == "Kinematics":
        return KinematicObservation(env, **config)
    elif config["type"] == "KinematicsGoal":
        return KinematicsGoalObservation(env, **config)
    else:
        raise ValueError("Unkown observation type")
