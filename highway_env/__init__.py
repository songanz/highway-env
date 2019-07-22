from gym.envs.registration import register

register(
    id='highway-v0',
    entry_point='highway_env.envs:HighwayEnv',
)

register(
    id='highway-continuous-v0',
    entry_point='highway_env.envs:HighwayEnvCon',
)

register(
    id='highway-continuous-intrinsic-rew-v0',
    entry_point='highway_env.envs:HighwayEnvCon_intrinsic_rew',
)

register(
    id='highway-continuous-imagine-v0',
    entry_point='highway_env.envs:HighwayEnvCon_imagine',
)

register(
    id='highway-discrete-v0',
    entry_point='highway_env.envs:HighwayEnvDis',
)

register(
    id='highway-discrete-intrinsic-rew-v0',
    entry_point='highway_env.envs:HighwayEnvDis_intrinsic_rew',
)

register(
    id='highway-discrete-imagine-v0',
    entry_point='highway_env.envs:HighwayEnvDis_imagine',
)

register(
    id='merge-v0',
    entry_point='highway_env.envs:MergeEnv',
)

register(
    id='roundabout-v0',
    entry_point='highway_env.envs:RoundaboutEnv',
)

register(
    id='two-way-v0',
    entry_point='highway_env.envs:TwoWayEnv',
    max_episode_steps=15
)

register(
    id='parking-v0',
    entry_point='highway_env.envs:ParkingEnv',
    max_episode_steps=20
)
