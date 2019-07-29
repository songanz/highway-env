import highway_env
import gym

# env = gym.make("highway-continuous-v0")
env = gym.make("merge-v0")

done = False
while not done:
    # action = {'steering': 0, 'acceleration': 1}  # Your agent code here
    action = 0  # Your agent code here
    # action = [-1,0]  # "acceleration": action[0], "steering": action[1]
    obs, reward, done, _ = env.step(action)
    env.render()
