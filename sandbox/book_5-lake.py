import gym
import time
import torch

import matplotlib.pyplot as plt

from gym.envs.registration import register
register(
    id="FrozenLakeNotSlippery-v0",
    entry_point="gym.envs.toy_text:FrozenLakeEnv",
    kwargs={"map_name": "4x4", "is_slippery": False}
)

env = gym.make("FrozenLakeNotSlippery-v0")
#env = gym.make("FrozenLake-v0")

number_of_states = env.observation_space.n
number_of_actions = env.action_space.n

gamma = 1

max_episodes = 1000

Q = torch.zeros([number_of_states, number_of_actions])
C = torch.zeros([number_of_states, number_of_actions])
policy = torch.argmax(Q, dim=1)

for i in range(max_episodes):
    state = env.reset()






Q = torch.zeros([number_of_states, number_of_actions])
#print(Q)

num_episodes = 1000
steps_total = []
rewards_total = []

for i in range(num_episodes):
    state = env.reset()

    step = 0
    #for step in range(100):
    while True:
        step += 1

        random_values = Q[state] + torch.rand(1, number_of_actions) / 1000

        action = torch.max(random_values, 1)[1][0].item()
        #action = env.action_space.sample()

        new_state, reward, done, info = env.step(action)

        q = reward + gamma * torch.max(Q[new_state]).item()
        Q[state, action] = q

        state = new_state

        #time.sleep(0.4)

        #env.render()

        #print(new_state)
        #print(info)

        if done:
            steps_total.append(step)
            rewards_total.append(reward)
            #print("episode finished after {} steps".format(step))

            break

print(Q)

print("percent of episodes finished successfully: {0}".format(sum(rewards_total)/num_episodes))
print("percent of episodes finished successfully (last 100): {0}".format(sum(rewards_total[-100:])/100))
print("average number of steps: {}".format(sum(steps_total) / num_episodes))
print("average number of steps (last 100): {}".format(sum(steps_total[-100:]) / 100))