import gym
import numpy as np

#import agent
from agents.dqn import DQN as Agent

# create game
env = gym.make('CartPole-v0')
env.reset()

# create greedy agent
agent = Agent(env, epsilon=0.0, epsilon_min=0.0)
agent.load('dqn_agent.h5')

def preprocess(state):
    return np.reshape(state, [1, env.observation_space.shape[0]])

for e in range(10):
    state = env.reset()

    for t in range(200):
        env.render()

        action = agent.getAction(preprocess(state))
        next_state, _, done, _ = env.step(action)
        state = next_state
        if done:
            break

    print("Episode {} finished after {} timesteps".format(e, t+1))

