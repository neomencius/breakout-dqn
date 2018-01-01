import gym
import argparse

#import agent
from agents.dqn import DQN as Agent

#parser = argparse.ArgumentParser()

# create game
env = gym.make('CartPole-v0')
env.reset()

# create agent
agent = Agent(env.observation_space, env.action_space)

for episode in range(5000):
    state = env.reset()

    for t in range(100):
        env.render()


