import gym
import argparse
import numpy as np

#import agent
from agents.dqn import DQN as Agent

parser = argparse.ArgumentParser()
parser.add_argument('--render', dest='render', action='store_true')
parser.set_defaults(render=False)

args = vars(parser.parse_args())

# create game
env = gym.make('CartPole-v0')
env.reset()

# create agent
agent = Agent(env)


def preprocess(state):
    return np.reshape(state, [1, env.observation_space.shape[0]])

for e in range(350):
    state = env.reset()

    for t in range(200):
        if args['render']: # render it
            env.render()

        action = agent.getAction(preprocess(state))
        next_state, reward, done, _ = env.step(action)
        agent.observe(preprocess(state), action, preprocess(next_state), reward, done)

        state = next_state
        if done:
            break

    print("Episode {} finished after {} timesteps".format(e, t+1))

agent.save('dqn_agent')
