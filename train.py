import gym
import argparse

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
agent = Agent(env, env.observation_space, env.action_space)

for episode in range(5000):
    state = env.reset()

    for t in range(100):
        if args['render']: # render it
            env.render()

        action = agent.getAction(state)
        next_state, reward, done, _ = env.step(action)
        agent.observe(state, action, next_state, reward, done)

        state = next_state
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
