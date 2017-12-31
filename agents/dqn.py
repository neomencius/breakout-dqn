from agent import Agent

class DQN(Agent):
    def __init__(self, state_space, action_space):
        self.state_size = state_space.shape[0]
        self.action_size = action_space.n

        self.action_space = action_space

    def observe(self, state, action, next_state, reward, done):
        return

    def getAction(self, state):
        return self.action_space.sample() # random action
