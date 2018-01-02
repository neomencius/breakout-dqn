# agent super class

class Agent:
    def __init__(self, state_space, action_space):
        raise NotImplementedError()

    def observe(self, state, action, nextState, reward, done):
        raise NotImplementedError()

    def get_action(self, state):
        raise NotImplementedError()


    def save(self, name):
        raise NotImplementedError()

    def load(self, name):
        raise NotImplementedError()

