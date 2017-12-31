# agent super class

class Agent:
    def __init__(self, state_space, action_space):
        util.raiseNotDefined()

    def observe(self, state, action, nextState, reward, done):
        util.raiseNotDefined()

    def get_action(self, state):
        util.raiseNotDefined()
