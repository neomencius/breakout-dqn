import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from agent import Agent

class DQN(Agent):
    def __init__(self, env, discount=0.9, epsilon=1.0, epsilon_min=0.01, epsilon_log_decay=0.9995, alpha=0.01, alpha_decay=0.01, batch_size=256):
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.env = env

        self.memory = deque(maxlen=8000)
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_log_decay
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.batch_size = batch_size

        self.model = self._build_model()
        self.model_prev = self._build_model()

        print 'state size:', self.state_size
        print 'action size:', self.action_size

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.01))
        return model

    def observe(self, state, action, next_state, reward, done):
        self.memory.append((state, action, next_state, reward, done))
        self.update()

        if done: # end of episode
            print "epsilon:", self.epsilon
            self.replay()

    def getAction(self, state):
        if (np.random.random() <= self.epsilon):
            return self.env.action_space.sample()
        else:
            return np.argmax(self.model.predict(state))

    def update(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def replay(self):
        x_batch, y_batch = [], []
        sample_size = min(len(self.memory), self.batch_size)
        minibatch = random.sample(self.memory, sample_size)

        for state, action, next_state, reward, done in minibatch:
            y_target = self.model.predict(state)
            y_target[0][action] = reward if done else reward + self.discount * np.max(self.model.predict(next_state)[0])
            x_batch.append(state[0])
            y_batch.append(y_target[0])

        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)

