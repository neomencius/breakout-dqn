import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from agent import Agent

class DQN(Agent):
    def __init__(self, env, state_space, action_space, discount=0.9, epsilon=1.0, epsilon_min=0.01, epsilon_log_decay=0.995, alpha=0.01, alpha_decay=0.01, batch_size=64):
        self.state_size = state_space.shape[0]
        self.action_size = action_space.n

        self.env = env

        self.memory = deque(maxlen=10000)
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_log_decay
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.batch_size = batch_size

        self.model = self._build_model()
        self.model_prev = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.action_size, activation='sigmoid'))
        model.compile(loss='mse', optimizer=Adam(lr=0.01))
        return model

    def observe(self, state, action, next_state, reward, done):
        state = self.preprocess_state(state)
        next_state = self.preprocess_state(next_state)

        self.memory.append((state, action, next_state, reward, done))
        self.update()

        if done: # end of episode
            self.replay()

    def getAction(self, state):
        state = self.preprocess_state(state)

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

    def preprocess_state(self, state):
        return np.reshape(state, [1, 4])
