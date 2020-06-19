#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 12:43:54 2020

@author: not gwenda yet
"""

#  A car is on a one-dimensional track, positioned between two "mountains".
#  The goal is to drive up the mountain on the right; however,
#  the car's engine is not strong enough to scale the mountain in a single pass. Therefore,
#  the only way to succeed is to drive back and forth to build up momentum.


import gym
import random
from keras import Sequential
from collections import deque
from keras.layers import Dense
from keras.optimizers import adam
import matplotlib.pyplot as plt
from keras.activations import linear, relu

import numpy as np
env = gym.make('LunarLander-v2')
env.seed(0)
np.random.seed(0)




class DQN:

    """ Implementation of deep q learning algorithm """

    def __init__(self, action_space, state_space):

        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 1.0
        self.gamma = .99
        self.batch_size = 64
        self.epsilon_min = .01
        self.lr = 0.001
        self.epsilon_decay = .996
        self.memory = deque(maxlen=1000000)
        self.model = self.build_model()

    def build_model(self):

        model = Sequential()
        model.add(Dense(150, input_dim=self.state_space, activation=relu))
        model.add(Dense(120, activation=relu))
        model.add(Dense(self.action_space, activation=relu))
        model.compile(loss='mse', optimizer=adam(lr=self.lr))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self):

        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + self.gamma*(np.amax(self.model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.model.predict_on_batch(states)
        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def get_weights(self):
        return self.model.get_weights()


def train_dqn(episode):

    loss = []
    #weights = np.load('weights_relu_pos.npy',allow_pickle=True)
    agent = DQN(env.action_space.n, 14)
    for e in range(episode):
        state = env.reset()
        pre_state = state[0:6]*-1
        pre_state[pre_state<0] = 0
        state[state<0] = 0
        state = np.concatenate((pre_state,state),axis=None)
        state = np.reshape(state, (1, 14))
        score = 0
        max_steps = 3000
        for i in range(max_steps):
            action = agent.act(state)
            env.render()
            next_state, reward, done, _ = env.step(action)
            score += reward
            pre_state = next_state[0:6]*-1
            pre_state[pre_state<0] = 0
            next_state[next_state<0] = 0
            next_state = np.concatenate((pre_state,next_state),axis=None)
            next_state = np.reshape(next_state, (1, 14))
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay()
            if done:
                print("episode: {}/{}, score: {}".format(e, episode, score))
                break
        loss.append(score)

        # Average score of last 100 episode
        is_solved = np.mean(loss[-100:])
        if is_solved > 200:
            print('\n Task Completed! \n')
            break
        print("Average over last 100 episode: {0:.2f} \n".format(is_solved))
        
    return loss, agent


if __name__ == '__main__':

    print(env.observation_space)
    print(env.action_space)
    episodes = 1000
    loss, agent = train_dqn(episodes)
    weights = agent.model.get_weights()[0]
    np.save('weights_relu_place2',weights)
    biases = agent.model.get_weights()[1]
    np.save('biases_relu_place2',biases)
    env.close()
    plt.plot([i+1 for i in range(0, len(loss), 2)], loss[::2])
    plt.show()
