#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 10:48:31 2020

@author: gwenda
"""


#  A car is on a one-dimensional track, positioned between two "mountains".
#  The goal is to drive up the mountain on the right; however,
#  the car's engine is not strong enough to scale the mountain in a single pass. Therefore,
#  the only way to succeed is to drive back and forth to build up momentum.


import gym
import matplotlib.pyplot as plt

import torch


import numpy as np
env = gym.make('LunarLander-v2')
env.seed(0)
np.random.seed(0)


class SNet(torch.nn.Module):
    def __init__(self, num_input, num_hidden, num_output):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(SNet, self).__init__()

        self.num_hidden = num_hidden

        num_in = num_input
        self.totals = []
        layers = []
        # First layer is just concatenated inputs with hidden
        for num_out in num_hidden:
            layers.append(torch.nn.Linear(num_in, num_out))
            self.totals.append(torch.zeros((1,num_out), device=None))
            #self.totals.append(torch.zeros((1,num_out), device=None, requires_grad=True))
            num_in = num_out

        num_out = num_output
        layers.append(torch.nn.Linear(num_in, num_out))
        self.totals.append(torch.zeros((1,num_out), device=None))
                
        #self.sigmoid = torch.nn.Sigmoid()
        #self.l_relu = torch.nn.LeakyReLU()
        #self.relu = torch.nn.ReLU()
        # CrossEntropyLoss does the soft max computation.
        #self.SoftMax = nn.Softmax(dim=1)
        # NLLLoss, nn.LogSoftmax()
        self.layers = torch.nn.Sequential(*layers)
        

        
    def copy(self, in_net):
        self.cpu()
        in_net.cpu()
        for l1, l0 in zip(self.layers, in_net.layers):
            if not l1 is None and not l0 is None:
                l1.bias = torch.nn.Parameter(torch.FloatTensor(l0.bias.clone()))
                l1.weight = torch.nn.Parameter(torch.FloatTensor(l0.weight.clone()))
        self.cuda(0)
        in_net.cuda(0)

        


    def set_neuron_bias(self, layer_idx, neuron_idx, bias):
        """ Set the weight of only a single synapse.
        """
        layer = self.layers[layer_idx]
        layer.bias[neuron_idx] = bias
        

        
    def set_synapse_weight(self, layer_idx, neuron_idx, synapse_idx, weight):
        """ Set the weight of only a single synapse.
        """
        layer = self.layers[layer_idx]
        layer.weight[synapse_idx, neuron_idx] = weight
        
    
    
    def set_neuron_biases(self, biases):
        """ Set the biases of all neurons.
            Need to have the shape of biases match shape of neurons in network
        """
        for l in range(len(self.layers)):
            layer = self.layers[l]
            for n in range(len(layer.bias)):
                layer.bias[n] = biases[l][n]
                    
    def set_synapse_weights(self, weights):
        """ Set the weight of all synapses.
            Need to have the shape of weights match shape of weights in network
        """
        for l in range(len(self.layers)):
            layer = self.layers[l]
            for n in range(len(layer.weight)):
                for s in range(len(layer.weight[0])):
                    layer.weight[n,s] = weights[l][s][n]

        
    def forward(self, x):
        """
        Integrate and fire.
        Input values should be 1 or 0.

        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        num = len(self.layers)
        for idx in range(num):
            totals = self.totals[idx]
            # Linear
            x = self.layers[idx](x)
            totals += x
            if idx != num-1:
                out = (totals >= torch.Tensor([0.1])).float() * .1
                totals -= out
                x = out
                
        return out
        
    def get_output(self):
        empty_in = torch.Tensor(np.zeros((1,14)))
        out = self.forward(empty_in)
        while sum(out[0]) > 0:
            out = self.forward(empty_in)
            
        return self.totals[-1]
        
    
    def reset(self):
        for i in range(len(self.totals)):
            self.totals[i] = torch.zeros((1,len(self.totals[i][0])), device=None)


def process_input(state):
    pre_state = state[0:6]*-1
    pre_state[pre_state<0] = 0
    state[state<0] = 0
    new_state = np.concatenate((pre_state,state),axis=None)
    new_state = [round(x,1) for x in new_state]
    new_state = np.reshape(new_state, (1, 14))
    return new_state


def run_dqn(episode):
    
    scores = []

    weights = np.load('weights_relu_place.npy',allow_pickle=True)
    weights = [arr.tolist() for arr in weights]
    biases = np.load('biases_relu_place.npy',allow_pickle=True)
    biases = [arr.tolist() for arr in biases]
    
    agent = SNet(14,[150,120],env.action_space.n)
    agent.set_neuron_biases(biases)
    agent.set_synapse_weights(weights)
    
    for e in range(episode):
        state = env.reset()
        state = process_input(state)
        score = 0
        max_steps = 3000
        for i in range(max_steps):
            action = [0,0,0,0]
            
            while state[state>0].size != 0:
                agent(torch.Tensor(np.minimum(state, .1).tolist()))
                state[state>0] -= .1
                state[state<0.1] = 0

            action = np.argmax(agent.get_output().tolist())
            agent.reset()
            env.render()
            
            next_state, reward, done, _ = env.step(action)
            score += reward
            next_state = process_input(next_state)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}".format(e, episode, score))
                break
        scores.append(score)

        # Average score of last 100 episode
        is_solved = np.mean(scores[-100:])

        print("Average over last 100 episode: {0:.2f} \n".format(is_solved))
        

    return scores


if __name__ == '__main__':
    
    print(env.observation_space)
    print(env.action_space)
    episodes = 200
    loss = run_dqn(episodes)
    env.close()
    plt.hist(loss[::2],bins=20)
    plt.show()
