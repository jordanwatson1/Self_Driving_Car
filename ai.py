"""
File: ai.py
Author: Jordan Watson
Date: 01/02/2023
Artificial Intelligence A-Z™: Learn How To Build An AI
Email: jordanwatson915@gmail.com
Description: ai.py is the AI, or brains, for my self-driving car project using pytorch. It will include the neural network
which will allow the car to learn and dodge obstacles on a basic road using Deep Q-Learning. Stochasitc gradiant descent is 
used in combination with torch.optim
"""

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

class Network(nn.Module):
    """
    Neural Network that will decide which actions to take.
    :param nn.Module: inherit the tools from the nn.module class
    """
    
    def __init__(self, input_size, nb_action):
        """
        Define the variables attached to the Neural Network. This acts as 
        the architecture of the NN. 
        :param input_size: input layer, composed of 5 input neurons (5D for the encoded vector of the input state that describe one state of the environment)
        :param nb_action: output layer, the possible actions that can be played at a given state  
        """
        super(Network, self).__init__() # inherit from the nn.module class to the Network child class
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30) # full connection between the input layer and the hidden layer (30 neurons in the hidden layer)
        self.fc2 = nn.Linear(30, nb_action) # full connection between the hidden layer and the output layer (30 neurons in the hidden layer)
    
    def forward(self, state):
        """
        forward() performs forward propagation and activates the neurons in the NN (signals).
        Using the rectifier activation function because we are dealing with a non-linear problem.
        :param state: the input of the NN  
        :return: The Q-values, the outputs of the NN. One Q-value for each action
        """
        x = F.relu(self.fc1(state)) # relu -> the rectifier function
        q_values = self.fc2(x)
        return q_values

class ReplayMemory(object):
    """
    Implementing experience replay. Keeping track of up to capacity
    states previous to the current state.
    """

    def __init__(self, capacity):
        """
        Define the variables attached to the ReplayMemory.
        :param capacity: the long term memory capacity
        """
        self.capacity = capacity
        self.memory = [] # store the last 100 events

    def push(self, event):
        """
        Add a new event to memory.
        :param event: the new event to be appended to memory (st, st+1, at, rt)
        """
        self.memory.append(event)

        # Delete oldest memory if capacity is at its limit
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        """
        Takes ransom samples from memory to improve the Deep Q-Learning process.
        :param batch_size: the size of the stochastic/ mini batch gradient descent
        :return: return the randomly chosen samples 
        """
        # if list = ((1,2,3), (4,5,6)) -> zip(*list) = ((1,4) (2,3), (5,6))
        samples = zip(*random.sample(self.memory, batch_size)) # grab the random samples
        return map(lambda x: Variable(torch.cat(x, 0)), samples) # put each sample into a torch variable that contains a tensor and a gradient

class Dqn():
    """
    Deep Q-Learning that contains functions to select the right action at each
    time, update function, score function to get an idea of how the leraning is going,
    save function to save the model/ brain of the car, and a load function to load a saved model.
    """

    def __init__(self, input_size, nb_action, gamma):
        """ 
        Define the variables attached to the Deep Q-Learning model
        :param input_size: input layer, composed of 5 input neurons (5D for the encoded vector of the input state that describe one state of the environment)
        :param nb_action: output layer, the possible actions that can be played at a given state
        """
        self.gamma = gamma
        self.reward_window = [] # keep track of how the training is going by averaging the last 100 rewards, showing how the mean of the rewards is evolving
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0) # 5 dimensions -> convert to torch tensor
        self.last_action = 0
        self.last_reward = 0

    def select_action(self, state):
        """
        Selects the best action at each time using a softmax approach
        :param state: refers to the input state
        :return: the action to choose in the current state
        """
        # Ex: softmax([1,2,3]) = [0.04,0.11,0.85] => softmax([1,2,3]*3) = [0,0.02,0.98]
        probs = F.softmax(self.model(Variable(state, volatile = True))*100) # Temperature=100
        action = probs.multinomial(num_samples=1)
        return action.data[0,0]

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        """
        Train the NN with forward and backpropagation using stochastic gradient descent.
        Take a batch of transistions from the memory to learn the different outputs for each of the states
        to determine what is favorable and what is not.
        :param batch_state: a batch of states that the AI has been in previously
        :param batch_next_state: a batch of next states
        :param batch_reward: rewards of the previous batch states
        :param batch_action: actions of the previous batch states
        """
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1) # gather preferred actions chosen
        next_outputs = self.model(batch_next_state).detach().max(1)[0] # getting max of the Q values (Bellman's Equation)
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target) # temporal difference loss
        
        # backpropagation
        self.optimizer.zero_grad() 
        td_loss.backward(retain_graph = True)
        self.optimizer.step()
    
    def update(self, reward, new_signal):
        """
        Updates all elements of a transition (action, state, reward) as soon as the AI reaches a new state
        and select the action.
        :param reward: the reward from making a move to a new state (last reward)
        :param new_signal: the signal states from making a move to a new state [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]
        :return: returns the action to take next from the current state
        """
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)

        # Make AI learn from past 100 events
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000: # the last 1000 means of the last 100 rewards
            del self.reward_window[0]
        return action

    def score(self):
        """
        Calculate the mean of the rewards in reward_window
        :return: return the means of the rewards in the reward_window
        """
        return sum(self.reward_window)/(len(self.reward_window)+1.) # +1. to make sure length != 0
    
    def save(self):
        """
        Save the brain of the car when the user wants to exit the application
        """
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
    
    def load(self):
        """
        Allow the user to load the brain of a previously trained model that was saved
        """
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")