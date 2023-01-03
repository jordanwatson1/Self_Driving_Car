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
        :param self: refers to the object created from the Network class
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
        :param self: refers to the object instance created from the Network class
        :param state: the input of the NN  
        :return: The Q-values, the outputs of the NN. One Q-value for each action
        """
        x = F.relu(self.fc1(state)) # relu -> the rectifier function
        q_values = self.fc2(x)
        return q_values

class ReplayMemory(object):
    """
    Implementing experience replay. Keeping track of up to capacity, or 100 in our program, 
    states previous to the current state.
    """

    def __init__(self, capacity):
        """
        Define the variables attached to the ReplayMemory. This acts as 
        the architecture of the NN.
        :param self: refers to an instance of a ReplayMemory object        
        :param capacity: the long term memory capacity
        """
        self.capacity = capacity
        self.memory = [] # store the last 100 events

    def push(self, event):
        """
        Add a new event to memory.
        :param self: refers to an instance of a ReplayMemory object
        :param event: the new event to be appended to memory (st, st+1, at, rt)
        """
        self.memory.append(event)

        # Delete oldest memory if capacity is at its limit
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        """
        Takes ransom samples from memory to improve the Deep Q-Learning process.
        :param self: refers to an instance of a ReplayMemory object
        :param batch_size: the size of the stochastic/ mini batch gradient descent
        :return: return the randomly chosen samples 
        """
        # if list = ((1,2,3), (4,5,6)) -> zip(*list) = ((1,4) (2,3), (5,6))
        samples = zip(*random.sample(self.memory, batch_size)) # grab the random samples
        return map(lambda x: Variable(torch.cat(x, 0)), samples) # put each sample into a torch variable that contains a tensor and a gradient







