#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
import os
import sys
import copy

import torch
import torch.nn.functional as F
import torch.optim as optim

from agent import Agent
from dqn_model import DQN
from .replay_buffer import ReplayBuffer

"""
you can import any package and define any extra function as you need
"""

class SampleAgent(Agent):
    def __init__(self, env, args, model, device):
        """
        Initialize everything you need here.
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        super(SampleAgent,self).__init__(env, args, model, device)
        ###########################
        # YOUR IMPLEMENTATION HERE #

        # Load optimizer and loss function here
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = torch.nn.SmoothL1Loss()

        print(self.model)
        # Any agent specific items done here
        self.target_model = copy.deepcopy(self.model)

    def make_action(self, observation, epsilon, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array [preprocessed before input]
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        # with torch.no_grad(): ?
        if np.random.rand(1) < 1. - epsilon or test:
        	print(np.array([observation]))
        	print(np.array([observation]).shape)
        	print(self.model)
        	print(dir(self.env))
        	if self.env.env.unwrapped.spec.id == 'Breakout-v0':
        		actions = self.model.forward(torch.from_numpy(np.array([observation]).transpose(0, 3, 1, 2)).float().to(self.device))
        	elif self.env.env.unwrapped.spec.id == 'MountainCar-v0':
        		print(torch.from_numpy(np.array([observation])).float().to(self.device).shape)
        		actions = self.model.forward(torch.from_numpy(np.array([observation])).float().to(self.device))
        	print(actions)
        	action = torch.argmax(actions).item()
        else:

            action = np.random.randint(self.env.env.action_space.n)
        print(action, self.env.env.action_space.n)
        ###########################
        return action
    
    def push(self, replay_entry=None):
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.
        
        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.buffer.push(replay_entry)
        
        ###########################
        
        
    def replay_buffer(self):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        return self.buffer.get_mini_batch()
        
        ###########################

    def can_train(self):
        if self.buffer.len() >= self.min_buffer_size:
            return True
        return False
        

    def train(self, episode=None, step=None, current_tuple=None):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        # Reset our gradient
        self.optimizer.zero_grad()

        # Get mini-batch for training
        training_states, training_actions, training_rewards, training_next_states, training_dones = self.replay_buffer()

        # Obtain our predictions



        if self.env.env.unwrapped.spec.id == 'Breakout-v0':
        	predicted_Q = self.model(torch.from_numpy(training_states.transpose(0, 3, 1, 2)).float().to(self.device))
        elif self.env.env.unwrapped.spec.id == 'MountainCar-v0':
        	predicted_Q = self.model(torch.from_numpy(training_states).float().to(self.device))


        predicted_Q_A = predicted_Q.gather(1, torch.from_numpy(training_actions).to(self.device).unsqueeze(1)).squeeze()

        # Obtain target Q values
        with torch.no_grad():
            print(training_next_states.shape)

            if self.env.env.unwrapped.spec.id == 'Breakout-v0':
                target_Q = self.target_model(torch.from_numpy(training_next_states.transpose(0, 3, 1, 2)).float().to(self.device))
            elif self.env.env.unwrapped.spec.id == 'MountainCar-v0':
                target_Q = self.target_model(torch.from_numpy().float().to(self.device))

            target_Q_A = target_Q.max(dim=1)[0]
            target_Q_A_idx = target_Q.argmax(dim=1)
            target_Q_A[training_dones] = 0.0
            target_Q_A = torch.from_numpy(training_rewards).to(self.device) + (target_Q_A * self.gamma)

        # Calculate our loss
        loss = self.criterion(predicted_Q_A.float(), target_Q_A.float())
    
        # Perform gradient descent
        loss.backward()

        # Update parameters
        self.optimizer.step()

        # Periodically update target model
        if step % self.target_update_steps == 0:
            self.target_model = copy.deepcopy(self.model)
        
        # Return loss
        return loss.detach().item()
