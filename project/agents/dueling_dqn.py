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

        self.env = env

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
            actions = self.model.forward(torch.from_numpy(np.array([observation]).transpose(0, 3, 1, 2)).float().to(self.device))
            action = torch.argmax(actions).item()
        else:
            action = np.random.randint(self.env.env.action_space.n)
        
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

        self.game_shape = (32,4,84,84)

        # Get mini-batch for training
        training_states, training_actions, training_rewards, training_next_states, training_dones = self.replay_buffer()
        states = torch.FloatTensor(training_states, device = self.device).reshape(self.game_shape)
        actions = torch.LongTensor(training_actions, device = self.device)
        rewards = torch.FloatTensor(training_rewards, device = self.device)
        next_states = torch.FloatTensor(training_next_states, device = self.device).reshape(self.game_shape)
        dones = torch.FloatTensor(training_dones, device = self.device)

        curr_Q = self.model.forward(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        next_Q = self.model.forward(next_states)
        max_next_Q = torch.max(next_Q, 1)[0]
        #print(rewards)
        #print(rewards.squeeze(0))
        expected_Q = rewards + self.gamma*max_next_Q
        #expected_Q = rewards.squeeze(1) + self.gamma * max_next_Q

        loss = torch.nn.functional.mse_loss(curr_Q, expected_Q)
        self.optimizer.zero_grad()
        loss.backward()

        # Update parameters
        self.optimizer.step()

        # Periodically update target model
        if step % self.target_update_steps == 0:
            self.target_model = copy.deepcopy(self.model)
        
        # Return loss
        return loss.detach().item()
