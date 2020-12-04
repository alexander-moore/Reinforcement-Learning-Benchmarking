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
import ipdb

"""
you can import any package and define any extra function as you need
"""

class ActorCriticAgent(Agent):
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

        super(ActorCriticAgent,self).__init__(env, args, model, device)
        ###########################
        # YOUR IMPLEMENTATION HERE #

        # Create actor and critic models
        self.actor = self.model_class().to(self.device)
        self.critic = self.model_class().to(self.device)

        # These actions are a mean and standard deviation
        self.actor.num_actions = 2
        self.critic.num_actions = 2

        self.log_probs = None

        # Unclear if this will end up being correct
        self.n_outputs = 1

        # Load optimizer and loss function here
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        self.criterion = torch.nn.SmoothL1Loss()

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

        # Get our return from the network
        mu, sigma, val = self.actor.forward(torch.from_numpy(np.array([observation]).transpose(0, 3, 1, 2)).float().to(self.device))

        # Sample an action from the distribution
        action_probs = torch.distributions.Normal(mu, sigma)
        probs = action_probs.sample(sample_shape=torch.Size([self.n_outputs]))

        # Make sure these are within the proper boundaries
        probs = probs.flatten()
        probs[0] = torch.tanh(probs[0])
        probs[1:3] = torch.sigmoid(probs[1:3])

        # Store the log probability for calculating loss later
        self.log_probs = action_probs.log_prob(probs).to(self.device)

        ###########################

        # Return the actual action (probs is a bad name at this point...)
        return probs.cpu().numpy()
    
    def push(self, replay_entry=None):
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.
        
        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        # Replay buffer not used
        #self.buffer.push(replay_entry)
        
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
        return True
        

    def train(self, episode=None, step=None, current_tuple=None):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #

        # Get tuple information
        current_observation = current_tuple[0]
        action = current_tuple[1]
        reward = current_tuple[2]
        next_observation = current_tuple[3]
        episode_complete = current_tuple[4]

        # Reset our gradients
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        # Get critic's view
        _, _, critic_value_ = self.critic.forward(torch.from_numpy(np.array([next_observation]).transpose(0, 3, 1, 2)).float().to(self.device))
        _, _, critic_value = self.critic.forward(torch.from_numpy(np.array([current_observation]).transpose(0, 3, 1, 2)).float().to(self.device))
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)
        delta = ((reward + self.gamma * critic_value_ * (1 - int(episode_complete))) - critic_value)

        # Combined loss
        actor_loss = -1 * self.log_probs * delta
        actor_loss = actor_loss.sum()
        critic_loss = delta**2
        loss = actor_loss + critic_loss
        loss.backward()

        # Update parameters
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        # Return loss
        return loss.detach().item()
