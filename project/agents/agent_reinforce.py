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


class REINFORCE(Agent):
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

        super(REINFORCE, self).__init__(env, args, model, device)
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.num_actions = self.env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load optimizer and loss function here
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
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
        if self.num_actions == 4:
            policy = self.model.forward(
                    torch.from_numpy(np.array([observation]).transpose(0, 3, 1, 2)).float().to(self.device))
            action = np.random.choice(self.num_actions, p=np.squeeze(policy.detach().cpu().numpy()))
        else:
            mean, sigma = self.model.forward(torch.from_numpy(observation).float().to(self.device))

            # Sample an action from the distribution
            action_distro = torch.distributions.Normal(mean, sigma)
            raw_action = action_distro.sample(sample_shape=torch.Size([1]))

            # Make sure these are within the proper boundaries
            raw_action = raw_action.flatten()
            action = torch.tanh(raw_action)

            # Store the log probability for calculating loss later
            self.log_probs = action_distro.log_prob(action).to(self.device)

            ###########################

            # Return the actual action (probs is a bad name at this point...)
            return action.cpu().numpy()

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
        training_states = self.buffer.replay_buffer_states
        training_actions = self.buffer.replay_buffer_actions
        training_rewards = self.buffer.replay_buffer_rewards
        # training_next_states = self.buffer.replay_buffer_next_states
        # training_dones = self.buffer.replay_buffer_dones
        return training_states, training_actions, training_rewards

        ###########################

    def can_train(self):
        return self.buffer.replay_buffer_dones[-1]


    def train(self, episode=None, step=None, current_tuple=None):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        # Reset our gradient
        self.optimizer.zero_grad()

        # Get Full-batch for training
        training_states, training_actions, training_rewards = self.replay_buffer()

        returns = []
        log_probabilities = []

        for i in range(len(training_states)):
            state = training_states[i]
            action = training_actions[i]
            log_probabilities.append(self.calculate_log_prob(state, action))
            Gt = 0
            step = 0
            for reward in list(training_rewards)[i:-1]:
                Gt += self.gamma ** step * reward
                step += 1
            returns.append(Gt)
        policy_gradients = []
        for Gt, log in zip(returns, log_probabilities):
            policy_gradients.append(-log * Gt)

        # Calculate our loss
        loss = torch.stack(policy_gradients).sum()

        # Perform gradient descent
        loss.backward()

        # Update parameters
        self.optimizer.step()

        # Clear buffer
        self.buffer.clear()

        # Return loss
        return loss.detach().item()

    def calculate_log_prob(self, state, action):
        policy = self.model.forward(
            torch.from_numpy(np.array([state]).transpose(0, 3, 1, 2)).float().to(self.device))
        log_prob = torch.log(policy.squeeze(0)[action])
        return log_prob
