#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim

from agent import Agent
from dqn_model import DQN
from .replay_buffer import ReplayBuffer

"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)


class Agent_DQN(Agent):
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

        super(Agent_DQN,self).__init__(env, args, model, device)
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        # catch model args
        
        self.batch_size = batch_size_arg_here
        
        sample_state = env.reset()
        self.game_dim = sample_state.shape
        self.state_dim = sample_state.unsqueeze(0).shape
        self.batch_dim = torch.cat(self.batch_size*[sample_state.unsqueeze(0)]).shape
        
        print(self.game_dim)
        print(self.state_dim)
        print(self.batch_dim)
        
        self.model = model(self.model_args)
        self.Qtarget = model(self.model_args)
        self.replay_buffer = agents.replay_buffer.ReplayBuffer()

        # Construct your optimizer here
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # model-specific arguments
        self.target_update_interval = 5000

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        ###########################
        pass
    
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
        if test:
            action = torch.argmax(self.Q(observation))
            
        else:
            if self.epsilon >= torch.rand(1):
                action = torch.randint(self.num_actions)
        ###########################
        return int(action)
    
    def push(self, replay_entry=None):
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.
        
        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
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
        training_next_states = self.buffer.replay_buffer_next_states
        training_dones = self.buffer.replay_buffer_dones
        return training_states, training_actions, training_rewards, training_next_states, training_dones
        
        
        ###########################

    def can_train(self):
       return  True
        

    def train(self, episode=None, step=None, current_tuple=None):
        """
        Implement your training algorithm here
        agent_dqn training occurs largely in agent_runner.py : the difference is that agent_runner does the outside
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        if not self.can_train():
            RaiseError('Can\'t train: self.can_train == False')
                    
        self.optimizer.zero_grad()
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, next_states, rewards, terminals = zip(*minibatch)
        
        q_input = torch.stack(next_states, axis = 0).reshape(self.batch_size, self.game_size)
        yj = torch.Tensor(rewards, device = self.device) + self.gamma * torch.max(self.Qtarget(q_input), dim=1)[0]
        
        yj[terminals == True] = rewards[terminals == True] # special case of terminals, overwritten
        
        states = torch.stack(states, axis = 0).reshape(self.batch_size, self.game_size)
        Q_out = self.Q(states)
        # HERE DOWN NEEDS HELP:
        outlist = torch.zeros((self.batch_size, 1), device = device)

        loss = loss_func(outlist, torch.Tensor(yj).reshape(16,1))
        loss.backward()
        Q_optimizer.step()
        return 0
        
        ###########################
        
