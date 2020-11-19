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
import agents.replay_buffer

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
        
        self.game_dim = env.reset().shape
        
        self.model = model(self.model_args)
        self.replay_buffer = deque(maxlen = 10000)

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
        
        
        
        ###########################
        return 

    def can_train(self):
        True
        

    def train(self, episode=None, step=None, current_tuple=None):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        if not self.can_train():
            RaiseError('Can\'t train: self.can_train == False')
            
        for i in range(self.n_iterations):
            state = self.env.reset()
            state_processed = torch.Tensor((state/255).reshape(1, self.game_size), device = self.device)
            
            terminal = False
            while not terminal:
                action = self.make_action(state_processed, test = False)
                new_state, reward, terminal, _ = self.env.step(action)
                new_state_processed = torch.Tensor((new_state/255).reshape(game_size), device = self.device) # can use .unsqueeze(0) to add go (x) -> (1,x)
                
                self.replay_buffer.append([state_processed, action, new_state_processed, reward, terminal])
                state_processed = new_state_processed
                
                # pop deque here? or does it self-pop
                
                if len(self.replay_buffer) > self.learning_start:
                    self.optimizer.zero_grad()
                    batch = random.sample(self.replay_buffer, self.batch_size)
                    states, actions, next_states, rewards, terminals = zip(*minibatch)
                    
                    q_input = torch.stack(next_states, axis = 0).reshape(self.batch_size, self.game_size)
                    yj = torch.Tensor(rewards, device = self.device) + self.gamma * torch.max(self.Qtarget(q_input), dim=1)[0]
                    
                    yj[terminals == True] = rewards[terminals == True] # special case of terminals, overwritten
                    
                    states = torch.stack(states, axis = 0).reshape(self.batch_size, self.game_size)
                    Q_out = self.Q(states)
                    
                    #...
                

        return 0
        
        ###########################
        
