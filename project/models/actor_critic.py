#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import ipdb


class ActorCritic(nn.Module):
    """
    Initialize a CNN for actor critic.
    """

    def __init__(self, state_dims=2, num_actions=1):
        """
        Parameters:
        -----------
        in_channels: number of channel of input.
                i.e The number of most recent frames stacked together, here we use 4 frames, which means each state in Breakout is composed of 4 frames.
        num_actions: number of action-value to output, one-to-one correspondence to action in game.

        You can add additional arguments as you need.
        In the constructor we instantiate modules and assign them as
        member variables.
        """
        super(ActorCritic, self).__init__()
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.num_actions = num_actions
        self.state_dims = state_dims

        self.linear1 = th.nn.Linear(state_dims, 512)
        self.linear2 = th.nn.Linear(512, 512)
        self.linear3 = th.nn.Linear(512, self.num_actions)
        self.output_value = th.nn.Linear(in_features=512, out_features=1)
        self.output_stds = th.nn.Linear(512, self.num_actions)

        #self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=32, kernel_size=8, stride=4)
        #self.act1 = nn.ReLU()
        #self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        #self.act2 = nn.ReLU()
        #self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        #self.act3 = nn.ReLU()
        ## 3136 = breakout
        ## 4096 = car racing
        #self.fully_connected_hidden = nn.Linear(in_features=4096, out_features=512)
        #self.act4 = nn.ReLU()
        #self.output_mu = nn.Linear(in_features=512, out_features=self.num_actions)
        #self.output_std = nn.Linear(in_features=512, out_features=self.num_actions)
        #self.output_value = nn.Linear(in_features=512, out_features=1)

    def forward(self, x):
        ## Normalize data to be between 0 and 1
        #x = x / 255.0

        #x = self.conv1(x)
        #x = self.act1(x)
        #x = self.conv2(x)
        #x = self.act2(x)
        #x = self.conv3(x)
        #x = self.act3(x)
        #x = x.contiguous().view(x.size()[0], -1)
        #x = self.fully_connected_hidden(x)
        #x = self.act4(x)
        x = self.linear1(x)
        x = th.relu(x)
        x = self.linear2(x)
        x = th.relu(x)
        means = self.linear3(x)
        stds = self.output_stds(x)
        #mu = self.output_mu(x)

        means = th.tanh(means)
        stds = F.softplus(stds)
        value = self.output_value(x)

        # Here, we return a mean and standard deviation for
        # EACH action as well as the value for this input state
        return means, stds, value
