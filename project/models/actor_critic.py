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

    def forward(self, x):
        x = self.linear1(x)
        x = th.relu(x)
        x = self.linear2(x)
        x = th.relu(x)
        means = self.linear3(x)
        stds = self.output_stds(x)

        means = th.tanh(means)
        stds = F.softplus(stds)
        value = self.output_value(x)

        # Here, we return a mean and standard deviation for
        # EACH action as well as the value for this input state
        return means, stds, value
