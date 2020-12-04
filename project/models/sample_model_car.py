#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F


class SampleModel(nn.Module):
    """Initialize a deep Q-learning network

    Hints:
    -----
        Original paper for DQN
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf

    This is just a hint. You can build your own structure.
    """

    def __init__(self, num_actions=3, state_size = 2):
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
        super(SampleModel, self).__init__()
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.num_actions = num_actions
        self.state_size = state_size

        self.fc1 = nn.Linear(state_size, state_size*2)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)

        self.fc4 = nn.Linear(in_features=self.fc3.out_features, out_features=self.num_actions)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #

        # Normalize data to be between 0 and 1
        x = x / 255.0

        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act1(x)
        x = self.fc3(x)
        x = self.act1(x)
        x = self.fc4(x)

        ###########################
        return x
