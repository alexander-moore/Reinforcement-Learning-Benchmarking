#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F


class ActorCriticDiscrete(nn.Module):


    def __init__(self, in_channels=4, num_actions=4):
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
        super(ActorCriticDiscrete, self).__init__()
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.in_channels = in_channels
        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=32, kernel_size=8, stride=4)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.act3 = nn.ReLU()
        self.fully_connected_hidden = nn.Linear(in_features=3136, out_features=512)
        self.act4 = nn.ReLU()
        self.action_layer = nn.Linear(in_features=512, out_features=self.num_actions)
        self.value_layer = nn.Linear(in_features=512, out_features=1)
        self.softmax = nn.Softmax(dim=-1)

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

        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = x.contiguous().view(x.size()[0], -1)
        x = self.fully_connected_hidden(x)
        x = self.act4(x)

        # Action head
        actions = self.action_layer(x)
        actions = self.softmax(actions)

        # Value head
        value = self.value_layer(x)

        ###########################
        return actions, value
