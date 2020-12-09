#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class PPODiscrete(nn.Module):
    """
    Initialize a deep neural network for PPO. Leveraged one used for DQN in the Atari environment,
    specifically for Breakout. However, the final full-connected layers need to be changed to output
    the policy (pi -- actor) and value (critic) functions.

    """

    def __init__(self, in_channels=4, num_actions=4):
        """
        Parameters:
        -----------
        in_channels: number of channel of input.
                i.e The number of most recent frames stacked together, here we use 4 frames,
                which means each state in Breakout is composed of 4 frames.
        num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(PPODiscrete, self).__init__()
        ###########################

        # The 1st layer takes an 84x84 frame and outputs a 20x20 frame.
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        # The 2nd layer outputs a 9x9 frame.
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # The 3rd layer outputs a 7x7 frame.
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # The 4th layer is a full-connected layer and outputs 512 features.
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        # self.fc5 = nn.Linear(512, num_actions)

        # The 5th layer (also fully-connected) is to get logit for π (replaces self.fc5 originally).
        # Logit function result in a probability (of actions). This is the actor.
        self.piLogit = nn.Linear(in_features=512, out_features=num_actions)

        # And finally a fully-connected layer to get the value function (the critic).
        self.value = nn.Linear(in_features=512, out_features=1)

    def init_weights(m):
        """
        Define a function to set the initial weights.

        """
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.1)
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        ###########################

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Input tensor must be flattened first.
        x = F.relu(self.fc4(x.view(x.size(0), -1)))

        # pi, the policy parameters, is the actor.
        pi = Categorical(logits=self.piLogit(x))

        # action-value (or state-value) function is the critic.
        value = self.value(x).reshape(-1)

        # x = self.fc5(x)

        # For continuous action space, we can add distribution by setting pi to mu
        # but have to get std. dev. as well.
        # self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        # std = self.log_std.exp().expand_as(mu)
        # dist = Normal(mu, std)
        ###########################
        # return x
        return pi, value
