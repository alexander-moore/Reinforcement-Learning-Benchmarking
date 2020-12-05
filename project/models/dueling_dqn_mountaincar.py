#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/dxyang/DQN_pytorch/blob/master/model.py
class Dueling_DQN(nn.Module):
    def __init__(self, game_dim, num_actions):
        super(Dueling_DQN, self).__init__()
        self.num_actions = num_actions
        
        self.fc1 = nn.Linear(game_dim, game_dim*2)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)

        self.fc1_adv = nn.Linear(self.fc2.out_features, out_features=8)
        self.fc1_val = nn.Linear(self.fc2.out_features, out_features=8)

        self.fc2_adv = nn.Linear(in_features=self.fc1_adv.out_features, out_features=num_actions)
        self.fc2_val = nn.Linear(in_features=self.fc1_val.out_features, out_features=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = x.view(x.size(0), -1)

        adv = self.relu(self.fc1_adv(x))
        val = self.relu(self.fc1_val(x))

        adv = self.fc2_adv(adv)
        val = self.fc2_val(val).expand(x.size(0), self.num_actions)
        
        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.num_actions)
        return x
