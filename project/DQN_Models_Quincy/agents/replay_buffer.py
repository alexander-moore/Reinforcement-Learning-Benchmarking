#!/usr/bin/env python

from collections import deque
import numpy as np
import random

class ReplayBuffer():
    def __init__(self, buffer_size=10000, batch_size=128):
        self.buffer_size = buffer_size
        self.batch_size=batch_size
        self.replay_buffer_states = deque(maxlen=self.buffer_size)
        self.replay_buffer_actions = deque(maxlen=self.buffer_size)
        self.replay_buffer_rewards = deque(maxlen=self.buffer_size)
        self.replay_buffer_next_states = deque(maxlen=self.buffer_size)
        self.replay_buffer_dones = deque(maxlen=self.buffer_size)

    def push(self, replay_entry=None):
        if replay_entry:
            self.replay_buffer_states.append(replay_entry[0])
            self.replay_buffer_actions.append(replay_entry[1])
            self.replay_buffer_rewards.append(replay_entry[2])
            self.replay_buffer_next_states.append(replay_entry[3])
            self.replay_buffer_dones.append(replay_entry[4])

    def get_mini_batch(self):
            return [np.array(x) for x in zip(*random.sample(list(zip(
                self.replay_buffer_states, 
                self.replay_buffer_actions, 
                self.replay_buffer_rewards, 
                self.replay_buffer_next_states, 
                self.replay_buffer_dones)),
                self.batch_size))]

    def len(self):
        return len(self.replay_buffer_states)

    def clear(self):
        self.replay_buffer_states.clear()
        self.replay_buffer_actions.clear()
        self.replay_buffer_rewards.clear()
        self.replay_buffer_next_states.clear()
        self.replay_buffer_dones.clear()

#WHY IS RETURN ON MINI BATCH DOUBLE BUMPED IN?
#WHERE IS LEARNING RATE IN ALL THIS