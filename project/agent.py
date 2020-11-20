"""

### NOTICE ###
DO NOT revise this file

"""
import sys
from environment import Environment
import torch
from agents.replay_buffer import ReplayBuffer

class Agent(object):
    def __init__(self, env, args, model_class, device):
        self.env = env
        self.lr = args.lr
        self.gamma = args.gamma
        self.replay_buffer_size = args.replay_buffer_size
        self.replay_buffer_batch_size = args.replay_buffer_batch_size
        self.min_buffer_size = args.min_buffer_size
        self.device = device
        self.target_update_steps = args.target_network_update_interval
        self.model_class = model_class
        self.model = model_class().to(device)

        self.buffer = ReplayBuffer(self.replay_buffer_size, self.replay_buffer_batch_size)

    def make_action(self, observation, epsilon, test=True):
        """
        Return predicted action of your agent
        This function must exist in agent

        Input:
            When running dqn:
                observation: np.array
                    stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        raise NotImplementedError("Subclasses should implement this!")


    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        raise NotImplementedError("Subclasses should implement this!")

    def can_train(self):
        """
        
        Indicates the agent has all resources needed for training.

        """
        raise NotImplementedError("Subclasses should implement this!")

    def train(self, episode=None, step=None, current_tuple=None):
        """

        Training method called at each step during training.

        """
        raise NotImplementedError("Subclasses should implement this!")
