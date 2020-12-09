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
from test import test
"""
you can import any package and define any extra function as you need
"""
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import pickle

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)

class Metrics():
    def __init__(self, name=None, run_path=None, args=None):
        self.run_path = run_path

        # Save off original arguments
        with open(f'{self.run_path}/args.pkl', 'wb') as f:
            pickle.dump(args, f)

        # Training measurements
        self.episode = []
        self.step = []
        self.epsilon = []
        self.episode_loss = []
        self.episode_reward = []
        self.max_episode_reward = []
        self.episode_moving_average_30 = []

        # Test measurements
        self.test_episode = []
        self.test_reward_average_last_100 = []

    def add_training_measurement(self, episode, step, epsilon, episode_loss, episode_reward):
        # Capture measurements
        self.episode.append(episode)
        self.step.append(step)
        self.epsilon.append(epsilon)
        self.episode_loss.append(episode_loss)
        self.episode_reward.append(episode_reward)
        self.max_episode_reward.append(max(self.episode_reward))
        self.episode_moving_average_30.append(sum(self.episode_reward[-30:]) / 30)

    def add_test_measurement(self, episode, reward):
        # Capture measurements
        self.test_episode.append(episode)
        self.test_reward_average_last_100.append(reward)

    def write_metrics(self):
        # Training Metrics
        metrics_df = pd.DataFrame({
            'episode':self.episode,
            'step':self.step,
            'epsilon':self.epsilon,
            'episode_loss':self.episode_loss,
            'max_episode_reward':self.max_episode_reward, 
            'episode_moving_average_30':self.episode_moving_average_30,
        })

        # Write out to CSV
        metrics_df.to_csv(f'{self.run_path}/{self.episode[-1]}_training_metrics.csv', index=False)

        # Create reward plot
        plt.figure()
        plt.plot(metrics_df.episode, metrics_df.episode_moving_average_30)
        plt.xlabel('Number of Episodes')
        plt.ylabel('Average Reward in Last 30 Episodes')
        plt.title('Reward vs. Episode')
        plt.savefig(f'{self.run_path}/{self.episode[-1]}_training_reward_plot.png')
        plt.close()

        # Do not do anything with test metrics until we have some
        if len(self.test_episode) == 0:
            return

        # Test Metrics
        test_metrics_df = pd.DataFrame({
            'episode':self.test_episode,
            'test_reward_average_last_100':self.test_reward_average_last_100,
        })

        # Write out to CSV
        test_metrics_df.to_csv(f'{self.run_path}/{self.episode[-1]}_test_metrics.csv', index=False)

        plt.figure()
        plt.plot(self.test_episode,self.test_reward_average_last_100) 
        plt.xlabel('Training Episode')
        plt.ylabel('Mean Reward Over 100 Episodes')
        plt.title('Reward vs. Episode')
        plt.savefig(f'{self.run_path}/{self.episode[-1]}_test_reward_plot.png')
        plt.close()

    def display(self):
        print(f'[{self.episode[-1]}]') 
        print('Training Episode Metrics:')
        print(f'epsilon = [{self.epsilon[-1]}], steps = [{self.step[-1]}], episode_loss = [{self.episode_loss[-1]}], max_episode_reward = [{self.max_episode_reward[-1]}], episode_moving_average_30 = [{self.episode_moving_average_30[-1]}]')
        print('')

        # Do not do anything with test metrics until we have some
        if len(self.test_episode) == 0:
            return

        print('Test Metrics:')
        print(f'test_reward_average_last_100 = [{self.test_reward_average_last_100[-1]}]')
        print('')

class AgentRunner():
    def __init__(self, env, args, agent, model_class):
        """
        Initialize everything you need here.
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        ###########################
        # YOUR IMPLEMENTATION HERE #

        self.env = env
        self.args = args

        # Pytorch device to use
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Episodes to execute
        self.start_episode = args.start_episode
        self.num_episodes = args.start_episode + args.num_episodes
        self.test_cycle_interval = args.test_cycle_interval

        # Rendering options
        self.render_train = args.render_train
        self.render_test = args.render_test

        # Epsilon options
        self.epsilon = args.epsilon
        self.epsilon_min = args.epsilon_min
        self.epsilon_steps = args.epsilon_steps
        self.epsilon_decay_per_step = (self.epsilon - self.epsilon_min) / self.epsilon_steps

        # Construct agent
        # Note:  Agent superclass instantiates the model, we simply hold on
        #        to the class (prolly don't need to though...)
        self.model_class = model_class
        self.agent = agent(self.env, self.args, self.model_class, self.device)

        if args.model_path and args.optimizer_path:
            self.load_weights(args.model_path, args.optimizer_path)

        # Create main archive area if needed
        self.archive_dir = args.archive_dir
        if not os.path.exists(self.archive_dir):
            os.makedirs(self.archive_dir, exist_ok=True)

        # Create a new directory to store data from this run
        now = datetime.datetime.now()
        self.run_name = args.run_name + '-' + now.strftime('%Y.%m.%d.%H.%M.%S')
        os.mkdir(f'{self.archive_dir}/{self.run_name}')

        # Create training/test metrics object
        self.metrics_display_interval = 100
        self.metrics_save_interval = 1000
        self.training_metrics = Metrics(name='training_metrics', run_path=f'{self.archive_dir}/{self.run_name}', args=args)

        # Set model to eval
        if args.test_dqn:
            self.agent.model.eval()

    def load_weights(self, model_path, optimizer_path):
        print(f'Loading weights using [{model_path}], [{optimizer_path}]')
        self.agent.model.load_state_dict(
            torch.load(model_path))

        self.agent.optimizer.load_state_dict(
            torch.load(optimizer_path))

    def save_weights(self, episode):
        torch.save(self.agent.model.state_dict(), f'{self.archive_dir}/{self.run_name}/{episode}_model.pth')
        torch.save(self.agent.optimizer.state_dict(), f'{self.archive_dir}/{self.run_name}/{episode}_optimizer.pth')
            
    def perform_epsilon_decay(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay_per_step
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min
        else:
            self.epsilon = self.epsilon_min

    def run_test_cycle(self, current_training_episode=None):
        # Put model into test mode
        self.agent.model.eval()

        total_test_rewards = 0.0
        for i in range(100):
            test_state = self.env.reset()

            if self.render_test:
                self.env.env.render()

            test_done = False
            test_episode_reward = 0.0

            #playing one game
            while(not test_done):
                test_action = self.agent.make_action(test_state, self.epsilon, test=True)

                if self.render_test:
                    self.env.env.render()

                test_state, test_reward, test_done, test_info = self.env.step(test_action)
                test_episode_reward += test_reward
            total_test_rewards += test_episode_reward

        self.training_metrics.add_test_measurement(current_training_episode, total_test_rewards / 100)

        # Put model back into training mode
        self.agent.model.train()

    def train(self):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.agent.model.train()
        
        current_step = 0
        for i in range(self.start_episode, self.num_episodes):
            episode_reward = 0
            episode_loss = 0
            current_observation = self.env.reset()

            if self.render_train:
                self.env.env.render()

            episode_complete = False
            while not episode_complete:
                current_step += 1

                with torch.no_grad():
                    action = self.agent.make_action(current_observation, self.epsilon, test=False)

                next_observation, reward, episode_complete, info = self.env.step(action)
                episode_reward += reward

                if self.render_train:
                    self.env.env.render()

                self.perform_epsilon_decay()

                # Take action and add it to the replay buffer
                current_tuple = (current_observation, action, reward, next_observation, episode_complete)
                self.agent.push(replay_entry=current_tuple)

                # Update our current state
                current_observation = next_observation

                if self.agent.can_train():
                   loss = self.agent.train(episode=i, step=current_step, current_tuple=current_tuple)
                   episode_loss += loss

            # Check if we should perform a round of testing
            if i % self.test_cycle_interval == 0:
                self.run_test_cycle(current_training_episode=i)

            # Capture metrics after each episode
            self.training_metrics.add_training_measurement(i, current_step, self.epsilon, episode_loss, episode_reward)

            # Display measurements to screen periodically
            if i % self.metrics_display_interval == 0:
                self.training_metrics.display()

            # Save metrics and models to disk periodically
            if i % self.metrics_save_interval == 0:
                self.training_metrics.write_metrics()
                self.save_weights(i)

    def test(self):
        test(self.agent, self.env, total_episodes=100, render_test=self.render_test)
