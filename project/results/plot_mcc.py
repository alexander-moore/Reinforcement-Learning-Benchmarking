#!/usr/bin/env python

import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import ipdb

# Read in our test data
reinforce = pd.read_csv('./ReinforceTEST-2020.12.08.10.50.57/4000_test_metrics.csv')
actor_critic = pd.read_csv('./ContinuousMountainClimber-v0_actor_critic_2020.12.05.10.38.28/10000_test_metrics.csv')
with open('./MountainCarContinuous_PPO_Results_pkl_15104.data', 'rb') as f:
    ppo = pickle.load(f)
ppo = np.repeat(ppo[3], 30)

# Plot test data
plt.figure()
plt.plot(reinforce.episode, reinforce.test_reward_average_last_100)
plt.plot(actor_critic.episode, actor_critic.test_reward_average_last_100)
plt.plot(range(len(ppo)), ppo)
plt.xlabel('Training Iterations')
plt.ylabel('Mountain Car Continuous 100-Iteration Testing Reward')
plt.title('Testing Rewards for Continuous Mountain Car')
plt.legend(['REINFORCE', 'A2C', 'PPO'])
plt.savefig('../../final/mcc_testing_reward.png', figsize=[3,2])

# Read in training data
reinforce = pd.read_csv('./ReinforceTEST-2020.12.08.10.50.57/4000_training_metrics.csv')
actor_critic = pd.read_csv('./ContinuousMountainClimber-v0_actor_critic_2020.12.05.10.38.28/10000_training_metrics.csv')

# Plot training data
plt.figure()
plt.plot(reinforce.episode, reinforce.episode_moving_average_30)
plt.plot(actor_critic.episode, actor_critic.episode_moving_average_30)
plt.xlabel('Training Iterations')
plt.ylabel('Mountain Car Continuous 30-Iteration Training Reward')
plt.title('Training Rewards for Continuous Mountain Car')
plt.legend(['REINFORCE', 'A2C'])
plt.savefig('../../final/mcc_training_reward.png', figsize=[3,2])
