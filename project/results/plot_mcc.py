#!/usr/bin/env python

import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Read in our test data
reinforce = pd.read_csv('./ReinforceTEST-2020.12.08.10.50.57/4000_test_metrics.csv')
actor_critic = pd.read_csv('./ContinuousMountainClimber-v0_actor_critic_2020.12.05.10.38.28/10000_test_metrics.csv')
with open('./MountainCarContinuous_PPO_Results_pkl_10112.data', 'rb') as f:
    ppo3 = pickle.load(f)
ppo3 = ppo3[3]

ppo = []
for l in ppo3:
   for item in l:
       ppo.append(item)

ppo = np.array(ppo)
ppo_means = []

for i in range(1,ppo.shape[0]):
    ppo_means.append(np.mean(ppo[i-100:i]))

# Plot test data
plt.figure()
plt.plot(reinforce.episode, reinforce.test_reward_average_last_100)
plt.plot(actor_critic.episode, actor_critic.test_reward_average_last_100)
plt.plot(range(len(ppo_means)), ppo_means)
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
