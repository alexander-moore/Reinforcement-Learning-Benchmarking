#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque, namedtuple
import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from agent import Agent
from ppo_model import PPO
from dqn_model import DQN

import pickle

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)


class AgentPPO(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example: 
            parameters for neural network
            initialize Q net and target Q net
            parameters for replay buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """
        # parameters For PPO.
        #
        # # Number of episodes to train.
        # self.numTrainEpisodes = 35000
        # # Replay buffer size.
        # self.replayBufferSize = 15000
        # # Learning rate. This is same as alpha.
        # self.learningRate = 0.00005
        # Discount factor.
        self.gamma = 0.99
        # # Minimum buffer size before starting to train DQN.
        # self.trainStart = 5000
        # # Interval for updating target DQN (# of steps).
        # self.targetUpdate = 5000
        # Min-batch size
        self.miniBatchSize = 32
        # Number of actions.
        self.numActions = 4
        # # Decaying epsilon parameters.
        # self.epsilonMax = 1.0
        # self.epsilonMin = 0.05
        # self.epsilonStep = 1000000
        # # Compute epsilon decay for each step.
        # self.epsilonDecay = ((self.epsilonMax - self.epsilonMin) / self.epsilonStep)
        # Learning rate decay steps (# of "episodes" -- frameIdx/num_step)
        self.lr = 1.5e-4
        self.lrStep = 10000
        self.lrMin = 0.000025

        # Initialize loss.
        self.allLosses = []
        self.allEntropy = []
        # # Initialize epsilon.
        # self.epsilon = self.epsilonMax
        # Initialize replay buffer.
        # self.memory = []
        # Initialize device.
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # # Initialize DQNs.
        # self.Q = PPO().float().to(self.dev)
        # self.TargetQ = PPO().float().to(self.dev)
        # Initialize PPO.
        self.model = PPO().float().to(self.dev)
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr)

        super(AgentPPO, self).__init__(env)
        ###########################
        # Define action if in test mode.

        if args.test_dqn:
            # You can load your model here (when in test mode):
            print('Loading trained model: ')

            # Hard-code file name of model for testing.
            # modelName = './Breakout_DDQN_Best_Model_v7.pth'

            # Use this if parsing an argument for file name of the model to test.
            # if args.modelName == 'none':
            #     modelName = './Breakout_DDQN_Best_Model_v7.pth'
            # else:
            #     modelName = args.modelName

            # print('%s' % modelName)

            # This is required when running test on a model that was trained on a different device.
            # self.Q = PPO().float().to(self.dev)
            # if self.dev.type == 'cpu':
            #     self.Q.load_state_dict(torch.load(modelName, map_location=self.dev))
            # else:
            #     self.Q.load_state_dict(torch.load(modelName))
            #
            # self.Q.eval()

    def init_game_setting(self):
        """
        Testing function will call this function at the beginning of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # No parameters will be initialized here.
        ###########################
        pass

    def make_action(self, observation, test=False):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        ###########################
        # If a random number is less than the current epsilon and is NOT in test mode,
        # then randomly select an action. Otherwise, act greedily and go with max Q
        # value based on inline-network/DQN.
        # if np.random.rand() <= self.epsilon and not test:
        #     return random.randrange(self.numActions)
        # else:
        #     with torch.no_grad():
        #         observation = (np.float32(observation) / 255.0).transpose((2, 0, 1))
        #         observation = torch.from_numpy(observation).unsqueeze(0)
        #         action = self.Q(observation.to(self.dev)).argmax(dim=1).item()
        
        ###########################
        # return action
        pass
    
    def push(self, transition):
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.
        
        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        ###########################
        # Append the transition to the replay buffer if the buffer is not full.
        # Otherwise, remove the the oldest transition and then append the new one.
        # Note that .pop(0) doesn't work because self.memory is a deque and NOT a list.
        # if len(self.memory) < self.replayBufferSize:
        #     self.memory.append(transition)
        # else:
        #     self.memory.popleft()
        #     self.memory.append(transition)
        pass

    def replay_buffer(self):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """
        ###########################
        # Retrieve a mini-batch of a specified # of transitions from the buffer randomly

        ###########################
        # return minibatch
        pass

    def testEnv(self, env):
        """
        Define an internal test mode to monitor total reward during training.
        """
        state = env.reset()
        # Added the following two lines.
        # state = (np.float32(state) / 255.0).transpose((2, 0, 1))
        # state = torch.from_numpy(state).unsqueeze(0)
        state = torch.FloatTensor(state)
        done = False
        totalReward = 0
        while not done:
            with torch.no_grad():
                # Modified the following line.
                # state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                state = torch.FloatTensor(state).to(self.dev)
                self.model.eval()
                dist, _ = self.model(state.to(self.dev))

                # Instead of just sampling from the distribution, use argmax() during test.
                # print(dist.sample().numpy().squeeze())
                # print(dist.probs.detach().numpy())
                # print(dist.probs.detach().cpu().numpy().argmax().item())
                # nextState, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
                nextState, reward, done, _ = env.step(torch.tanh(dist.sample()).numpy())
                # nextState = (np.float32(nextState) / 255.0).transpose((2, 0, 1))
                # nextState = torch.from_numpy(nextState).unsqueeze(0)
                state = nextState
                totalReward += reward

        return totalReward

    def GAE(self, nextValue, rewards, masks, values, gamma=0.99, tau=0.95):
        """
        Compute Generalized Advantage Estimation.
        """
        values = values + [nextValue]
        gae = 0
        returns = []
        # print(nextValue)
        # print(len(rewards))
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            # print(delta)
            gae = delta + gamma * tau * masks[step] * gae
            # print(gae)
            # print(values[step])
            returns.insert(0, gae + values[step])

        return returns

    def ppoIter(self, mini_batch_size, states, actions, log_probs, returns, advantage):
        batch_size = states.size(0)

        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, mini_batch_size)
            # print(rand_ids)
            # actions and log_probs are only one dimensional tensors, so the original code doesn't make sense for
            # those two variables. Could have unsqueezed it in the main code.
            # yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :
            #                                                                          ], advantage[rand_ids, :]
            yield states[rand_ids, :], actions[rand_ids], log_probs[rand_ids], returns[rand_ids], advantage[
                                                                                                     rand_ids]

    def ppoUpdate(self, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
        for _ in range(ppo_epochs):
            for state, action, old_log_probs, return_, advantage in self.ppoIter(mini_batch_size, states, actions,
                                                                                 log_probs, returns, advantages):
                dist, value = self.model(state)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()

                # Calculate surrogate function.
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

                actor_loss = - torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()

                loss = 0.5 * critic_loss + actor_loss - 0.01 * entropy  # Changed from 0.001 to 0.01

                # Save computed losses and entropy.
                self.allLosses.append(loss.clone().detach().numpy())
                self.allEntropy.append(entropy.clone().detach().numpy())

                # print(loss.detach().numpy())

                # optimizer.zero_grad()
                self.optimizer.zero_grad()
                loss.backward()
                # Try clipping the gradient. Got an error due to vanishing and or exploding gradient Prob < 0. It
                # doesn't work to solve that issue. Furthermore, it tends to cause test rewards/scores to plateau at
                # 0.4.
                # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=100.0)
                # optimizer.step()
                self.optimizer.step()

    def train(self):
        # Change env to envs to match code.
        envs = self.env

        # Comment out the two lines below since the # of channels for Breakout should be 4 and # of action is also 4.
        # num_inputs = envs.observation_space.shape[0]
        # num_outputs = envs.action_space.shape[0]

        device = self.dev

        # Hyper-parameters (will add to super)
        # hidden_size = 256
        # Learning rate.
        lr = self.lr
        # Max number of steps to take for each episode. Or is it # of steps of the environment per update? 1024
        num_steps = 250     # was 128, 500 in several cases
        # Number of training mini-batches per update. Typically 4.
        mini_batch_size = 5     # was 4, 5 in one case
        # How about the batch size used during learning? It's equal to number of steps of the environment in this
        # case (num_steps).
        # Number of training epochs per update.
        ppo_epochs = 4     # was 4 10 in one case
        # Reward threshold for auto-stop.
        threshold_reward = 90

        # Comment out original model name (ActorCritic) and remove input parameters.
        # model = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)
        self.model = PPO().float().to(device)
        # optimizer = optim.Adam(model.parameters(), lr=lr)
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr)

        # Maximum number of frames to run (total # of actions to take in the environment).
        max_frames = 10000
        frame_idx = 0
        test_rewards = []
        testReward = 0.
        testRewardMean = 0.
        allRewards = []
        iterSteps = []
        # allEntropy = []

        # state will be an array of size 2 (1x2) with the first being the position and second
        # being the velocity.
        state = envs.reset()

        # Won't need the following statements.
        # state = (np.float32(state) / 255.0).transpose((2, 0, 1))
        # state = torch.from_numpy(state).unsqueeze(0)
        early_stop = False

        # print(state.shape)

        while frame_idx < max_frames and not early_stop:

            log_probs = []
            values = []
            states = []
            actions = []
            rewards = []
            masks = []
            entropy = 0

            # # Try learning rate annealing.
            # lr = lr - ((frame_idx / num_steps) / self.lrStep)
            # if lr < self.lrMin:
            #     lr = self.lrMin
            #
            # # Reset optimizer with new learning rate.
            # self.optimizer = optim.Adam(params=self.model.parameters(), lr=lr)

            for _ in range(num_steps):
                # Not sure if I need to the following line since it's already been converted to torch tensor above.
                state = torch.FloatTensor(state).to(device)
                dist, value = self.model(state)

                action = dist.sample()

                # Need to keep action within the -1 to 1 bound. Two methods, use tanh or clamp.
                action = torch.tanh(action)
                action = torch.clamp(action, -1.0, 1.0)

                if np.isnan(action):
                    print('True')
                    print(action)
                    # action = torch.tensor([0.5])

                # In this case, do not squeeze. Needs to be a numpy ndarray.
                # next_state, reward, done, _ = envs.step(action.cpu().numpy())
                next_state, reward, done, _ = envs.step(action.cpu().numpy())

                # Added the following lines to normalize and convert state/next_state
                # next_state = (np.float32(next_state) / 255.0).transpose((2, 0, 1))
                # next_state = torch.from_numpy(next_state).unsqueeze(0)

                log_prob = dist.log_prob(action)
                entropy += dist.entropy().mean()

                log_probs.append(log_prob)
                values.append(value)

                # Save all rewards for each iteration.
                iterSteps.append(frame_idx)
                allRewards.append(reward)
                # allEntropy.append(entropy.clone().detach().numpy())

                # Need to convert reward to list.
                # rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
                rewards.append(torch.FloatTensor([reward]).unsqueeze(1).to(device))
                # Need the square
                # bracket because it needs a list. Otherwise, one can get all sorts of numbers.
                masks.append(torch.FloatTensor([1 - done]).unsqueeze(1).to(device))

                # states.append(state)
                states.append(state.numpy()[np.newaxis, :])
                # np.append([states], [state.numpy()])

                actions.append(action)

                state = next_state
                frame_idx += 1

                if frame_idx % 30 == 29:
                    # Modified the following line.
                    for _ in range(30):
                        testReward = self.testEnv(envs)
                        testRewardMean = np.mean([testReward])

                    # test_reward = np.mean([self.testEnv(envs) for _ in range(30)])
                    # if episode % 10 == 9:  # Print every 10th episode
                    # print('[%d] Mean Reward (last 30 frames): %3.3f' % (frame_idx + 1, test_reward))
                    print('[%d] Mean Reward (last 30 frames): %3.3f' % (frame_idx + 1, testRewardMean))
                    # print(test_reward)
                    test_rewards.append([testReward])
                    # plot(frame_idx, test_rewards)
                    if testRewardMean > threshold_reward:
                        # Added the following line:
                        print(testRewardMean)
                        early_stop = True

            # Not sure if I need to the following line since it's already been converted to torch tensor above.
            next_state = torch.FloatTensor(next_state).to(device)
            # next_state = next_state.to(device)
            _, next_value = self.model(next_state)

            # returns = compute_gae(next_value, rewards, masks, values)
            returns = self.GAE(next_value, rewards, masks, values)

            states = np.stack(states).squeeze()

            # Need to either squeeze returns or unsqueeze values to get advantage to have the same number of rows.
            # Choose to squeeze return. Less memory intensive if we avoid using torch.cat.
            # returns = torch.cat(returns).detach()
            # returns = torch.cat(returns).detach().squeeze()
            returns = torch.tensor(returns)
            # print(returns.size())
            # log_probs = torch.cat(log_probs).detach()
            log_probs = torch.tensor(log_probs)
            # print(log_probs.size())
            # values = torch.cat(values).detach()   #   Pick this one.
            # values = torch.cat(values).unsqueeze(1).detach()
            values = torch.tensor(values)
            # print(values.size())
            states = torch.tensor(states)
            # actions = torch.cat(actions)
            actions = torch.tensor(actions)
            # print(actions.size())
            advantage = returns - values
            # print(advantage.size())
            # Any need to standardize the advantage? Some implementations do that.
            advantage = (advantage - advantage.mean()) / advantage.std()

            self.ppoUpdate(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage)

        outFile = np.array([iterSteps, self.allLosses, self.allEntropy, test_rewards]).transpose()
        # print(np.max(iterSteps))
        # print(len(self.allLosses))
        with open('MountainCarContinuous_PPO_Results_pkl_' + str(np.max(iterSteps)+1) + '.data',
                  'wb') as fid:
            pickle.dump(outFile, fid)

        # Save both networks/models.
        torch.save(self.model.state_dict(), './MountainCarContinuous_PPO' + '_Model_' + str(np.max(iterSteps)+1) +
                   '.pth')
        # torch.save(self.TargetQ.state_dict(), './Breakout_DDQN_' + str(devName) + '_TargetModel_' + str(epName) + '_'
        #            + str(step) + '.pth')

        ###########################
