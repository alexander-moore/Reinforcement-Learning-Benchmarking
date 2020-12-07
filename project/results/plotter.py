import torch
data1 = torch.load('C:/Users/Alex/Documents/cs525/project/results/dqn_training_car.pt')
data2 = torch.load('C:/Users/Alex/Documents/cs525/project/results/ddqn_training_car.pt')
data3 = torch.load('C:/Users/Alex/Documents/cs525/project/results/dueling_dqn_training_car.pt')

import matplotlib.pyplot as plt

def plot_it(input):
	plt.plot(input)
	plt.title('Reward vs. Episode')
	plt.xlabel('Training Episodes')
	plt.ylabel('Mean Reward over 100 Episodes')
	plt.show()

plot_it(data1)
plot_it(data2)
plot_it(data3)