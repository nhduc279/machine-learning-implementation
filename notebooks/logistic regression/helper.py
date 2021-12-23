import torch
import matplotlib.pyplot as plt

### Plotting the loss ###
def plot_training_loss(cost):
	plt.plot(range(len(cost)), cost)
	plt.xlabel('Epoch')
	plt.ylabel('Loss: Negative-Log Likelihood')
	plt.show()

### Plotting the loss ###
def plot_minibatch_loss(minibatch_cost):
	plt.plot(range(len(minibatch_cost)), minibatch_cost)
	plt.xlabel('Epoch')
	plt.ylabel('Binary Cross Entropy')
	plt.show()

### compute accuracy score ###
def compute_accuracy(y_pred, y_true):
	return torch.sum(y_pred==y_true) / y_true.shape[0]
