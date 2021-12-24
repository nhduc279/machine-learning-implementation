import yaml
import time
import torch
import torch.nn.functional as F

# read prepared hyperameters
with open('settings.yaml', mode='r', encoding='utf8') as settings_file:
	SETTINGS = yaml.load(settings_file, Loader=yaml.FullLoader)

# determine whether the system support CUDA or not
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


#####################################################
#### Implement Logistic Regression using PyTorch API 
#####################################################

cost, minibatch_cost = [], [] 
start_time = time.time()

class LogisticRegression(torch.nn.Module): 

	"""
	Logistic Regression for binary classification
	"""

	def __init__(self, num_features):
		super(LogisticRegression, self).__init__()

		# attributes
		self.cost_ = cost
		self.minibatch_cost_ = minibatch_cost

		# parameters
		self.linear = torch.nn.Linear(in_features=num_features, out_features=1)
		self.linear.weight.detach().zero_()
		self.linear.bias.detach().zero_()


	# forward probagation
	def forward(self, x):
		logits = self.linear(x)
		probas = torch.sigmoid(logits)
		return probas


	### Training ### 
	def fit(self, x, y, num_epochs=SETTINGS['NUM_EPOCHS'], 
				minibatch_size=SETTINGS['MINIBATCH_SIZE'], 
				seed=SETTINGS['RANDOM_SEED']):

		# use stochastic gradient descent as an optimizer
		optimizer = torch.optim.SGD(self.parameters(), lr=SETTINGS['LEARNING_RATE'])

		# shuffle idx
		shuffle_idx = torch.randperm(y.size(0), dtype=torch.long)
		minibatches = torch.split(shuffle_idx, minibatch_size)
		torch.manual_seed(seed)

		for e in range(num_epochs):

			for minibatch_idx in minibatches:
			# comp. probas
				probas = self.forward(x[minibatch_idx])

				# comp. loss
				loss = F.binary_cross_entropy(probas, y[minibatch_idx].view(-1, 1), reduction='mean')

				# reset gradients from previous round
				optimizer.zero_grad()

				# comp. gradients
				loss.backward()

				# model updates
				optimizer.step()

		
				minibatch_cost.append(loss)
				self.cost = minibatch_cost


			### Logging ###
			with torch.no_grad():  # context manager, to save memory during inference
				probas = self.forward(x)
				curr_loss = F.binary_cross_entropy(probas, y.view(-1, 1))
				accuracy = self.evaluate(x, y)
				print('Epoch: %03d' %(e+1), end='')
				print(' | ACC train: %.3f' %accuracy, end='')
				print(' | Loss: %.3f' %curr_loss, end='')
				print(' | Time elapsed: %.2f min' % ((time.time() - start_time) / 60))
				cost.append(curr_loss)
				self.cost_ = cost
		print('\n - Total training time: %.2f min' %((time.time() - start_time) / 60))



	### Evaluate ###
	def evaluate(self, x, y):
	    y_pred_probas = self.forward(x).view(-1)
	    y_pred = torch.where(y_pred_probas > .5, 1, 0)
	    accuracy = torch.sum(y_pred==y) / y.size(0)
	    return accuracy


	### Make predictions ###
	def predict(self, x):
	    y_pred_probas = self.forward(x).view(-1)
	    y_pred = torch.where(y_pred_probas > .5, 1, 0)
	    return y_pred
