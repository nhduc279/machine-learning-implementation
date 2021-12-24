import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from my_model import LogisticRegression
from helper import plot_training_loss, plot_minibatch_loss, compute_accuracy


###################
#### Data Prep.
###################


# load training data
df = pd.read_table('data\\toy.txt', names=['feature_1', 'feature_2', 'target'])

# prepare X (features) and y (target)
X = df[['feature_1', 'feature_2']]
y = df['target']


# determine whether the system support CUDA or not
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# split data for training and testing (train:60, test:30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# convert numpy arrays to pytorch tensors
X_train = torch.tensor(X_train.values, dtype=torch.float32, device=DEVICE)
X_test = torch.tensor(X_test.values, dtype=torch.float32, device=DEVICE)
y_train = torch.tensor(y_train.values, dtype=torch.float32, device=DEVICE)
y_test= torch.tensor(y_test.values, dtype=torch.float32, device=DEVICE)


#################################
### Model Training 
#################################


# model fitting
logreg = LogisticRegression(num_features=X_train.size(1))
logreg.to(DEVICE)
logreg.fit(X_train, y_train)


# model evaluation on the test set
y_pred = logreg.predict(X_test)
test_accuracy = compute_accuracy(y_test, y_pred)
print(f' - Test Acc: {test_accuracy:.3f}')


# plotting the loss
plot_training_loss(logreg.cost_)
plot_minibatch_loss(logreg.minibatch_cost_)
