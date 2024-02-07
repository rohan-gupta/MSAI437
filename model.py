import numpy as np
import pandas as pd


def InitializeParameters(X, K):
	'''
	Input -> 
		1) X: Training Data
		2) K: Number of nodes in Hidden Layer

	Output ->
		1) dictionary of parameters
	'''
	input_layer_nodes = X.shape[0]
	hidden_layer_nodes = K
	output_layer_nodes = 1
	# hidden layer
	w1 = np.random.randn(hidden_layer_nodes, input_layer_nodes) * 0.01
	b1 = np.zeros(shape = (hidden_layer_nodes, 1))
	# output layer
	w2 = np.random.randn(output_layer_nodes, hidden_layer_nodes) * 0.01
	b2 = np.zeros(shape = (output_layer_nodes, 1))

	return {
	    'w1':w1,
	    'b1':b1,
	    'w2':w2,
	    'b2':b2
	}

def ApplySigmoid(x):
  return 1/(1+np.exp(-x))

def ApplyRelu(x):
  return x*(x>0)


def ForwardPropogation(X, parameters):
	'''
	Input ->
		1) X: training data
		2) parameters: dictionary of parameters

	Output ->
		1) Results from linear transformation & activation values
	'''
	w1, b1 = parameters['w1'], parameters['b1']
	w2, b2 = parameters['w2'], parameters['b2']

	# Hidden Layer
	# Activation -> Relu
	z1 = np.dot(w1, X) + b1
	a1 = ApplyRelu(z1)

	# Output Layer
	# Activation -> Sigmoid
	# to do // check activation -> throwing error here
	z2 = np.dot(w2, a1) + b2
	a2 = ApplySigmoid(z2)

	return {
	    'z1':z1,
	    'a1':a1,
	    'z2':z2,
	    'a2':a2
	}

def ComputeCost(preds, Y):
	m = Y.shape[1]
	epsilon = 1e-15
	preds = np.clip(preds, epsilon, 1-epsilon)

	cost = -1/m*np.sum(Y*np.log(preds) + (1-Y)*np.log(1-preds))
	return cost

def BackwardPropogation(parameters, forward_pass, X, Y):
	'''
	Input ->
		1) parameters: dicitonary of parameters
		2) forward_pass: dictionary of values after forward pass
		3) X: training data
		4) Y: targets

	Output ->
		1) dictionary of gradients of weights
	'''
	m = Y.size

	w1, b1 = parameters['w1'], parameters['b1']
	w2, b2 = parameters['w2'], parameters['b2']

	a1, a2 = forward_pass['a1'], forward_pass['a2']

	# Output Layer
	dz2 = a2 - Y
	dw2 = np.dot(dz2, a1.T)/m
	db2 = np.sum(dz2, axis = 1, keepdims = True)/m

	# Hidden Layer
	da1 = np.dot(w2.T, dz2)
	dz1 = da1 * (np.int64(a1>0))
	dw1 = np.dot(dz1, X.T)/m
	db1 = np.sum(dz1, axis = 1, keepdims = True)/m

	return {
	      'dw1':dw1,
	      'db1':db1,
	      'dw2':dw2,
	      'db2':db2
	  }

def UpdateParameters(parameters, correction, alpha):
	'''
	Inputs ->
		1) parameters: dictionary of parameters
		2) correction: gradient of weights after backprop
		3) alpha: learning rate

	Outputs ->
		1) parameters: replaces old weights with new weights
	'''
	w1, b1 = parameters['w1'], parameters['b2']
	w2, b2 = parameters['w2'], parameters['b2']

	dw1, db1 = correction['dw1'], correction['db1']
	dw2, db2 = correction['dw2'], correction['db2']
	#print(type(w1))
	#print(type(dw1))
	w1_new = w1 - alpha * dw1
	b1_new = b1 - alpha * db1

	w2_new = w2 - alpha * dw2
	b2_new = b2 - alpha * db2

	parameters['w1'], parameters['b1'] = w1_new, b1_new
	parameters['w2'], parameters['b2'] = w2_new, b2_new

	return parameters


def NeuralNetworkModel(X_train, Y_train, X_val, Y_val, K, iterations, alpha):
	'''
	Inputs ->
		1) X_train: x training data
		2) Y_train: y training data
		3) X_val: x validation data
		4) Y_val: y validation data
		5) K: number of nodes in hidden layer
		6) iterations: number of iterations
		7) alpha: learning rate

	Outputs ->
		1) parameters: final parameters of model after training
		2) train_loss_vals: loss values for training data (for every epoch)
		3) val_loss_vals: loss values for validation data (for every epoch)
	'''
	train_loss_vals = []
	val_loss_vals = []
	parameters = InitializeParameters(X_train, K)

	for _ in range(iterations):
		print("Epoch: ",_)

		forward_pass_train = ForwardPropogation(X_train, parameters)
		train_loss = ComputeCost(forward_pass_train['a2'], Y_train)
		train_loss_vals.append(train_loss)

		print(forward_pass_train['a2'])
		print(Y_train)
		
		forward_pass_val = ForwardPropogation(X_val, parameters)
		val_loss = ComputeCost(forward_pass_val['a2'], Y_val)
		val_loss_vals.append(val_loss)

		correction = BackwardPropogation(parameters, forward_pass_train, X_train, Y_train)
		parameters = UpdateParameters(parameters, correction, alpha)
		print(f"Training Loss: {train_loss}\tValidation Loss: {val_loss}\n\n")

	return parameters, train_loss_vals, val_loss_vals



def LoadData(train_path, test_path, val_path):
	'''
	Inputs ->
		1) train_path: path to training file
		2) test_path: path to test file
		3) val_path: path to validation file

	Outputs ->
		1) dictionary of data
	'''
	train = pd.read_csv(train_path)
	test = pd.read_csv(test_path)
	val = pd.read_csv(val_path)

	x_train, y_train = train.drop(['label'], axis = 1), train['label']
	x_test, y_test = test.drop(['label'], axis = 1), test['label']
	x_val, y_val = val.drop(['label'], axis = 1), val['label']

	return {
		'x_train':x_train,
		'y_train':y_train,
		'x_val':x_val,
		'y_val':y_val,
		'x_test':x_test,
		'y_test':y_test
	}

if __name__ == "__main__":
  # reading datas
	gaussian_data = LoadData('HW#1/two_gaussians_train.csv', 'HW#1/two_gaussians_test.csv', 'HW#1/two_gaussians_valid.csv')
	# getting appropriate splits
  x_train, y_train, x_val, y_val = gaussian_data['x_train'],gaussian_data['y_train'], gaussian_data['x_val'], gaussian_data['y_val']
	# converting data types
  y_train, y_val = np.asarray(y_train).reshape(y_train.size, 1), np.asarray(y_val).reshape(y_val.size,1)
	# running neural network model
  run = NeuralNetworkModel(x_train, y_train, x_val, y_val, K = 32, iterations = 50, alpha = 0.01)
