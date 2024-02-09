import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def InitializeParameters(X, K):
	'''
	Input -> 
		1) X: Training Data
		2) K: Number of nodes in Hidden Layer

	Output ->
		1) dictionary of parameters
	'''
	input_layer_nodes = X.shape[1]
	hidden_layer_nodes = K
	output_layer_nodes = 1
	# hidden layer
	w1 = np.random.randn(hidden_layer_nodes, input_layer_nodes) * 0.01
	b1 = np.zeros(shape = (1, hidden_layer_nodes))
	# output layer
	w2 = np.random.randn(output_layer_nodes, hidden_layer_nodes) * 0.01
	b2 = np.zeros(shape = (1, output_layer_nodes))

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
	z1 = np.dot(X, w1.T) + b1 
	a1 = ApplyRelu(z1)

	# Output Layer
	# Activation -> Sigmoid
	# to do // check activation -> throwing error here
	z2 = np.dot(a1, w2.T) + b2
	a2 = ApplySigmoid(z2)

	return {
	    'z1':z1,
	    'a1':a1,
	    'z2':z2,
	    'a2':a2
	}

def ComputeAccuracy(predictions, labels):
    '''
    Compute the accuracy of predictions.
    Inputs:
        - predictions: numpy array of model predictions
        - labels: numpy array of actual labels
    Output:
        - accuracy: float, the percentage of correct predictions
    '''
    preds_class = predictions > 0.5
    accuracy = np.mean(preds_class == labels)
    return accuracy

def ComputeCost(preds, Y):
	m = Y.shape[0]
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


	m = Y.shape[0]

	w1, b1 = parameters['w1'], parameters['b1']
	w2, b2 = parameters['w2'], parameters['b2']

	a1, a2 = forward_pass['a1'], forward_pass['a2']

	# Output Layer
	dz2 = a2 - Y
	dw2 = np.dot(dz2.T, a1) / m
	db2 = np.sum(dz2, axis=0, keepdims=True) / m

	# Hidden Layer
	da1 = np.dot(dz2, w2)
	dz1 = da1 * (a1 > 0)
	dw1 = np.dot(dz1.T, X) / m
	db1 = np.sum(dz1, axis=0, keepdims=True) / m

	return {
		'dw1': dw1,
		'db1': db1,
		'dw2': dw2,
		'db2': db2
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
	w1, b1 = parameters['w1'], parameters['b1']
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
	train_acc_vals, val_acc_vals = [], []  # Lists to store accuracy values

	parameters = InitializeParameters(X_train, K)

	for _ in range(iterations):
		print("Epoch: ",_)
		#Training
		forward_pass_train = ForwardPropogation(X_train, parameters)
		train_loss = ComputeCost(forward_pass_train['a2'], Y_train)
		train_acc = ComputeAccuracy(forward_pass_train['a2'], Y_train)  # Compute training accuracy
		train_acc_vals.append(train_acc)  
		train_loss_vals.append(train_loss)

		# print(forward_pass_train['a2'])
		# print(Y_train)
		#Validation
		forward_pass_val = ForwardPropogation(X_val, parameters)
		val_loss = ComputeCost(forward_pass_val['a2'], Y_val)
		val_acc = ComputeAccuracy(forward_pass_val['a2'], Y_val) 
		val_loss_vals.append(val_loss)
		val_acc_vals.append(val_acc)  

		correction = BackwardPropogation(parameters, forward_pass_train, X_train, Y_train)
		parameters = UpdateParameters(parameters, correction, alpha)
		print(f"Epoch {_}: Training Loss: {train_loss}, Training Accuracy: {train_acc}, Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")

	return parameters, train_loss_vals, val_loss_vals, train_acc_vals, val_acc_vals



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

def experiment_with_parameters(x_train, y_train, x_val, y_val, hidden_layer_sizes, learning_rates, iteration_counts):
    """
    Experiment with various hyperparameters to find the best configuration.
    """
    results = []

    for K in hidden_layer_sizes:
        for alpha in learning_rates:
            for iterations in iteration_counts:
                # Initialize parameters and train the model
                parameters, train_loss_vals, val_loss_vals, train_acc_vals, val_acc_vals = NeuralNetworkModel(
                    x_train, y_train, x_val, y_val, K=K, iterations=iterations, alpha=alpha
                )
                
                # Evaluate the model on the validation set
                final_val_acc = val_acc_vals[-1]  # Last accuracy value as the final accuracy
                final_val_loss = val_loss_vals[-1]  # Last loss value as the final loss

                # Store the results
                results.append({
                    'hidden_layer_size': K,
                    'learning_rate': alpha,
                    'iterations': iterations,
                    'final_val_acc': final_val_acc,
                    'final_val_loss': final_val_loss
                })

    # Sort results by validation accuracy, highest first
    results_sorted = sorted(results, key=lambda x: x['final_val_acc'], reverse=True)
    return results_sorted


def plot_decision_boundary(X, y, model, parameters):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()], parameters)
    Z = Z['a2']
    Z = Z > 0.5
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap=plt.cm.Spectral)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')


if __name__ == "__main__":
	# reading data
	# data = LoadData('HW1/data/center_surround_train.csv', 'HW1/data/center_surround_test.csv', 'HW1/data/center_surround_valid.csv')
	# data = LoadData('HW1/data/spiral_train.csv', 'HW1/data/spiral_test.csv', 'HW1/data/spiral_valid.csv')
	data = LoadData('HW1/data/two_gaussians_train.csv', 'HW1/data/two_gaussians_test.csv', 'HW1/data/two_gaussians_valid.csv')
	# data = LoadData('HW1/data/xor_train.csv', 'HW1/data/xor_test.csv', 'HW1/data/xor_valid.csv')
	# getting appropriate splits
	x_train, y_train, x_val, y_val = data['x_train'],data['y_train'], data['x_val'], data['y_val']
	# converting data types
	y_train, y_val = np.asarray(y_train).reshape(y_train.size, 1), np.asarray(y_val).reshape(y_val.size,1)
	# Find best Param
	hidden_layer_sizes = [4, 8, 16, 32, 64]
	learning_rates = [0.1, 0.01, 0.001, 0.0001]
	iteration_counts = [100, 500, 1000]
	best_param = experiment_with_parameters(x_train, y_train, x_val, y_val, hidden_layer_sizes, learning_rates, iteration_counts)[0]

	# Run model on best param 
	parameters, train_loss_vals, val_loss_vals, train_acc_vals, val_acc_vals = NeuralNetworkModel(x_train, y_train, x_val, y_val, K=best_param["hidden_layer_size"], iterations=best_param["iterations"], alpha=best_param["learning_rate"])

	x_test = data["x_test"]
	y_test = data["y_test"]
	plot_decision_boundary(x_test, y_test, ForwardPropogation, parameters)
	plt.show()

	# Plotting the training and validation loss
	plt.figure(figsize=(10, 5))
	plt.subplot(1, 2, 1)
	plt.plot(train_loss_vals, label='Training Loss')
	plt.plot(val_loss_vals, label='Validation Loss')
	plt.title('Loss over iterations')
	plt.xlabel('Iterations')
	plt.ylabel('Loss')
	plt.legend()

	# Plotting the training and validation accuracy
	plt.subplot(1, 2, 2)
	plt.plot(train_acc_vals, label='Training Accuracy')
	plt.plot(val_acc_vals, label='Validation Accuracy')
	plt.title('Accuracy over iterations')
	plt.xlabel('Iterations')
	plt.ylabel('Accuracy')
	plt.legend()

	plt.tight_layout()
	print(best_param)
	plt.show(block=True)

