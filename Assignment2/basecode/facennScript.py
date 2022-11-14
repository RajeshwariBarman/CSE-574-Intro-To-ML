'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import numpy as np
import pickle
from scipy.optimize import minimize
from math import sqrt


def one_K_coding_scheme(training_label, n_class):
    size = np.size(training_label)
    out = np.zeros((size, n_class), dtype=int)
    for i in range(size):
        index = int(training_label[i])
        out[i][index] = 1
    return out


def feed_Forward_Func(training_data, w1, w2):
    data = training_data.transpose()

    data_bias = np.ones((1, np.size(data, 1)), dtype=int)

    data = np.concatenate((data, data_bias), axis=0)

    # Hidden Layer Start

    # Equation 1
    hidden_layer_intermediate = np.dot(w1, data)

    # Equation 2
    hidden_layer_output = sigmoid(hidden_layer_intermediate)

    # Hidden Layer End

    # Output Layer Start

    hidden_layer_bias = np.ones((1, np.size(hidden_layer_output, 1)), dtype= int)

    hidden_layer_output = np.concatenate((hidden_layer_output, hidden_layer_bias), axis=0)

    # Equation 3
    output_layer_intermediate = np.dot(w2, hidden_layer_output)

    # Equation 4
    output_layer_output = sigmoid(output_layer_intermediate)

    # Output Layer End

    return data, hidden_layer_output, output_layer_output



# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon
    return W



# Replace this with your sigmoid implementation
def sigmoid(z):
    Z = 1.0 / (1.0 + np.exp(-1.0 * z))
    return Z
    
# Replace this with your nnObjFunction implementation
def nnObjFunction(params, *args):
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    obj_val = 0
    obj_grad = np.array([])

    data_count = training_data.shape[0]

    # 1 of K Coding of labels Start
    labels = one_K_coding_scheme(training_label, n_class)
    # 1 of K Coding of labels End

    labels = labels.transpose()

    """******** Feed Forward Start ********"""
    # Equation 1, 2, 3 and 4
    data, hidden_layer_output, output_layer_output = feed_Forward_Func(training_data, w1, w2)

    """******** Feed Forward End ********"""

    """******** Bacpropagagtion Start ********"""
    # Equation 5
    log_likelihood_error_function = labels * np.log(output_layer_output) + (1 - labels) * np.log(
        1 - output_layer_output)

    # Equation 6 and 7
    log_likelihood_error = (-1) * (np.sum(log_likelihood_error_function[:]) / data_count)

    # Equation 8 and 9
    output_layer_delta = (output_layer_output - labels)

    w2_error = np.dot(output_layer_delta, hidden_layer_output.transpose())

    # Equation 10, 11 and 12
    hidden_layer_delta = np.dot(w2.transpose(), output_layer_delta) * (hidden_layer_output * (1 - hidden_layer_output))

    w1_error = np.dot(hidden_layer_delta, data.transpose())
    w1_error = w1_error[:-1, :]

    # Equation 15
    regularization_term = ((np.sum(w1 ** 2) + np.sum(w2 ** 2)) / (2 * data_count)) * lambdaval

    obj_val = log_likelihood_error + regularization_term

    # Equation 16 and 17
    w1_gradient = (w1_error + lambdaval * w1) / data_count
    w2_gradient = (w2_error + lambdaval * w2) / data_count

    """******** Bacpropagagtion End ********"""

    obj_grad = np.concatenate((w1_gradient.flatten(), w2_gradient.flatten()), axis=0)
    return (obj_val, obj_grad)


# Replace this with your nnPredict implementation
def nnPredict(w1,w2,data):
    labels = np.array([])
    # Your code here
    training_data, hidden_layer_output, output_layer_output = feed_Forward_Func(data, w1, w2)
    labels = np.argmax(output_layer_output, axis=0)
    return labels

# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

"""**************Neural Network Script Starts here********************************"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 25
# set the number of nodes in output unit
n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# set the regularization hyper-parameter
lambdaval = 20
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter' :50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
params = nn_params.get('x')
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#Test the computed parameters
predicted_label = nnPredict(w1,w2,train_data)
#find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,validation_data)
#find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,test_data)
#find the accuracy on Validation Dataset
print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
