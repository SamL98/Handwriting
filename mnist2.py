import numpy as np
from mnist import load_data
import itertools
import collections

# Dimensionality constants
N = 50000
T = 10000
V = 10000
D = 784
h1 = 20
h2 = 20
m = 10

#np.random.seed(0)
np.set_printoptions(threshold=np.nan)

# Initializing of training variables
X = np.zeros((N, D))
y = np.zeros((N, 1))

X_valid = np.zeros((V, D))
y_valid = np.zeros((V, 1))

X_test = np.zeros((T, D))
y_test = np.zeros((T, 1))

"""Define all non-linear activation functions"""

# Sigmoid (logistic activation function)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Derivative of the sigmoid function
def dsigmoid(y):
    return np.multiply(y, (1 - y))

# Softmax activation function
def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

"""Define the base layer (linear or non-linear) with virtual functions"""
class Layer(object):
    # Return an iterator of the parameters
    def get_params_iter(self):
        return []

    # Return a list of the gradients over the parameters of the layer
    def get_params_grad(self, X, output_grad):
        return []

    # Feed the input forward through the layer
    def get_output(self, X):
        pass

    # Return the input of the gradient of the layer
    def get_input_grad(self, Y, output_grad=None, T=None):
        pass

"""Define the fully-connectede linear layer"""
class LinearLayer(Layer):
    # Initialize weights
    def __init__(self, n_in, n_out):
        self.W = np.random.randn(n_in, n_out) * 0.1
        self.b = np.zeros(n_out)

    def get_params_iter(self):
        return itertools.chain(np.nditer(self.W, op_flags=['readwrite']),
                                np.nditer(self.b, op_flags=['readwrite']))

    def get_output(self, X):
        return np.dot(X, self.W) + self.b

    def get_params_grad(self, X, output_grad):
        JW = np.dot(X.T, output_grad)
        Jb = np.sum(output_grad, axis=0)
        return [g for g in itertools.chain(np.nditer(JW), np.nditer(Jb))]

    def get_input_grad(self, Y, output_grad):
        return np.dot(output_grad, self.W.T)

"""Define the logistic activation layer"""
class LogisticLayer(Layer):
    def get_output(self, X):
        return sigmoid(X)

    def get_input_grad(self, Y, output_grad):
        return np.multiply(dsigmoid(Y), output_grad)

"""Define the softmax output layer"""
class SoftmaxLayer(Layer):
    def get_output(self, X):
        return softmax(X)

    def get_input_grad(self, Y, T):
        return (Y - T) / Y.shape[0]
        #for i in range(T.shape[0]):
        #    Y[i][T[i]] -= 1
        #return Y / Y.shape[0]

    def get_cost(self, Y, T):
        return -np.mean(np.multiply(T, np.log(Y)))
        #return -np.mean(np.log(Y))

# Feed the input forward through the given layers
def feed_forwards(input_samples, layers):
    activations = [input_samples]
    X = input_samples
    for layer in layers:
        Y = layer.get_output(X)
        activations.append(Y)
        X = activations[-1]
    return activations

# Propagate the output backwards through the given layers
def feed_backwards(activations, targets, layers):
    param_grads = collections.deque()
    output_grad = None

    for layer in reversed(layers):
        Y = activations.pop()
        if output_grad is None:
            input_grad = layer.get_input_grad(Y, targets)
        else:
            input_grad = layer.get_input_grad(Y, output_grad)
        X = activations[-1]
        grads = layer.get_params_grad(X, output_grad)
        param_grads.appendleft(grads)
        output_grad = input_grad

    return list(param_grads)

# Update the params of the given layers based on their gradients
def update_params(layers, param_grads, learning_rate):
    for layer, layer_backprop_grads in zip(layers, param_grads):
        for param, grad in zip(layer.get_params_iter(), layer_backprop_grads):
            param -= learning_rate * grad

import random
# Shuffle the training batches
def shuffle(x, y):
    indeces = [i for i in range(y.shape[0])]
    random.shuffle(indeces)
    xtmp, ytmp = np.zeros(x.shape), np.zeros(y.shape)
    i = 0
    for index in indeces:
        xtmp[i] = x[index]
        ytmp[i] = y[index]
        i += 1
    return xtmp, ytmp

# Fit the given layers to the input and labels
def fit(X_train, y_train, X_valid, y_valid, layers):
    accuracies = []

    max_iter = 10
    learning_rate = 0.35

    for iteration in range(max_iter):
        i = 0
        batches = zip(np.array_split(X_train, n_batches, axis=0),
                      np.array_split(y_train, n_batches, axis=0))
        for x_batch, y_batch in batches:
            x_batch, y_batch = shuffle(x_batch, y_batch)
            activations = feed_forwards(x_batch, layers)
            prediction = activations[-1]
            minibatch_cost = layers[-1].get_cost(activations[-1], y_batch)
            minibatch_costs.append(minibatch_cost)
            param_grads = feed_backwards(activations, y_batch, layers)
            update_params(layers, param_grads, learning_rate)
            accuracies.append(accuracy(prediction, y_batch))

            if i % 100 == 0:
                print("Epoch %d, minibatch %d" % (iteration, i))
                print("Cost {}".format(minibatch_costs[-1]))
                print("Accuracy: {}%".format(accuracy(prediction, y_batch)*100))

            i += 1

        activations = feed_forwards(X_train, layers)
        train_cost = layers[-1].get_cost(activations[-1], y_train)
        training_costs.append(train_cost)

        activations = feed_forwards(X_valid, layers)
        validation_cost = layers[-1].get_cost(activations[-1], y_valid)
        validation_costs.append(validation_cost)

        print("Validation cost: ", validation_cost)
        if len(validation_costs) >= 3:
            if np.mean(validation_costs[-3:]) >= 0.975:
                print("Model is good enough, halting training.")
                return

        #if len(validation_costs) > 3:
        #    if validation_costs[-1] >= validation_costs[-2] >= validation_costs[-3]:
        #        break

    iterations = iteration + 1

def accuracy(pred, exp):
    num_correct = 0
    for x, y in zip(pred, exp):
        if np.argmax(x) == np.argmax(y):
            num_correct += 1
    return num_correct/exp.shape[0]

def predict(X, layers):
    activations = feed_forwards(X, layers)
    return np.argmax(activations[-1])

def test(X, y, layers):
    activations = feed_forwards(X, layers)
    pred = activations[-1]
    return accuracy(pred, y)

def construct_network():
    layers = []
    layers.append(LinearLayer(D, h1))
    layers.append(LogisticLayer())
    layers.append(LinearLayer(h1, h2))
    layers.append(LogisticLayer())
    layers.append(LinearLayer(h2, m))
    layers.append(SoftmaxLayer())
    return layers

import matplotlib.pyplot as plt

iterations = 0
batch_size = 25
n_batches = 0

minibatch_costs = []
training_costs = []
validation_costs = []

def convert_to_onehot(T):
    tmp = np.zeros((len(T), m))
    tmp[np.arange(len(T)), T] = 1
    return tmp

def save_network(layers):
    with open("mnist_weights2.txt", "w") as f:
        i = 0
        for layer in layers:
            if type(layer) is LinearLayer:
                w_file, b_file = ('W{}.txt'.format(i), 'b{}.txt'.format(i))
                np.savetxt(w_file, layer.W)
                np.savetxt(b_file, layer.b)
                f.write("FC\n{}\n{}\n".format(w_file, b_file))
            elif type(layer) is LogisticLayer:
                f.write("Activation\nfunc\n{}\n".format("sigmoid"))
            elif type(layer) is SoftmaxLayer:
                f.write("Output\nfunc\n{}\n".format("softmax"))
            
            i += 1

if __name__ == '__main__':
    X_train, t_train, X_valid, t_valid, X_test, t_test = load_data()
    y_train, y_valid, y_test = (convert_to_onehot(t_train), 
                                convert_to_onehot(t_valid), 
                                convert_to_onehot(t_test))

    n_batches = X_train.shape[0] // batch_size
    layers = construct_network()
    fit(X_train, y_train, X_valid, y_valid, layers)

    acc = test(X_test, y_test, layers)
    if acc > 0.75:
        save_network(layers)
    print("Model is finished with accuracy: {}%".format(acc*100))

    # Plot the minibatch, full training set, and validation costs
    minibatch_x_inds = np.linspace(0, len(minibatch_costs), num=len(minibatch_costs))
    training_x_inds = np.linspace(0, len(training_costs), num=len(training_costs))
    validation_x_inds = np.linspace(0, len(validation_costs), num=len(validation_costs))
    # Plot the cost over the iterations
    plt.plot(minibatch_x_inds, minibatch_costs, 'k-', linewidth=0.5, label='cost minibatches')
    plt.plot(training_x_inds, training_costs, 'r-', linewidth=2, label='cost full training set')
    plt.plot(validation_x_inds, validation_costs, 'b-', linewidth=3, label='cost validation set')
    # Add labels to the plot
    plt.xlabel('iteration')
    plt.ylabel('$\\xi$', fontsize=15)
    plt.title('Decrease of cost over backprop iteration')
    plt.legend()
    x1,x2,y1,y2 = plt.axis()
    plt.axis((0,iterations,0,2.5))
    plt.grid()
    plt.show()
