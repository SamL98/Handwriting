import numpy as np
import gzip
import pickle

# Dimensionality constants
N = 50000
T = 10000
D = 784
h = 10
m = 10

np.random.seed(0)

# Initializing of training variables
X = np.zeros((N, D))
y = np.zeros((N, 1))

X_test = np.zeros((T, D))
y_test = np.zeros((T, 1))

"""Load the MNIST data (training and test sets)"""
def load_data():
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        train, valid, test = pickle.load(f, fix_imports=True, encoding='latin1')
        return train[0], train[1], valid[0], valid[1], test[0], test[1]

"""Neural Network simulating an XOR gate."""
class MLP:
    def __init__(self, n_in, n_h, n_out):
        self.W1 = np.random.randn(n_in, n_h) * 0.1
        self.b1 = np.zeros((1, h))

        self.W2 = np.random.randn(n_h, n_out) * 0.1
        self.b2 = np.zeros((1, n_out))

    """Sigmoid activation function."""
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    """Derivative of the sigmoid activation function."""
    def sigmoid_prime(self, z):
        exp_z = np.exp(-z)
        return exp_z/((1+exp_z)**2)

    """Softmax activation function"""
    def softmax(self, z):
        e_z = np.exp(z - np.max(z))
        return e_z / e_z.sum()

    """Propagate the given X and y forward through the network."""
    def for_prop(self, X, y):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2

        self.num_examples = y.shape[0]

        self.probs = self.softmax(self.z2)
        self.correct_probs = np.zeros((self.num_examples,))
        for i in range(self.num_examples):
            self.correct_probs[i] = self.probs[i][y[i]]

        return self.correct_probs

    """Propagate the given probabilities backwards through the network."""
    def back_prop(self, scores, X, l1=0):
        data_loss = np.mean(-np.log(self.correct_probs))
        reg_loss = 0.5*l1*np.mean(self.W1*self.W1) + 0.5*l1*np.mean(self.W2*self.W2)
        self.loss = data_loss+reg_loss

        dloss = self.probs
        for i in range(self.num_examples):
            dloss[i, int(y[i])] -= 1
        dloss /= self.num_examples

        dW2 = np.dot(self.a1.T, dloss)
        db2 = np.sum(dloss, axis=0, keepdims=True)

        dz1 = np.dot(dloss, self.W2.T)
        dz1 = self.sigmoid_prime(dz1)

        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        dW2 += l1 * self.W2
        dW1 += l1 * self.W1

        return dW1, db1, dW2, db2

    """Train the network given the X and y."""
    def fit(self, X, y, epochs=100, epsilon=0.2, l1=0.5, minibatches=20):
        batch_size = y.shape[0] // minibatches
        for epoch in range(epochs):
            for i in range(minibatches):
                X_batch = X[i*batch_size:(i+1)*batch_size,:]
                y_batch = y[i*batch_size:(i+1)*batch_size]

                y_hat = self.for_prop(X_batch, y_batch)
                dW1, db1, dW2, db2 = self.back_prop(y_hat, X_batch, l1=l1)

                if i % 10 == 0:
                    print("Accuracy: {0}%".format(self.accuracy(y_batch, np.argmax(self.probs[:10], axis=1))*100))
                    print("Actual: ", y_batch[:10])
                    print("Predicted: ", np.argmax(self.probs[:10], axis=1))
                    print("Probs: ", self.correct_probs[:10][:3])

                    if epoch % 10 == 0:
                        print("--- Epoch {}, Minibatch {} ---".format(epoch, i))
                        print("Loss: ", self.loss)

                self.W1 += -epsilon * dW1
                self.b1 += -epsilon * db1
                self.W2 += -epsilon * dW2
                self.b2 += -epsilon * db2

    """Test the network on the given X and y."""
    def test(self, X, y):
        _ = self.for_prop(X, y)
        y_pred = np.argmax(self.probs, axis=1)
        print("Predicted: ", y_pred)
        print("Actual: ", y)
        print("Accuracy: {0}%".format(self.accuracy(y, y_hat)*100))
        if self.accuracy(y, y_hat) > 0.75:
            print(self.W1)
            print(self.b1)
            print(self.W2)
            print(self.b2)

    """Determine the accuracy of the network."""
    def accuracy(self, y, y_hat):
        num_correct = 0
        for pred, calc in zip(y, y_hat):
            if pred == calc:
                num_correct += 1
        return num_correct/self.num_examples

if __name__ == '__main__':
    X, y, _, _, X_test, y_test = load_data()
    mnist = MLP(D, h, m)
    mnist.fit(X, y)
    mnist.test(X_test, y_test)
