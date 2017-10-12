import numpy as np
from mnist2 import construct_network, predict
from PIL import Image

def read_saved_weights():
    weights = []
    with open('mnist_weights2.txt') as f:
        fileiter = iter(f)
        for line in fileiter:
            if line.strip() == 'FC':
                W = np.genfromtxt(next(fileiter).strip())
                b = np.genfromtxt(next(fileiter).strip())
                weights.append((W, b))
    
    return weights

def create_saved_network():
    network = construct_network()
    weights = read_saved_weights()
    for weight, layer in zip(weights, [network[0], network[2], network[4]]):
        layer.W = weight[0]
        layer.b = weight[1]
    return network

if __name__ == '__main__':
    im = Image.open('test.jpg')
    data = im.getdata()

    X = np.zeros((1,784))
    y = np.zeros((1, 10))

    network = create_saved_network()
        
    output = predict(X, network)
    print(output)