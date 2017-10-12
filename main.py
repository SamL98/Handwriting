import cv2
import numpy as np
from classifier import create_saved_network
import mnist2 as mnist

drawing = False
prev = (None, None)

n_batches = 5
num_pred = 0
curr_pred = None
training = False

Xtr = np.zeros((10, 784))
ytr = np.zeros((10, 1))
Xv = np.zeros((5, 784))
yv = np.zeros((5, 1))
network = create_saved_network()

def train(n):
    global num_pred, curr_pred, Xtr, ytr, Xv, yv, network
    if num_pred <= 10:
        Xtr[num_pred-1] = curr_pred
        ytr[num_pred-1] = n
    else:
        Xv[num_pred-1] = curr_pred
        yv[num_pred-1] = n

    curr_pred = None

    if num_pred >= 15:
        num_pred = 0
        mnist.fit(Xtr, ytr, Xv, yv, network)

def classify():
    global drawing, num_pred, curr_pred, training, network

    if drawing:
        return

    X = cv2.resize(img, (0, 0), fx=(1/10), fy=(1/10))
    prediction = mnist.predict(np.reshape(1.0 - X/255, (1, 784)), network)
    print("Prediction: ", prediction)

    curr_pred = prediction
    num_pred += 1
    training = True

def draw(event, x, y, flags, param):
    global drawing, prev

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        prev = (x, y)

    elif drawing and event == cv2.EVENT_MOUSEMOVE:
        cv2.line(img, prev, (x, y), (0, 0, 0), 5)
        prev = (x, y)

    elif drawing and event == cv2.EVENT_LBUTTONUP:
        if (x, y) == prev:
            cv2.circle(img, prev, 5, (0, 0, 0), -1)

        drawing = False
        prev = (None, None)
    

def initialize_image():
    img = np.ones((280, 280, 1), np.uint8)
    return img * 255

img = initialize_image()

cv2.namedWindow('canvas')
cv2.setMouseCallback('canvas', draw)

while(1):
    cv2.imshow('canvas', img)
    k = cv2.waitKey(20)
    if k == 27:
        break
    elif k == 4:
        img = initialize_image()
        prev = (None, None)
        drawing = False
        training = False
    elif k == 19:
        classify()
    elif k >= 48 and training:
        print("Expected: ", (k - 48))
        train(k - 48)
        training = False

cv2.destroyAllWindows()
