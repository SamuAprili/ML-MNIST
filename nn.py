import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# returns 'x_train' as the data train, 'y_train' as the labels and 'x_test' as the data test, 'y_test' as the labels
# 50 - 50 splitting
def splitting_train_test_data():
    data_train = np.array(pd.read_csv('train.csv'))
    N, d = data_train.shape # N is the number of samples; 'd' is the number of features plus the label

    np.random.shuffle(data_train)    
    data_train = data_train.T
    
    x_train = data_train[1:d, 0:int(N/2)]
    y_train = data_train[0, 0:int(N/2)]
    x_test = data_train[1:d, int(N/2) + 1:N - 1]
    y_test = data_train[0, int(N/2) + 1:N - 1]
    return x_train / 255.0, y_train, x_test / 255.0, y_test

# Initializes the weights and biases randomly, between -0.5 and 0.5
def init_param():
    W1 = (np.random.rand(10, 784) - 0.5) * 0.1
    W2 = (np.random.rand(10, 10) - 0.5) * 0.1
    b1 = (np.random.rand(10, 1) - 0.5) * 0.1
    b2 = (np.random.rand(10, 1) - 0.5) * 0.1
    return W1, W2, b1, b2

# Activation function of the first layer
def ReLU(Z):
    return np.maximum(Z, 0) # element wise operation

# First derivative of ReLU
def d_ReLU(Z):
    return Z > 0
    
# Activation function that returns the predicted probabilities
def softmax(Z):
    c = np.max(Z, axis=0, keepdims=True)
    exp_Z = np.exp(Z - c) # trick to prevent stack overflow
    
    res = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
    return res
    
def forward_propagation(W1, W2, b1, b2, A0):
    Z1 = W1.dot(A0) + b1
    A1 = ReLU(Z1)
    
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    
    return Z1, Z2, A1, A2

# returns the probabilities taking the labels as input; 'Y' is 1 by N; 'res' will be 10 by N
def labels_to_prob(Y):
    N = Y.size
    res = np.zeros((10, N))
    
    for i in range(0, N):
        res[int(Y[i]), i] = 1
        
    return res 


def back_propagation(A2, A1, A0, Z1, Y, W2):
    N = Y.size
    p = labels_to_prob(Y)
    dZ2 = A2 - p
    dW2 = 1 / N * dZ2.dot(A1.T)
    db2 = 1 / N * np.sum(dZ2, axis=1, keepdims=True)
    
    dZ1 = W2.T.dot(dZ2) * d_ReLU(Z1)
    dW1 = 1 / N * dZ1.dot(A0.T)
    db1 = 1 / N * np.sum(dZ1, axis=1, keepdims=True) 
    
    return dW1, dW2, db1, db2

def get_predictions(A2):
    return np.argmax(A2, 0)
        
def get_error(predicted, Y):
    print(predicted, Y)
    return sum(predicted != Y) / Y.size 

# computes the parameters' optimization  
def gradient_descend(alpha, iterations):
    X, Y, pass1, pass2 = splitting_train_test_data()
    W1, W2, b1, b2 = init_param()
    
    for i in range(1, iterations + 1):
        Z1, Z2, A1, A2 = forward_propagation(W1, W2, b1, b2, X)
        dW1, dW2, db1, db2 = back_propagation(A2, A1, X, Z1, Y, W2)

        # new parameters
        W1 = W1 - alpha * dW1
        W2 = W2 - alpha * dW2
        b1 = b1 - alpha * db1
        b2 = b2 - alpha * db2
        
        if i % 50 == 0: # every 50 iterations
            print(A2)
            pred = get_predictions(A2)
            print("Iteration: ", i)
            print(get_error(pred, Y))
            
    return W1, W2, b1, b2
            
 

def testing(W1, W2, b1, b2):
    pass1, pass2, X, Y = splitting_train_test_data()
    Z1, Z2, A1, A2 = forward_propagation(W1, W2, b1, b2, X)
    
    return get_error(get_predictions(A2), Y)
    
    
    
        
if __name__ == "__main__":
    W1, W2, b1, b2 = gradient_descend(0.1, 1500)
    
    print("Model error: ", testing(W1, W2, b1, b2))

