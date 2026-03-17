import torch
import numpy as np

# STEP 1

def LoadBatch(filename):
    # help taken from: https://www.cs.toronto.edu/~kriz/cifar.html
    # https://stackoverflow.com/questions/71747126/float64-normalisation-in-pytorch

    """X = image pixel data, dxn (raw data is nxd), float32/64, entries 0.0 and 1.0
    n = 10000, d = 3072
    Y = Kxn, K = 10, n = 10000, entries 0.0 and 1.0
    Y is one-hot encoded, meaning that for each column, one entry is 1
    y = vector of len(n)"""
    import pickle
    with open (filename, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
    X = datadict[b'data'].astype(np.float64).T / 255.0 # normalizing by max pixel value
    y = np.array(datadict[b'labels']).astype(int)

    n = X.shape[1]
    K = 10
    Y = np.zeros((K, n), dtype=np.float64)
    for i in range(n):
        Y[y[i], i] = 1 

    return X, Y, y

# STEP 2
    
# Read in and store the data
trainX, trainY, trainy = LoadBatch("Assignment 1/Datasets/cifar-10-batches-py/data_batch_1")
valX, valY, valy = LoadBatch("Assignment 1/Datasets/cifar-10-batches-py/data_batch_2")
testX, testY, testy = LoadBatch("Assignment 1/Datasets/cifar-10-batches-py/test_batch")

d = trainX.shape[0]
K = trainY.shape[0] 

# Preprocess input data 
mean_X = np.mean(trainX, axis=1).reshape(d, 1)
std_X = np.std(trainX, axis=1).reshape(d, 1)

# Normalize
trainX = (trainX - mean_X) / std_X
valX = (valX - mean_X) / std_X
testX = (testX - mean_X) / std_X

# STEP 3

# Initialization 
rng = np.random.default_rng()
# get the BitGenerator used by default_rng
BitGen = type(rng.bit_generator)
# use the state from a fresh bit generator
seed = 42
rng.bit_generator.state = BitGen(seed).state
init_net = {} #dictionary
init_net['W'] = .01*rng.standard_normal(size = (K, d)) # entries are small random numbers, mean 0, std 0.01
init_net['b'] = np.zeros((K, 1)) # entries are all zeros, shape Kx1

# STEP 4
def ApplyNetwork(X, network):
    # help taken from: https://www.geeksforgeeks.org/python/how-to-implement-softmax-and-cross-entropy-in-python-and-pytorch/
    # https://www.geeksforgeeks.org/python/python-accessing-key-value-in-dictionary/
    """X=dxn
    network=dictionary{'W', 'b'}
    P=Kxn"""
    W = network["W"]
    b = network["b"]
    s = np.dot(W, X) + b
    exp_val = np.exp(s)
    exp_val_sum = np.sum(exp_val, axis=0)
    P = exp_val / exp_val_sum
    return P

P = ApplyNetwork(trainX[:, 0:100], init_net)
print(P.shape) 
print(np.sum(P, axis=0))

# STEP 5

def ComputeLoss(P, y):
    """P=Kxn
    y=1xn
    L=scalar, 1x1 = mean cross entropy loss"""
    n = P.shape[1]
    L = 1/n * (-np.log(P[y, np.arange(n)]))
    return L