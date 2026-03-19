import numpy as np
import copy
from torch_gradient_computations import ComputeGradsWithTorch

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
    
# Read and store data
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
    # iter over the row from y and column from np.arange and recieve then log average of loss
    # = probability of the correct classes for each image
    L = -np.mean(np.log(P[y, np.arange(n)]))
    
    return L

# STEP 6

def ComputeAccuracy(P,y):
    """P=Kxn
    y=len(n)
    acc=scalar
    """
    # indices of max val.
    predictions = np.argmax(P, axis=0) 
    
    acc = np.mean(predictions == y) * 100

    return acc

# STEP 7

def BackwardPass(X, Y, P, network, lam):
    """X=dxn
    Y=Kxn
    P=Kxn
    grads=dictionary with keys W and b."""
    # Help taken from last slides in lecture 3
    G_batch = P - Y
    n = G_batch.shape[1]
    W = network["W"]

    # gradient wrt W
    grad_W = 1/n * np.dot(G_batch, X.T) + 2 * lam * W

    #gradient wrt b
    grad_b = 1/n * np.sum(G_batch, axis=1, keepdims=True)

    grads = {'W': grad_W, 'b': grad_b}
    return grads


# CHECK

d_small = 10
n_small = 3
lam = 0

small_net = {}
small_net['W'] = .01*rng.standard_normal(size = (10, d_small))
small_net['b'] = np.zeros((10, 1))

X_small = trainX[0:d_small, 0:n_small]
Y_small = trainY[:, 0:n_small]

P = ApplyNetwork(X_small, small_net)
my_grads = BackwardPass(X_small, Y_small, P, small_net, lam)
torch_grads = ComputeGradsWithTorch(X_small, trainy[0:n_small], small_net, lam)


def check(g_a, g_n):
    eps=1e-6
    num = np.abs(g_a - g_n)
    denom = np.maximum(eps, np.abs(g_a) + np.abs(g_n))
    
    error = num / denom
    
    return np.max(error)

error_W = check(my_grads['W'], torch_grads['W'])
error_b = check(my_grads['b'], torch_grads['b'])

print(f"Max error for W: {error_W}")
print(f"Max error for b: {error_b}")


# STEP 8

def MiniBatchGD(X_train, Y_train, X_val, Y_val, GDparams, init_net, lam, rng):
    trained_net = copy.deepcopy(init_net)
    
    n = X_train.shape[1]
    n_batch = GDparams["n_batch"]
    eta = GDparams["eta"]
    n_epochs = GDparams["n_epochs"]

    for _ in range(n_epochs):
        inds = rng.permutation(n)
        X_shuffled = X_train[:, inds]
        Y_shuffled = Y_train[:, inds]
        
        for j in range(int(n/n_batch)):
            j_start = j*n_batch
            j_end = (j+1)*n_batch
            Xbatch = X_shuffled[:, j_start:j_end]
            Ybatch = Y_shuffled[:, j_start:j_end]

            Pbatch = ApplyNetwork(Xbatch, trained_net)
            grads = BackwardPass(Xbatch, Ybatch, Pbatch, trained_net, lam)

            # GD update (mini-batches)
            trained_net["W"] = trained_net["W"] - eta * grads["W"]
            trained_net["b"] = trained_net["b"] - eta * grads["b"]

        y_t = np.argmax(Y_train, axis=0)
        y_v = np.argmax(Y_val, axis=0)
        
        train_loss = ComputeLoss(Pbatch, y_t)
        val_loss = ComputeLoss(Pbatch, y_v)

        print(f"training loss: {train_loss}")
        print(f"validation loss: {val_loss}")

    return trained_net
