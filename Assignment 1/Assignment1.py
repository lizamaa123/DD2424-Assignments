import numpy as np
import copy
from torch_gradient_computations import ComputeGradsWithTorch
import matplotlib.pyplot as plt

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

    grads = {"W": grad_W, "b": grad_b}
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

error_W = check(my_grads["W"], torch_grads["W"])
error_b = check(my_grads["b"], torch_grads["b"])

print(f"Max error for W: {error_W}")
print(f"Max error for b: {error_b}")

# STEP 8

def MiniBatchGD(X_train, Y_train, X_val, Y_val, GDparams, init_net, lam, rng):
    trained_net = copy.deepcopy(init_net)
    
    n = X_train.shape[1]
    n_batch = GDparams["n_batch"]
    eta = GDparams["eta"]
    n_epochs = GDparams["n_epochs"]

    y_train = np.argmax(Y_train, axis=0)
    y_val = np.argmax(Y_val, axis=0)

    # Useful later for plotting
    eval = {"train_loss": [], "train_cost": [], "val_loss": [], "val_cost": []}

    reg_term_init = lam * np.sum(trained_net["W"] ** 2)
    
    # Train data init
    P_train_init = ApplyNetwork(X_train, trained_net)
    train_loss_init = ComputeLoss(P_train_init, y_train)
    eval["train_loss"].append(train_loss_init)
    eval["train_cost"].append(train_loss_init + reg_term_init)
    
    # Val data init
    P_val_init = ApplyNetwork(X_val, trained_net)
    val_loss_init = ComputeLoss(P_val_init, y_val)
    eval["val_loss"].append(val_loss_init)
    eval["val_cost"].append(val_loss_init + reg_term_init)
    
    print(f"Epoch 0/{n_epochs} -> Train Cost: {train_loss_init + reg_term_init:.4f} and Val Cost: {val_loss_init + reg_term_init:.4f}")

    for epoch in range(n_epochs):
        # Shuffle data
        inds = rng.permutation(n)
        X_shuffled = X_train[:, inds]
        Y_shuffled = Y_train[:, inds]
        
        # Mini-batches
        for j in range(int(n/n_batch)):
            j_start = j*n_batch
            j_end = (j+1)*n_batch
            Xbatch = X_shuffled[:, j_start:j_end]
            Ybatch = Y_shuffled[:, j_start:j_end]

            Pbatch = ApplyNetwork(Xbatch, trained_net)
            grads = BackwardPass(Xbatch, Ybatch, Pbatch, trained_net, lam)

            # GD update 
            trained_net["W"] = trained_net["W"] - eta * grads["W"]
            trained_net["b"] = trained_net["b"] - eta * grads["b"]
        
        # L2-reg term
        reg_term = lam * np.sum(trained_net["W"] ** 2)
        
        # Training data evalutation
        P_train = ApplyNetwork(X_train, trained_net)
        train_loss = ComputeLoss(P_train, y_train)
        train_cost = train_loss + reg_term

        # Validation data evalutation
        P_val = ApplyNetwork(X_val, trained_net)
        val_loss = ComputeLoss(P_val, y_val)
        val_cost = val_loss + reg_term

        # Save evalutation
        eval["train_loss"].append(train_loss)
        eval["train_cost"].append(train_cost)
        eval["val_loss"].append(val_loss)
        eval["val_cost"].append(val_cost)

        print(f"Epoch {epoch+1}/{n_epochs} -> Train Cost: {train_cost:.4f} and Val Cost: {val_cost:.4f}")

    return trained_net, eval

# help taken from: https://sanjanasalkar.medium.com/building-an-image-classification-model-with-cifar-10-a-complete-guide-837ff4a50775
# https://stackoverflow.com/questions/925024/how-can-i-remove-the-top-and-right-axis
param_settings = [
    {"lam": 0,   "n_epochs": 40, "n_batch": 100, "eta": 0.1},
    {"lam": 0,   "n_epochs": 40, "n_batch": 100, "eta": 0.001},
    {"lam": 0.1, "n_epochs": 40, "n_batch": 100, "eta": 0.001},
    {"lam": 1,   "n_epochs": 40, "n_batch": 100, "eta": 0.001}
]

for i, exp in enumerate(param_settings):
    # Extract parameters for this iteration
    GDparams = {"n_batch": exp["n_batch"], "eta": exp["eta"], "n_epochs": exp["n_epochs"]}
    lam = exp["lam"]
    
    print(f"Running Experiment {i+1}: lam={exp["lam"]}, eta={exp["eta"]}")
    
    # Give the network same random seed (like assignment example)
    rng.bit_generator.state = BitGen(42).state 
    small_net = {}
    small_net['W'] = .01*rng.standard_normal(size = (10, d))
    small_net['b'] = np.zeros((10, 1))
    
    trained_net, eval = MiniBatchGD(trainX, trainY, valX, valY, GDparams, small_net, lam, rng)
    
    P_test = ApplyNetwork(testX, trained_net)
    test_acc = ComputeAccuracy(P_test, testy)
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    
    # PLOTTING LOSS AND COST 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss Plot
    ax1.plot(eval["train_loss"], label="training loss", color="green")
    ax1.plot(eval["val_loss"], label="validation loss", color="red")
    ax1.set_title(f"Loss Curve (lam={lam}, eta={GDparams["eta"]})")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_xlim(left=0)
    ax1.spines[['right', 'top']].set_visible(False)
    ax1.legend()
    
    # Cost Plot
    ax2.plot(eval["train_cost"], label="training cost", color="green")
    ax2.plot(eval["val_cost"], label="validation cost", color="red")
    ax2.set_title(f"Cost Curve (lam={lam}, eta={GDparams["eta"]})")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("loss")
    ax2.spines[['right', 'top']].set_visible(False)
    ax2.legend()    
    
    plt.savefig(f"Setting_{i+1}_Curves.png") 
    plt.show()
    
    # PLOTTING WEIGHT MATRIX
    Ws = trained_net['W'].transpose().reshape((32, 32, 3, 10), order='F')
    W_im = np.transpose(Ws, (1, 0, 2, 3))
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()
    
    for j in range(10):
        w_im = W_im[:, :, :, j]
        w_im_norm = (w_im - np.min(w_im)) / (np.max(w_im) - np.min(w_im))

        axes[j].imshow(w_im_norm)
        
    fig.suptitle(f"Learnt Weights (lam={lam}, eta={GDparams["eta"]})", fontsize=16)
    
    plt.savefig(f"Exp_{i+1}_Weights.png") 
    plt.show()