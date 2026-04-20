import numpy as np
import copy
from torch_gradient_computations_3 import ComputeGradsWithTorch
import matplotlib.pyplot as plt
import time

# !!!
# EXERCISE 1
# !!!

# Loading debugging data
debug_file = 'Assignment 3/debug_info.npz'
load_data = np.load(debug_file)
X1 = load_data['X1']
P = load_data['P']
X = load_data['X']
Y = load_data['Y']

Fs = load_data['Fs']
grad_Fs_flat = load_data['grad_Fs_flat']
conv_outputs_true = load_data['conv_outputs']
conv_flat = load_data['conv_flat']

W1 = load_data['W1']
W2 = load_data['W2']
b1 = load_data['b1']
b2 = load_data['b2']

n = X.shape[1] 
f = Fs.shape[0]    
nf = Fs.shape[3]    
np_dim = int(32 / f)

# First we reshape the 1D input images to 3D
X_ims = np.transpose(X.reshape((32, 32, 3, n), order='F'), (1, 0, 2, 3))

# Initializing output array (with zeros to start)
conv_outputs = np.zeros((np_dim, np_dim, nf, n))

# help taken from: https://stackoverflow.com/questions/48201729/difference-between-np-dot-and-np-multiply-with-np-sum-in-binary-cross-entropy-lo
# The for loop will loop through: 1. each image, 2. each sub window (one for height and one for width), 3. each filter
for i in range(n):
    for row in range(np_dim):
        for col in range(np_dim):
            for j in range(nf):
                # The stride start will be equal to the width of filter f
                # and the ending will be f added to start position
                start_row = row * f
                start_col = col * f
                end_row = start_row + f
                end_col = start_col + f

                # Need to extract X_ims patch from the image
                sub_patch = X_ims[start_row:end_row, start_col:end_col, :, i]
                
                # Extracting the F (4x4x) filter
                filter_k = Fs[:, :, :, j]

                conv_outputs[row, col, j, i] = np.sum(np.multiply(sub_patch, filter_k))

# If printed error is zero or very small that means it matches the ground truth, implementation is correct
difference = np.sum(np.abs(conv_outputs - conv_outputs_true))
print(f"Error: {difference}")

n_p = int(np_dim**2)
MX = np.zeros((n_p, f**2 * 3, n))

for i in range(n):
    l = 0
    for row in range(np_dim):
        for col in range(np_dim):
            start_row = row * f
            start_col = col * f
            end_row = start_row + f
            end_col = start_col + f

            X_patch = X_ims[start_row:end_row, start_col:end_col, :, i]

            MX[l, :, i] = X_patch.reshape((1, f*f*3), order='C')

            l += 1

# Computing convultion with matrix multip.
Fs_flat = Fs.reshape((f*f*3, nf), order='C')
conv_outputs_mat = np.zeros((n_p, nf, n))

for i in range(n):
    conv_outputs_mat[:, :, i] = np.matmul(MX[:, :, i], Fs_flat)

# Check comparison mat.mul. to loops
conv_outputs_flat = conv_outputs.reshape((n_p, nf, n), order='C')
diff_matmul = np.sum(np.abs(conv_outputs_mat - conv_outputs_flat))
print(f"Error (matmul vs. loops): {diff_matmul}")

# Check comparison with einsum to mat.mul.
conv_outputs_einsum = np.einsum('ijn, jl ->iln', MX, Fs_flat, optimize=True)
diff_einsum = np.sum(np.abs(conv_outputs_einsum - conv_outputs_mat))
print(f"Error (einsum vs. matMul): {diff_einsum}")

# !!!
# EXERCISE 2
# !!!

def ForwardPass(MX, network):
    W1 = network["W"][0]
    W2 = network["W"][1]
    b1 = network["b"][0]
    b2 = network["b"][1]
    Fs_flat = network["F"]

    bf = network.get("bf", np.zeros((Fs_flat.shape[1], 1)))

    n = MX.shape[2]
    n_p = MX.shape[0]
    nf = Fs_flat.shape[1]

    # using einsum and flattening the convolution output
    conv_outputs_einsum = np.einsum('ijn, jl ->iln', MX, Fs_flat, optimize=True)
    conv_outputs_einsum += bf.reshape((1, nf, 1))
    # Eq. (2)
    h = np.fmax(conv_outputs_einsum.reshape((n_p*nf, n), order='C'), 0) # conv_flat = h

    # Eq. (3) and (4)
    x1 = np.maximum(0, np.dot(W1, h) + b1)
    s = np.dot(W2, x1) + b2

    # softmax algo. Eq. (5)
    exp_val = np.exp(s)
    exp_val_sum = np.sum(exp_val, axis=0)
    P = exp_val / exp_val_sum

    fp_data = {"conv_flat": h, "X1": x1, "P": P}

    return P, fp_data

def BackwardPass(MX, Y, fp_data, network, lam):
    W1 = network["W"][0]
    W2 = network["W"][1]
    Fs_flat = network["F"]
    conv_flat = fp_data["conv_flat"]
    P = fp_data["P"]
    x1 = fp_data["X1"]

    n = MX.shape[2]
    n_p = MX.shape[0]
    nf = Fs_flat.shape[1]

    # From LECTURE 4
    G_batch = P - Y
    grad_W2 = 1/n * np.dot(G_batch, x1.T) + 2 * lam * W2
    grad_b2 = 1/n * np.sum(G_batch, axis=1, keepdims=True)

    # Backpropogate to first layer
    G_batch = np.dot(W2.T, G_batch)
    G_batch = G_batch * (x1 > 0) #ReLu

    grad_W1 = 1/n * np.dot(G_batch, conv_flat.T) + 2 * lam * W1
    grad_b1 = 1/n * np.sum(G_batch, axis=1, keepdims=True)

    # Propogate back to convolutional layer
    G_batch = np.dot(W1.T, G_batch)
    G_batch = G_batch * (conv_flat > 0)

    GG = G_batch.reshape((n_p, nf, n), order='C')

    grad_bf = 1/n * np.sum(GG, axis=(0, 2)).reshape((nf, 1))

    # # Eq. (22)
    MXt = np.transpose(MX, (1, 0, 2))
    grad_Fs_flat = 1/n * np.einsum('ijn, jln ->il', MXt, GG, optimize=True) + 2 * lam * Fs_flat

    grads = {"W": [grad_W1, grad_W2], "b": [grad_b1, grad_b2], "F": grad_Fs_flat, "bf": grad_bf}

    return grads

# DEBUGGING FORWARD PASS
# need to be in dictionary since that is the parameter format for the function
bf = np.zeros((nf, 1)) # nf entries
network = {"W": [W1, W2], "b": [b1, b2], "F": Fs_flat, "bf": bf}
P_debug, fp_data_debug = ForwardPass(MX, network)

diff_conv = np.sum(np.abs(fp_data_debug["conv_flat"] - conv_flat))
diff_X1 = np.sum(np.abs(fp_data_debug["X1"] - X1))
diff_P = np.sum(np.abs(P_debug - P))

print(f"Error in conv_flat: {diff_conv}")
print(f"Error in x1: {diff_X1}")
print(f"Error in P: {diff_P}")

"""
# DEBUGGING BACKWARD PASS
grads_debug = BackwardPass(MX, Y, fp_data_debug, network)

diff_grad_F = np.sum(np.abs(grads_debug["F"] - grad_Fs_flat))

print(f"Error in filter grads: {diff_grad_F}")
"""

# SANITY CHECK WITH TORCH
# First 5 images
MX_batch = MX[:, :, :5]
Y_batch = load_data['Y'][:, :5]
y_batch = load_data['y'][:5] 
network["bf"] = np.random.randn(nf, 1) * 0.1
lam = 0.01

P_batch, fp_data_batch = ForwardPass(MX_batch, network)
my_grads = BackwardPass(MX_batch, Y_batch, fp_data_batch, network, lam)
torch_grads = ComputeGradsWithTorch(MX_batch, y_batch, network, lam)

print("bf diff:", np.max(np.abs(torch_grads['bf'] - my_grads['bf'])))
print("W1 diff:", np.max(np.abs(torch_grads['W'][0] - my_grads['W'][0])))
print("W2 diff:", np.max(np.abs(torch_grads['W'][1] - my_grads['W'][1])))
print("F diff:", np.max(np.abs(torch_grads['F'] - my_grads['F'])))

# !!!
# EXERCISE 3
# !!!

# Following are from previous assignment

def LoadBatch(filename):
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

# Loading all 5 batches (test batch remains the same)
print("Loading 5 training batches for Searches")

X1, Y1, y1 = LoadBatch("Assignment 1/Datasets/cifar-10-batches-py/data_batch_1")
X2, Y2, y2 = LoadBatch("Assignment 1/Datasets/cifar-10-batches-py/data_batch_2")
X3, Y3, y3 = LoadBatch("Assignment 1/Datasets/cifar-10-batches-py/data_batch_3")
X4, Y4, y4 = LoadBatch("Assignment 1/Datasets/cifar-10-batches-py/data_batch_4")
X5, Y5, y5 = LoadBatch("Assignment 1/Datasets/cifar-10-batches-py/data_batch_5")
testX, testY, testy = LoadBatch("Assignment 1/Datasets/cifar-10-batches-py/test_batch")

X_all = np.hstack((X1, X2, X3, X4, X5))
Y_all = np.hstack((Y1, Y2, Y3, Y4, Y5))
y_all = np.hstack((y1, y2, y3, y4, y5))

# 49k and 1k train/val split
trainX = X_all[:, :-1000]
trainY = Y_all[:, :-1000]
trainy = y_all[:-1000]

valX = X_all[:, -1000:]
valY = Y_all[:, -1000:]
valy = y_all[-1000:]

d = trainX.shape[0]
mean_X = np.mean(trainX, axis=1).reshape(d, 1)
std_X = np.std(trainX, axis=1).reshape(d, 1)

trainX = (trainX - mean_X) / std_X
valX = (valX - mean_X) / std_X
testX = (testX - mean_X) / std_X

def ComputeEta(t, n_s, eta_min, eta_max):
    l = int(t / (2 * n_s))
    
    # Eq. (14)
    if (2 * l * n_s) <= t <= ((2 * l + 1) * n_s):
        eta_t = eta_min + ((t - 2 * l * n_s) / n_s) * (eta_max - eta_min)
        
    # Eq. (15)
    elif ((2 * l + 1) * n_s) <= t <= (2 * (l + 1) * n_s):
        eta_t = eta_max - ((t - (2 * l + 1) * n_s) / n_s) * (eta_max - eta_min)
        
    return eta_t

def ComputeAccuracy(P,y):
    # indices of max val.
    predictions = np.argmax(P, axis=0) 
    acc = np.mean(predictions == y) * 100

    return acc

def ComputeLoss(P, y):
    n = P.shape[1]
    L = -np.mean(np.log(P[y, np.arange(n)]))
    
    return L

# helper function to avoid repetition. Plots cost,loss and accuracy
def plot_results(metrics, lam, filename):
    steps = metrics["update_steps"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))

    # Loss Plot
    ax1.plot(steps, metrics["train_loss"], label="training", color="green")
    ax1.plot(steps, metrics["val_loss"], label="validation", color="red")
    ax1.set_title(f"Loss plot (lam={lam})")
    ax1.set_xlabel("update step")
    ax1.set_ylabel("loss")
    ax1.set_xlim(left=0)
    ax1.spines[['right', 'top']].set_visible(False)
    ax1.legend()

    # Accuracy Plot
    ax2.plot(steps, metrics["train_acc"], label="training", color="green")
    ax2.plot(steps, metrics["val_acc"], label="validation", color="red")
    ax2.set_title(f"Accuracy plot (lam={lam})")
    ax2.set_xlabel("update step")
    ax2.set_ylabel("accuracy")
    ax2.set_xlim(left=0)
    ax2.spines[['right', 'top']].set_visible(False)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def MiniBatchGD(MX_train, Y_train, MX_val, Y_val, GDparams, network, lam, rng):
    trained_net = copy.deepcopy(network)
    
    n = MX_train.shape[2]
    n_batch = GDparams["n_batch"]
    eta_min = GDparams["eta_min"]
    eta_max = GDparams["eta_max"]
    n_s = GDparams["n_s"]
    n_epochs = GDparams["n_epochs"]

    y_train = np.argmax(Y_train, axis=0)
    y_val = np.argmax(Y_val, axis=0)

    t = 0
    eval_step = int((2 * n_s) / 9)

    # Useful later for plotting
    eval = {"train_loss": [], "train_cost": [], "train_acc": [], "val_loss": [], "val_cost": [], "val_acc": [], "update_steps": []}
    
    for epoch in range(n_epochs):
        inds = rng.permutation(n)
        MX_shuffled = MX_train[:, :, inds]
        Y_shuffled = Y_train[:, inds]

        # Mini-batches
        for j in range(int(n/n_batch)):
            j_start = j*n_batch
            j_end = (j+1)*n_batch
            MXbatch = MX_shuffled[:, :, j_start:j_end]
            Ybatch = Y_shuffled[:, j_start:j_end]

            _, fp_data = ForwardPass(MXbatch, trained_net)
            grads = BackwardPass(MXbatch, Ybatch, fp_data, trained_net, lam)

            eta_t = ComputeEta(t, n_s, eta_min, eta_max)

            # GD update, Eq. (10) and (11)
            trained_net["W"][0] -= eta_t * grads["W"][0]
            trained_net["b"][0] -= eta_t * grads["b"][0]
            trained_net["W"][1] -= eta_t * grads["W"][1]
            trained_net["b"][1] -= eta_t * grads["b"][1]
            trained_net["F"] -= eta_t * grads["F"]
            trained_net["bf"] -= eta_t * grads["bf"]

            if t % eval_step == 0:
                # Evaluate training data 
                P_train, _ = ForwardPass(MX_train, trained_net)
                train_acc = ComputeAccuracy(P_train, y_train)
                train_loss = ComputeLoss(P_train, y_train)
                
                #  Evaluate validation data 
                P_val, _ = ForwardPass(MX_val, trained_net)
                val_acc = ComputeAccuracy(P_val, y_val)
                val_loss = ComputeLoss(P_val, y_val)
                
                # Calculate Cost (Loss + L2 Regularization for both layers -> from figure b) in ass.)
                l2_reg = lam * (np.sum(trained_net["W"][0]**2) + np.sum(trained_net["W"][1]**2))
                
                eval["train_loss"].append(train_loss)
                eval["train_cost"].append(train_loss + l2_reg)
                eval["train_acc"].append(train_acc)
                
                eval["val_loss"].append(val_loss)
                eval["val_cost"].append(val_loss + l2_reg)
                eval["val_acc"].append(val_acc)
                
                eval["update_steps"].append(t)
                
                print(f"Step {t} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | eta: {eta_t:.5f}")

            t += 1

    return trained_net, eval

# help taken from:
# https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
# https://www.geeksforgeeks.org/deep-learning/kaiming-initialization-in-deep-learning/

# He Initialization with input params: f, nf and nh (p. 10 and 11)
def Initialization(f, nf, nh):
    rng = np.random.default_rng(42)
    np_dim = int(32 / f)
    n_p = int(np_dim**2)
    net_params = {}

    # Scaling factors
    # Filter F has size fxfx3 (p. 2)
    F_scal = np.sqrt(2 / (f * f * 3))
    # W1 has size d0 = nf*np for single hidden layer d=nh
    W1_scal = np.sqrt(2 / (n_p * nf)) 
    W2_scal = np.sqrt(2 / nh)
    
    # Scaling weights
    Fs_flat = rng.standard_normal((f * f * 3, nf)) * F_scal
    W1 = rng.standard_normal((nh, n_p * nf)) * W1_scal
    W2 = rng.standard_normal((10, nh)) * W2_scal
    
    # Zero initialization for bias vectors (sizes on p. 9 and 11)
    bf = np.zeros((nf, 1))
    b1 = np.zeros((nh, 1))
    b2 = np.zeros((10, 1))

    net_params["W"] = [W1, W2]
    net_params["b"] = [b1, b2]
    net_params["F"] = Fs_flat
    net_params["bf"] = bf
    
    return net_params

def ComputeMX(X, f):
    n = X.shape[1]
    np_dim = int(32 / f)
    n_p = int(np_dim**2)

    # First we reshape the 1D input images to 3D
    X_ims = np.transpose(X.reshape((32, 32, 3, n), order='F'), (1, 0, 2, 3))
    MX = np.zeros((n_p, f**2 * 3, n))

    for i in range(n):
        l = 0
        for row in range(np_dim):
            for col in range(np_dim):
                start_row = row * f
                start_col = col * f
                end_row = start_row + f
                end_col = start_col + f

                X_patch = X_ims[start_row:end_row, start_col:end_col, :, i]

                MX[l, :, i] = X_patch.reshape((1, f*f*3), order='C')

                l += 1
    return MX

# 3 cycles = 6 * n_s = 4800 update steps
# 4800 steps / (49k/100) batches per epoch = int(9.8) = 10 epochs
param_settings = [
    {"f": 4, "nf": 10, "nh": 50, "lam": 0.003, "n_cycles": 3, "n_batch": 100, "eta_min": 1e-5, "eta_max": 1e-1, "n_s": 800, "n_epochs": 10}
]

for i, exp in enumerate(param_settings):
    GDparams = {"n_batch": exp["n_batch"], "eta_min": exp["eta_min"], "eta_max": exp["eta_max"], "n_s": exp["n_s"], "n_epochs": exp["n_epochs"]}
    f = exp["f"]
    nf = exp["nf"]
    nh = exp["nh"]
    lam = exp["lam"]

    print(f"Computing MX_train")
    MX_train = ComputeMX(trainX, f)
    print(f"Computing MX_val")
    MX_val = ComputeMX(valX, f)

    # Initializing network 
    net_params = Initialization(f, nf, nh)
    rng = np.random.default_rng(42)
    start_time = time.time() # timer start
    trained_net, eval = MiniBatchGD(MX_train, trainY, MX_val, valY, GDparams, net_params, lam, rng)

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Time: {training_time:.2f} s")

    MX_test = ComputeMX(testX, f)
    P_test, _ = ForwardPass(MX_test, trained_net)
    test_acc = ComputeAccuracy(P_test, testy)

    print(f"!!!FINAL TEST ACCURACY!!!: {test_acc:.2f}%")

    plot_results(eval, lam, f"Ass3_Ex3.png")