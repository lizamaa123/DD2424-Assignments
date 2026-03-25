import numpy as np
import copy
from torch_gradient_computations_2 import ComputeGradsWithTorch
import matplotlib.pyplot as plt

# EXERCISE 1

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

# Initialization 
def Initialization(m, d, K, seed):
    rng = np.random.default_rng()
    BitGen = type(rng.bit_generator)
    rng.bit_generator.state = BitGen(seed).state

    net_params = {}
    W1 = (1 / np.sqrt(d)) * rng.standard_normal(size = (m, d)) 
    W2 = (1 / np.sqrt(m)) * rng.standard_normal(size = (K, m))
    
    b1 = np.zeros((m, 1)) 
    b2 = np.zeros((K, 1)) 

    net_params["W"] = [W1, W2]
    net_params["b"] = [b1, b2]
    
    return net_params

m = 50 # nodes
net_params = Initialization(m, d, K, seed=42)

# EXERCISE 2

def ApplyNetwork(X, network):
    W1 = network["W"][0]
    W2 = network["W"][1]
    
    b1 = network["b"][0]
    b2 = network["b"][1]

    s1 = np.dot(W1, X) + b1
    h = np.maximum(0, s1)
    s = np.dot(W2, h) + b2

    exp_val = np.exp(s)
    exp_val_sum = np.sum(exp_val, axis=0)
    P = exp_val / exp_val_sum

    fp_data = {"s1": s1, "h": h, "P": P}

    return P, fp_data

P, fp_data = ApplyNetwork(trainX[:, 0:100], net_params)
print(P.shape) 

def ComputeAccuracy(P,y):
    # indices of max val.
    predictions = np.argmax(P, axis=0) 
    
    acc = np.mean(predictions == y) * 100

    return acc

def ComputeLoss(P, y):
    n = P.shape[1]
    L = -np.mean(np.log(P[y, np.arange(n)]))
    
    return L

def BackwardPass(X, Y, fp_data, network, lam):
    s1 = fp_data["s1"]
    h = fp_data["h"]
    P = fp_data["P"]
    
    W1 = network["W"][0]
    W2 = network["W"][1]

    n = X.shape[1]

    # Layer 2
    G_batch = P - Y

    grad_W2 = 1/n * np.dot(G_batch, h.T) + 2 * lam * W2
    grad_b2 = 1/n * np.sum(G_batch, axis=1, keepdims=True)

    # Backpropogate to Layer 1
    G_batch = np.dot(W2.T, G_batch)
    G_batch = G_batch * (s1 > 0)

    grad_W1 = 1/n * np.dot(G_batch, X.T) + 2 * lam * W1
    grad_b1 = 1/n * np.sum(G_batch, axis=1, keepdims=True)

    grads = {"W": [grad_W1, grad_W2], "b": [grad_b1, grad_b2]}

    return grads

# CHECK

d_small = 5
n_small = 3
m_small = 6
lam = 0

small_net = Initialization(m_small, d_small, K=10, seed=42)

X_small = trainX[0:d_small, 0:n_small]
Y_small = trainY[:, 0:n_small]

P, fp_data = ApplyNetwork(X_small, small_net)
my_grads = BackwardPass(X_small, Y_small, fp_data, small_net, lam)
torch_grads = ComputeGradsWithTorch(X_small, trainy[0:n_small], small_net, lam)

def check(g_a, g_n):
    eps=1e-6
    num = np.abs(g_a - g_n)
    denom = np.maximum(eps, np.abs(g_a) + np.abs(g_n))
    
    error = num / denom
    
    return np.max(error)

error_W1 = check(my_grads["W"][0], torch_grads["W"][0])
error_b1 = check(my_grads["b"][0], torch_grads["b"][0])
error_W2 = check(my_grads["W"][1], torch_grads["W"][1])
error_b2 = check(my_grads["b"][1], torch_grads["b"][1])

print(f"Max error for W1: {error_W1}")
print(f"Max error for b1: {error_b1}")
print(f"Max error for W2: {error_W2}")
print(f"Max error for b2: {error_b2}")

# SANITY CHECK, overfit to 100 examples
print("Sanity Check")
sanity_net = Initialization(m, d, K=10, seed=42)

X_sanity = trainX[:, 0:100]
Y_sanity = trainY[:, 0:100]
y_sanity = trainy[0:100]

eta = 0.1  
n_epochs = 200

for epoch in range(n_epochs):
    # Forward Pass
    P, fp_data = ApplyNetwork(X_sanity, sanity_net)
    
    # Compute Loss
    loss = ComputeLoss(P, y_sanity)
    
    # Print loss every 20 epochs 
    if epoch % 20 == 0 or epoch == n_epochs - 1:
        print(f"Epoch {epoch} -> Loss: {loss:.4f}")
        
    # Backward Pass
    grads = BackwardPass(X_sanity, Y_sanity, fp_data, sanity_net, lam)
    
    # GD
    sanity_net["W"][0] -= eta * grads["W"][0]
    sanity_net["b"][0] -= eta * grads["b"][0]
    
    sanity_net["W"][1] -= eta * grads["W"][1]
    sanity_net["b"][1] -= eta * grads["b"][1]


# EXERCISE 3

def ComputeEta(t, n_s, eta_min, eta_max):
    l = int(t / (2 * n_s))
    
    # Eq. (14)
    if (2 * l * n_s) <= t <= ((2 * l + 1) * n_s):
        eta_t = eta_min + ((t - 2 * l * n_s) / n_s) * (eta_max - eta_min)
        
    # Eq. (15)
    elif ((2 * l + 1) * n_s) <= t <= (2 * (l + 1) * n_s):
        eta_t = eta_max - ((t - (2 * l + 1) * n_s) / n_s) * (eta_max - eta_min)
        
    return eta_t

def MiniBatchGD(X_train, Y_train, X_val, Y_val, GDparams, network, lam):
    trained_net = copy.deepcopy(network)
    
    n = X_train.shape[1]
    n_batch = GDparams["n_batch"]
    eta_min = GDparams["eta_min"]
    eta_max = GDparams["eta_max"]
    n_s = GDparams["n_s"]
    n_epochs = GDparams["n_epochs"]

    t = 0
    eval_step = int((2 * n_s) / 10)

    # Useful later for plotting
    eval = {"train_loss": [], "train_cost": [], "train_acc": [], "val_loss": [], "val_cost": [], "val_acc": [], "update_steps": []}
    
    for epoch in range(n_epochs):
        # Shuffle data
        
        y_train = np.argmax(Y_train, axis=0)
        y_val = np.argmax(Y_val, axis=0)

        # Mini-batches
        for j in range(int(n/n_batch)):
            j_start = j*n_batch
            j_end = (j+1)*n_batch

            Xbatch = X_train[:, j_start:j_end]
            Ybatch = Y_train[:, j_start:j_end]

            Pbatch, fp_data = ApplyNetwork(Xbatch, trained_net)
            grads = BackwardPass(Xbatch, Ybatch, fp_data, trained_net, lam)

            eta_t = ComputeEta(t, n_s, eta_min, eta_max)

            # GD update 
            trained_net["W"][0] -= eta_t * grads["W"][0]
            trained_net["b"][0] -= eta_t * grads["b"][0]
            trained_net["W"][1] -= eta_t * grads["W"][1]
            trained_net["b"][1] -= eta_t * grads["b"][1]

            if t % eval_step == 0:
                # Evaluate training data 
                P_train, _ = ApplyNetwork(X_train, trained_net)
                train_acc = ComputeAccuracy(P_train, y_train)
                train_loss = ComputeLoss(P_train, y_train)
                
                #  Evaluate validation data 
                P_val, _ = ApplyNetwork(X_val, trained_net)
                val_acc = ComputeAccuracy(P_val, y_val)
                val_loss = ComputeLoss(P_val, y_val)
                
                # Calculate Cost (Loss + L2 Regularization for both layers)
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

GDparams = {"n_batch": 100, "eta_min": 1e-5, "eta_max": 1e-1, "n_s": 500, "n_epochs": 10}
lam = 0.01
m = 50
net_params = Initialization(m, d, K=10, seed=42)
trained_net, eval = MiniBatchGD(trainX, trainY, valX, valY, GDparams, net_params, lam)

P_test, _ = ApplyNetwork(testX, trained_net)
test_acc = ComputeAccuracy(P_test, testy) 
print(f"Final Test Accuracy: {test_acc:.2f}%")

steps = eval["update_steps"]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

# Cost Plot
ax1.plot(steps, eval["train_cost"], label="training", color="green")
ax1.plot(steps, eval["val_cost"], label="validation", color="red")
ax1.set_title("Cost plot")
ax1.set_xlabel("update step")
ax1.set_ylabel("cost")
ax1.spines[['right', 'top']].set_visible(False)
ax1.legend()

# Loss Plot
ax2.plot(steps, eval["train_loss"], label="training", color="green")
ax2.plot(steps, eval["val_loss"], label="validation", color="red")
ax2.set_title("Loss plot")
ax2.set_xlabel("update step")
ax2.set_ylabel("loss")
ax2.spines[['right', 'top']].set_visible(False)
ax2.legend()

# Accuracy Plot
ax3.plot(steps, eval["train_acc"], label="training", color="green")
ax3.plot(steps, eval["val_acc"], label="validation", color="red")
ax3.set_title("Accuracy plot")
ax3.set_xlabel("update step")
ax3.set_ylabel("accuracy")
ax3.spines[['right', 'top']].set_visible(False)
ax3.legend()

plt.tight_layout()
plt.savefig("Exercise3_Curves.png")
plt.show()