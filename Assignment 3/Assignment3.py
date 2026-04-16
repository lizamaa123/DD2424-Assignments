import numpy as np
from torch_gradient_computations_3 import ComputeGradsWithTorch

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


