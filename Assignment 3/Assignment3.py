import numpy as np
# !!!
# EXERCISE 1
# !!!

# Loading debugging data
debug_file = 'Assignment 3/debug_info.npz'
load_data = np.load(debug_file)
X = load_data['X']
Fs = load_data['Fs']
conv_outputs_true = load_data['conv_outputs']

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

n_p = np_dim**2
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

