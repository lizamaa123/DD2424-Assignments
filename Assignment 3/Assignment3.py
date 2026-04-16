import numpy as np

# Loading debugging data
debug_file = 'Assignment 3/debug_info.npz'
load_data = np.load(debug_file)
X = load_data['X']
Fs = load_data['Fs']
H_true = load_data['conv_outputs']

n = X.shape[1] 
f = Fs.shape[0]    
nf = Fs.shape[3]    
np_dim = int(32 / f)

# First we reshape the 1D input images to 3D
X_ims = np.transpose(X.reshape((32, 32, 3, n), order='F'), (1, 0, 2, 3))

# Initializing output array (with zeros to start)
H = np.zeros((np_dim, np_dim, nf, n))

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

                H[row, col, j, i] = np.sum(np.multiply(sub_patch, filter_k))

# If printed error is zero that means it matches the ground truth, implementation is correct
difference = np.sum(np.abs(H - H_true))
print(f"Error: {difference}")