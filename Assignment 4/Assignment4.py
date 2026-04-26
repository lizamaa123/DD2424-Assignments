import numpy as np
from torch_gradient_computations_column_wise import ComputeGradsWithTorch

# !!!
# EXERCISE 1
# !!!

# Read training data (text file)
book_dir = ''
book_fname = book_dir + 'Assignment 4/goblet_book.txt'
fid = open(book_fname, "r")
book_data = fid.read()
fid.close()
unique_chars = list(set(book_data))

K = len(unique_chars)
m = 100 
eta = 0.001
seq_length=25

# help taken from: https://www.geeksforgeeks.org/python/adding-items-to-a-dictionary-in-a-loop-in-python/

# Initalize empty dictionaries
char_to_ind = {}
ind_to_char = {}

# from 0 to K-1
for i, char in enumerate(unique_chars): 
    char_to_ind[char] = i
    ind_to_char[i] = char

print(f"Number of unique characters, K={K}")

def Initialization(m, K, seed):
    RNN = {}
    rng = np.random.default_rng(seed)

    b = np.zeros((m, 1)) 
    c = np.zeros((K, 1)) 

    RNN['b'] = b
    RNN['c'] = c
    RNN['U'] = (1/np.sqrt(2*K))*rng.standard_normal(size = (m, K))
    RNN['W'] = (1/np.sqrt(2*m))*rng.standard_normal(size = (m, m))
    RNN['V'] = (1/np.sqrt(m))*rng.standard_normal(size = (K, m))

    return RNN

def seqSynth(RNN, h0, x0, n):
    b = RNN['b']
    c = RNN['c']
    U = RNN['U']
    W = RNN['W'] 
    V = RNN['V']

    h = h0
    x = x0
    Y = np.zeros((K, n))
    
    for t in range(n):
        # Eq. 1-4
        a = np.dot(W, h) + np.dot(U, x) + b
        h = np.tanh(a)
        o = np.dot(V, h) + c

        exp_val = np.exp(o)
        exp_val_sum = np.sum(exp_val, axis=0)
        p = exp_val / exp_val_sum

        # sampling the next character
        cp = np.cumsum(p, axis=0)
        local_rng = np.random.default_rng() 
        a_random = local_rng.uniform(size=1)
        ii = np.argmax(cp - a_random > 0)

        # save sampled char. to output matrix
        Y[ii, t] = 1

        # update x to be one-hot of char.
        x = np.zeros((K, 1))
        x[ii, 0] = 1

    return Y

# Back-propogation (forward + backward pass)
def forwardPass(X, Y, RNN, h0): 
    b = RNN['b']
    c = RNN['c']
    U = RNN['U']
    W = RNN['W'] 
    V = RNN['V']

    n = X.shape[1]
    K = RNN['c'].shape[0]
    m = RNN['b'].shape[0]

    h = h0
    loss = 0

    # matrices to save hidden states, states and probs.
    H = np.zeros((m, n))
    A = np.zeros((m, n))
    P = np.zeros((K, n))

    for t in range(n):
        x = X[:, t:t+1]
        y = Y[:, t:t+1]
        
        # Eq. 1-4
        a = np.dot(W, h) + np.dot(U, x) + b
        h = np.tanh(a)
        o = np.dot(V, h) + c
        
        exp_val = np.exp(o)
        exp_val_sum = np.sum(exp_val, axis=0)
        p = exp_val / exp_val_sum
        
        # saving states for backward pass
        A[:, t:t+1] = a
        H[:, t:t+1] = h
        P[:, t:t+1] = p
        
        # Eq. 5
        loss += np.log(np.dot(y.T, p)[0, 0])
        loss = loss / n
        
        # update h
        h = h
        
    return loss, P, H, A, h

# help taken from:
# https://www.geeksforgeeks.org/python/backward-iteration-in-python/
def backwardPass(RNN, X, Y, P, H, A, h0):
    W = RNN['W'] 
    V = RNN['V']
    n = X.shape[1]

    grads = {}
    for i, j in RNN.items():
        grads[i] = np.zeros_like(j)

    # gradient of a_(t+1) initialization
    grad_a_next = np.zeros_like(RNN['b'])
    
    # looping backwards from n-1, n-2,...., 1
    for t in reversed(range(n)):
        # to get 2D vector
        p = P[:, t:t+1] # Kx1
        y = Y[:, t:t+1] # Kx1
        h = H[:, t:t+1] # mx1
        x = X[:, t:t+1] # Kx1
        a = A[:, t:t+1] # mx1

        if t == 0:
            h_prev = h0
        else:
            h_prev = H[:, t-1:t]
            
        # FROM LECTURE 9

        # gradient of o
        g = (p - y) / n
        
        grads['V'] += np.dot(g, h.T)
        grad_h = np.dot(V.T, g) + np.dot(W.T, grad_a_next)
        grad_a = grad_h * (1 - np.tanh(a)**2)
        grads['W'] += np.dot(grad_a, h_prev.T)
        grads['U'] += np.dot(grad_a, x.T)
        grads['c'] += g
        grads['b'] += grad_a

        grad_a_next = grad_a
        
    return grads

# helper method to turn the sliced X,Y in to one hot encoded matrices (Kxseq_length)
def onehotChar(chars, char_to_ind, K):
    seq_length = len(chars)
    Y = np.zeros((K, seq_length))
    for t, char in enumerate(chars):
        Y[char_to_ind[char], t] = 1
    return Y

X_chars = book_data[0:seq_length]
Y_chars = book_data[1:seq_length+1]

X = onehotChar(X_chars, char_to_ind, K)
Y = onehotChar(Y_chars, char_to_ind, K)

# Initializing
m_check = 10
h0_check = np.zeros((m_check, 1))
RNN_check = Initialization(m_check, K, seed=42)

# My gradients
loss, my_P, my_H, my_A, my_h = forwardPass(X, Y, RNN_check, h0_check)
my_grads = backwardPass(RNN_check, X, Y, my_P, my_H, my_A, h0_check)

# Pytorch gradients
y_int = np.array([char_to_ind[c] for c in Y_chars]) 
pytorch_grads = ComputeGradsWithTorch(X, y_int, h0_check, RNN_check)

# Evalutation
for key in my_grads.keys():
    diff = np.max(np.abs(my_grads[key] - pytorch_grads[key]))
    print(f"Maximum difference for {key}: {diff}")