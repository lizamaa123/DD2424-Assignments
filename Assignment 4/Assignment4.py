import numpy as np

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
for char, i in enumerate(unique_chars): 
    char_to_ind = {char: i}
    ind_to_char = {i: char}

print(f"Number of unique characters, K={K}")

RNN = {}
rng = np.random.default_rng(42)

b = np.zeros((m, 1)) 
c = np.zeros((K, 1)) 

RNN['b'] = [b]
RNN['c'] = [c]
RNN['U'] = (1/np.sqrt(2*K))*rng.standard_normal(size = (m, K))
RNN['W'] = (1/np.sqrt(2*m))*rng.standard_normal(size = (m, m))
RNN['V'] = (1/np.sqrt(m))*rng.standard_normal(size = (K, m))

def function(RNN, h0, x0, n):
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
        a = rng.uniform(size=1)
        ii = np.argmax(cp - a > 0)

        # save sampled char. to output matrix
        Y[ii, t] = 1

        # update x to be one-hot of char.
        x = np.zeros((K, 1))
        x[ii, 0] = 1

    return Y

