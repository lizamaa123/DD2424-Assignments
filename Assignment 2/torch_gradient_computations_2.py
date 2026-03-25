import numpy as np
import torch

def ComputeGradsWithTorch(X, y, network_params, lam):
    
    Xt = torch.from_numpy(X)

    L = len(network_params['W'])

    # will be computing the gradient w.r.t. these parameters    
    W = [None] * L
    b = [None] * L    
    for i in range(len(network_params['W'])):
        W[i] = torch.tensor(network_params['W'][i], requires_grad=True)
        b[i] = torch.tensor(network_params['b'][i], requires_grad=True)        

    ## give informative names to these torch classes        
    apply_relu = torch.nn.ReLU()
    apply_softmax = torch.nn.Softmax(dim=0)

    #### BEGIN your code ###########################
    s1 = torch.matmul(W[0], Xt) + b[0]
    H = apply_relu(s1)
    scores = torch.matmul(W[1], H) + b[1]
    
    # Apply the scoring function corresponding to equations (1-3) in assignment description 
    # If X is d x n then the final scores torch array should have size 10 x n 

    #### END of your code ###########################            

    # apply SoftMax to each column of scores     
    P = apply_softmax(scores)
    
    # compute the loss
    n = X.shape[1]
    loss = torch.mean(-torch.log(P[y, np.arange(n)]))
    
    l2_reg = 0
    for w in W:
        l2_reg += torch.sum(torch.pow(w, 2)) # squares the elements and sums them
        
    cost = loss + lam * l2_reg
    cost.backward()

    # extract the computed gradients and make them numpy arrays 
    grads = {}
    grads['W'] = [None] * L
    grads['b'] = [None] * L
    for i in range(L):
        grads['W'][i] = W[i].grad.numpy()
        grads['b'][i] = b[i].grad.numpy()

    return grads
