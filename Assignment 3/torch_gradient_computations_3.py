import numpy as np
import torch

def ComputeGradsWithTorch(MX, y, network_params, lam):
    
    MXt = torch.tensor(MX, dtype=torch.float32)

    L = len(network_params['W'])

    # will be computing the gradient w.r.t. these parameters    
    W = [None] * L
    b = [None] * L    
    # Add dtype=torch.float32 to all of these!
    for i in range(len(network_params['W'])):
        W[i] = torch.tensor(network_params['W'][i], requires_grad=True, dtype=torch.float32)
        b[i] = torch.tensor(network_params['b'][i], requires_grad=True, dtype=torch.float32)   

    F_t = torch.tensor(network_params['F'], requires_grad=True, dtype=torch.float32)    

    ## give informative names to these torch classes        
    apply_relu = torch.nn.ReLU()
    apply_softmax = torch.nn.Softmax(dim=0)

    n_p = MX.shape[0]
    n = MX.shape[2]
    nf = F_t.shape[1]
    bft = torch.tensor(network_params['bf'], requires_grad=True, dtype=torch.float32)

    #### BEGIN your code ###########################
    conv_outputs = torch.zeros((n_p, nf, n), dtype=torch.float32)
    for i in range(n):
        conv_outputs[:, :, i] = torch.matmul(MXt[:, :, i], F_t)

    conv_outputs += bft.view(1, nf, 1)
    conv_flat = conv_outputs.reshape((n_p * nf, n))
    H_conv = apply_relu(conv_flat)

    # C. Fully Connected Layer 1
    s1 = torch.matmul(W[0], H_conv) + b[0]
    H1 = apply_relu(s1)
    
    # D. Fully Connected Layer 2 (Scores)
    scores = torch.matmul(W[1], H1) + b[1]
    
    # Apply the scoring function corresponding to equations (1-3) in assignment description 
    # If X is d x n then the final scores torch array should have size 10 x n 

    #### END of your code ###########################            

    # apply SoftMax to each column of scores     
    P = apply_softmax(scores)
    
    # compute the loss
    loss = torch.mean(-torch.log(P[y, np.arange(n)]))
    
    l2_reg = torch.sum(torch.pow(F_t, 2))
    for w in W:
        l2_reg += torch.sum(torch.pow(w, 2))
        
    cost = loss + lam * l2_reg
    cost.backward()

    # extract the computed gradients and make them numpy arrays 
    grads = {}
    grads['W'] = [None] * L
    grads['b'] = [None] * L
    for i in range(L):
        grads['W'][i] = W[i].grad.numpy()
        grads['b'][i] = b[i].grad.numpy()

    grads['F'] = F_t.grad.numpy()
    grads['bf'] = bft.grad.numpy()

    return grads
