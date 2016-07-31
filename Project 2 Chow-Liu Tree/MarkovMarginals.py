def markovMarginals(x,o,p0,Tr,Ob):
    '''Compute p(o) and the marginal probabilities p(x_t|o) for a Markov model
       defined by P[xt=j|xt-1=i] = Tr(i,j) and P[ot=k|xt=i] = Ob(i,k) as numpy matrices'''
    dx,do = Ob.shape   # if a numpy matrix
    L = len(o)
    f = np.zeros((L,dx))
    r = np.zeros((L,dx))
    p = np.zeros((L,dx))
    f[0,:] = p0 * Ob[:,o[0]] # compute initial forward message
    from math import log
    log_pO = log(f[0,:].sum()) # update probability of sequence so far
    f[0,:] /= f[0,:].sum() # normalize (to match definition of f)
    for t in range(1,L):    # compute forward messages
        f[t,:] = Ob[:,o[t]]*np.dot(f[t-1,:], Tr)
        log_pO += log(f[t,:].sum())
        f[t,:] /= f[t,:].sum()
    r[L-1,:] = 1.0  # initialize reverse messages
    p[L-1,:] = f[L-1,:]*r[L-1,:]  # and marginals
    for t in range(L-2,-1,-1):
        r[t,:] = np.dot((r[t+1,:]*Ob[:,o[t+1]]),np.transpose(Tr))
        r[t,:] /= r[t,:].sum()
        p[t,:] = f[t,:]*r[t,:]
        p[t,:] /= p[t,:].sum()
    return log_pO, p, f, r
MarkovMarginal
