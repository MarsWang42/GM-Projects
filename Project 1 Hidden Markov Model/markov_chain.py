import numpy as np
from os import walk
mypath = 'proteins/'  # use path to data files
_, _, filenames = next(walk(mypath), (None, None, []))

mSeq = len(filenames)        # read in each sequence
o,x = [],[]
for i in range(mSeq):
    f = open( str(mypath) + filenames[i] , 'r')
    o.append( f.readline()[:-1] )  # strip trailing '\n'
    x.append( f.readline()[:-1] )
    f.close()


xvals, ovals = set(),set()  # extract the symbols used in x and o
for i in range(mSeq):
    xvals |= set(x[i])
    ovals |= set(o[i])


xvals = list( np.sort( list(xvals) ) )
ovals = list( np.sort( list(ovals) ) )
dx,do = len(xvals),len(ovals)

for i in range(mSeq):       # and convert to numeric indices
    x[i] = np.array([xvals.index(s) for s in x[i]])
    o[i] = np.array([ovals.index(s) for s in o[i]])


T = np.zeros((dx,dx), dtype=np.double)
O = np.zeros((dx,do), dtype=np.double)
p0 = np.bincount([row[0] for row in x], minlength = 8)

for i in range(len(x)):
    for j in range(x[i].size-1):
        T[x[i][j]][x[i][j+1]] = T[x[i][j]][x[i][j+1]] + 1
    for j in range(x[i].size):
        O[x[i][j]][o[i][j]] = O[x[i][j]][o[i][j]] + 1


from sklearn.preprocessing import normalize
p0 = normalize(p0, axis=1, norm='l1')
T = normalize(T, axis=1, norm='l1')
O = normalize(O, axis=1, norm='l1')

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

def most_likely_statement(o, p0, Tr, Ob):
    dx, do = Ob.shape
    L = len(o)
    f = np.zeros((L,dx), dtype=np.double)
    r = np.zeros((L,dx), dtype=np.int)
    f[0] = p0*Ob[:, o[0]]
    for t in range(1,L):    # compute forward messages
        for k in range(dx):
            f[t][k] = max(f[t-1]*np.transpose(Tr)[k]*Ob[:, o[t]])
            r[t][k] = np.argmax(f[t-1]*np.transpose(Tr)[k]*Ob[:, o[t]])
    ord = list()
    ord.append(np.argmax(f[L-1]))
    for t in range(L-2, -1, -1):
        ord.insert(0, r[t+1][ord[0]])
    return f,r,ord


for i in range(5):
  for j in range(5):
    print "T[{}][{}] = {}".format(i,j,T[i][j])

for i in range(5):
  for j in range(5):
    print "O[{}][{}] = {}".format(i,j,O[i][j])

for i in range(8):
  print "p[{}] = {}".format(i,a[0][i])

x = np.array([1,2,3])
o = np.array([1,2,4])

# Write the array to disk
with file('T.csv', 'w') as outfile:
    for data_slice in T:
        np.savetxt(outfile, data_slice[None], fmt='%d', delimiter=',')

with file('data1/o.txt', 'w') as outfile:
    for data_slice in o:
        np.savetxt(outfile, data_slice[None], fmt='%d', delimiter=',')
