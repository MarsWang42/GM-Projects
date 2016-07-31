from pyGM import *
import numpy as np
# Load the data points D, and the station locations (lat/lon)
D = np.genfromtxt('data/data.txt',delimiter=None)
loc = np.genfromtxt('data/locations.txt',delimiter=None)
m,n = D.shape # m = 2760 data points, n=30 dimensional # D[i,j] = 1 if station j observed rainfall on day i

count = np.zeros(30)
for j in range(n):
    for i in range(m):
        if D[i][j] == 1:
            count[j] += 1

x = count/m

count = np.zeros((30, 30, 4))
for i in range(n):
    for j in range(n):
        for k in range(m):
            if D[k][i] == 0 and D[k][j] == 0:
                count[i][j][0] += 1
            if D[k][i] == 1 and D[k][j] == 0:
                count[i][j][1] += 1
            if D[k][i] == 0 and D[k][j] == 1:
                count[i][j][2] += 1
            if D[k][i] == 1 and D[k][j] == 1:
                count[i][j][3] += 1

xij = np.zeros((30, 30, 4))
for i in range(n):
    for j in range(n):
        for k in range(4):
            xij[i][j][k] = count[i][j][k]/m

from math import log
I = np.zeros((30,30))
for i in range(n):
    I[i][i] = x[i]*log(x[i])+(1-x[i])*log(1-x[i])
    for j in range(i+1, n):
        I[j][i] = I[i][j] = xij[i][j][0]*log(xij[i][j][0]/((1-x[i])*(1-x[j])))+xij[i][j][1]*log(xij[i][j][1]/(x[i]*(1-x[j])))+xij[i][j][2]*log(xij[i][j][2]/((1-x[i])*x[j]))+xij[i][j][3]*log(xij[i][j][3]/(x[i]*x[j]))

#A = adjacency matrix, u = vertex u, v = vertex v
def weight(A, u, v):
    return A[u][v]

#A = adjacency matrix, u = vertex u
def adjacent(A, u):
    L = []
    for x in range(len(A)):
        if x != u:
            L.insert(0,x)
    return L

#Q = max queue
def extractMax(Q):
    q = Q[0]
    Q.remove(Q[0])
    return q

#Q = max queue, V = vertex list
def increaseKey(Q, K):
    for i in range(len(Q)):
        for j in range(len(Q)):
            if K[Q[i]] > K[Q[j]]:
                s = Q[i]
                Q[i] = Q[j]
                Q[j] = s

#V = vertex list, A = adjacency list, r = root
def prim(V, A, r):
    u = 0
    v = 0

    # initialize and set each value of the array P (pi) to none
    # pi holds the parent of u, so P(v)=u means u is the parent of v
    P = [None]*len(V)

    # initialize and set each value of the array K (key) to -999999
    K = [-999999]*len(V)

    # initialize the max queue and fill it with all vertices in V
    Q = [0]*len(V)
    for u in range(len(Q)):
        Q[u] = V[u]

    # set the key of the root to 0
    K[r] = 0
    increaseKey(Q, K)    # maintain the max queue

    # loop while the max queue is not empty
    while len(Q) > 0:
        u = extractMax(Q)    # pop the first vertex off the max queue

        # loop through the vertices adjacent to u
        Adj = adjacent(A, u)
        for v in Adj:
            w = weight(A, u, v)    # get the weight of the edge uv

            # proceed if v is in Q and the weight of uv is great than v's key
            if Q.count(v)>0 and w > K[v]:
                # set v's parent to u
                P[v] = u
                # v's key to the weight of uv
                K[v] = w
                increaseKey(Q, K)    # maintain the min queue
    return P

V = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]

P = prim(V, I, 0)
print(P)

ll = I[0][0]
for i in range(1,n):
    I1 = I[i][i] + I[P[i]][i]]
    ll += I1
print('log likelihood: {}'.format(ll))

import matplotlib.pyplot as plt
p,q = loc.shape
fig,ax=plt.subplots(1,1)
for i in range(1,p):
    x = [loc[i][0],loc[P[i]][0]]
    y = [loc[i][1],loc[P[i]][1]]
    plt.plot(x,y,'o-')
for i in range(p):
    plt.text(loc[i][0],loc[i][1],str(i))
plt.show()

import numpy as np
import matplotlib.pyplot as plt

loc = np.genfromtxt('data/locations.txt',delimiter=None)
E = np.genfromtxt('data/edges.txt', delimiter=None)

fig,ax=plt.subplots(1,1)
for e in E:
    u , v = int(e[0]), int(e[1])
    x = [loc[u][0],loc[v][0]]
    y = [loc[u][1],loc[v][1]]
    plt.plot(x,y,'o-')
for i in range(30):
    plt.text(loc[i][0],loc[i][1],str(i))


X = [Var(i,2) for i in range(30)]
P = [Factor(X[i],0.0) for i in range(n)]
for i in range(n):
    P[i][0] = 1-x[i]
    P[i][1] = x[i]


W = [[0] * n for i in range(n)]
for i in range(n):
  W[i][i] = x[i]
  for j in range(i+1, n):
    W[i][j] = Factor([X[i], X[j]], 0.0)
    W[i][j][0,0] = xij[i][j][0]
    W[i][j][1,0] = xij[i][j][1]
    W[i][j][0,1] = xij[i][j][2]
    W[i][j][1,1] = xij[i][j][3]
    W[j][i] = W[i][j]


factors = [Factor([X[int(e[0])], X[int(e[1])]], 1.0) for e in E]
pri = [1.0 for Xi in X]
for i in range(15):
    ll = 0
    for j, e in enumerate(E):
        model_ve = GraphModel(factors)
        k,l = int(e[0]), int(e[1])
        pri[k],pri[l] = 2.0, 2.0
        order = eliminationOrder(model_ve, orderMethod = 'minfill', priority = pri)[0]
        sumElim = lambda F,Xlist: F.sum(Xlist)   # helper function for eliminate
        model_ve.eliminate(order[:-2], sumElim)  # eliminate all but last two
        p = model_ve.joint()
        p /= p.sum()
        factors[j] *= W[k][l]/p  # update the factors
        pri[k],pri[l] = 1.0, 1.0

    model_ve = GraphModel(factors)
    order = eliminationOrder(model_ve, orderMethod = 'minfill', priority = pri)[0]
    sumElim = lambda F,Xlist: F.sum(Xlist)   # helper function for eliminate
    model_ve.eliminate(order, sumElim)  # eliminate all variables to get Z
    Z = model_ve.joint()

    # calculating the log likelihood
    for k in range(m):
        for j,e in enumerate(E):
            u, v = int(e[0]), int(e[1])
            a, b = int(D[k][u]), int(D[k][v])
            ll += log(factors[j][a,b])
    ll /= m
    ll -= log(Z.table)
    print('step: {} log likelihood: {} logZ: {}'.format(i+1, ll, log(Z.table)))




