import numpy as np
from qpsolvers import solve_qp

trades = np.loadtxt(open("matrix_qp.csv", "rb"), skiprows=0, dtype=np.dtype('Float64'))
trades = trades.reshape(329,329)

P = np.dot(trades.T, trades)
q = np.asarray(np.full((329, 1), 0.)).reshape(-1)

G = -np.eye(329, dtype=float)
h = np.asarray(np.full((329, 1), 0.)).reshape(-1)

# Monotonic decreasing

C = np.full((328, 329), 0.)
for i in range(0,328):
    C[i,i] = -1
    C[i,i+1] = 1

c = np.asarray(np.full((328, 1), 0.)).reshape(-1)

G = np.concatenate((G, C), axis=0)
h = np.concatenate((h, c), axis=0)

# Convexity

C = np.full((327, 329), 0.)
for i in range(0,327):
    C[i,i] = -1
    C[i,i+1] = 2
    C[i,i+2] = -1

c = np.asarray(np.full((327, 1), 0.)).reshape(-1)

G = np.concatenate((G, C), axis=0)
h = np.concatenate((h, c), axis=0)

A = np.asarray(np.full((329, 1), 1.)).reshape(-1)
b = np.array([1.])

#print(P.shape,P.dtype)
#print(q.shape,q.dtype)
#print(G.shape,G.dtype)
#print(h.shape,h.dtype)
#print(A.shape,A.dtype)
#print(b.shape,b.dtype)

qp = solve_qp(P, q, G, h, A, b, solver="cvxopt")
z = qp[0]
print((3000*qp/z))
