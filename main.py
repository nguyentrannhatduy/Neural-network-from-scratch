import numpy as np
from numpy._core import ndarray
from numpy._core.umath import log


def prepare_data():
    X = np.array([
      [150, 70],
      [254, 73],
      [312, 68],
      [120, 60],
      [154, 61],
      [212, 65],
      [216, 67],
      [145, 67],
      [184, 64],
      [130, 69]
    ])
    y = np.array([0,1,1,0,0,1,1,0,1,0])
    m = 10
    A0 = X.T
    L = 3
    n = [2, 3, 3, 1]
    Y = y.reshape(n[L], m)

    """
    Define X as 10 x 2 = m x n[0] matrix

    Define A_0 as X, the input layer

    Reshape target vector y to be output matrix Y with dims 10 x 1 = m x n[3]
    """

    return A0, Y, m, n, L
def initialize_network(layers: int, nodes: list[int], A0: np.ndarray, m):
    if len(nodes) - 1 != layers:
        raise ValueError("number of layers and length of nodes must equal")
    if layers < 2:
        raise ValueError("network must have at least 2 layers")
    assert A0.shape == (nodes[0], m)

    W, b = [], []
    A = [A0]
    for i in range(1, layers):
        W.append(np.random.rand(nodes[i], nodes[i - 1]))
        b.append(np.random.rand(nodes[i], 1))
    return W, b, A

def cost(y_hat: ndarray, y: ndarray):
    """
        y_hat should be a n^L x m matrix
        y should be a n^L x m matrix
    """
    losses = -((y * np.log(y_hat)) + (1 - y) * np.log(1 - y_hat))
    m = y_hat.reshape(-1).shape[0]

    summed_losses = np.sum(losses, axis=1) / m

    return np.sum(summed_losses)
def g(z): #sigmoid function
    return 1 / (1 + np.exp(-1 * z))

def general_feed_forward(W: list[np.ndarray], b: list[np.ndarray], A: list[np.ndarray]):
    if A[0] is None:
        raise ValueError("A[0] is none")
    if W[0] is None:
        raise ValueError("W[0] is none")
    if b[0] is None:
        raise ValueError("b[0] is none")
    Z = []

    for i in range (len(W)):
        z = W[i] @ A[i] + b[i]
        Z.append(z)
        A.append(g(z))
    return Z, A
def backprop(W: list[np.ndarray], b: list[np.ndarray], A: list[ndarray], nodes: list[int], m: int, Y: np.ndarray):
    if any(x is None for x in W):
        raise ValueError("W contains a None value")
    if any(x is None for x in b):
        raise ValueError("b contains a None value")
    if any(x is None for x in A):
        raise ValueError("A contains a None value") 
    updateW = []
    updateb = [] 
    dC_dZ = 1/m * (A[-1] - Y)

    for i in range (len(W) - 1, -1, -1):
        updateW.append(dC_dZ @ A[i].T)
        updateb.append(np.sum(dC_dZ, axis=1, keepdims=True))

        if i > 0:
            dC_dA_prev = W[i].T @ dC_dZ
            dC_dZ = dC_dA_prev * A[i + 1] * (1 - A[i + 1])
    updateW.reverse()
    updateb.reverse()
    return updateW, updateb
A0, Y, m, n, L = prepare_data()
W, b, A = initialize_network(L, n, A0, m)

def train_data():
    global W, b, A
    costs = []
    epochs = 1000
    alpha = 0.1
    A0 = A[0]

    for i in range (epochs):
        Z, A = general_feed_forward(W, b, [A0])
        err = cost(A[-1], Y)
        costs.append(err)

        updateW, updateb = backprop(W, b, A, n, m, Y)
        
        #update
        for j in range (len(W)):
            W[j] = W[j] - alpha * updateW[j]
            b[j] = b[j] - alpha * updateb[j]
        if i % 20 == 0:
            print(f"epoch {i}: cost = {err:4f}")
    return costs
train_data()



