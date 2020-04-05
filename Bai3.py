import numpy as np

D_train = np.array([[1, 4, 5, 0, 3],
                    [5, 1, 0, 5, 2],
                    [4, 1, 2, 5, 0],
                    [0, 3, 4, 0, 4]
                    ])


def matrix_factorization(r, K, b, stop_condition):
    item = len(r[0])
    user = len(r)
    W = np.random.randint(2, size=(user, K))
    H = np.random.randint(2, size=(item, K))
    i = 0
    while i < stop_condition:
        r_bar = np.zeros((user, item))
        r_bar = np.dot(W, H.T)
        e = r - r_bar
        W_new = W + b * np.dot(e, H)
        H_new = H + b * np.dot(W.T, e).T
        i += 1
        W = np.around(W_new, decimals=5)
        H = np.around(H_new, decimals=5)
    return W, H


print(matrix_factorization(D_train, 2, 0.1, 1000))
