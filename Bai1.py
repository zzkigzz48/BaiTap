import numpy as np

D_train = np.array([[1, 4, 5, 0, 3],
                    [5, 1, 0, 5, 2],
                    [4, 1, 2, 5, 0],
                    [0, 3, 4, 0, 4]
                    ])


def sort_sim(val):
    return val[1]


def sim_cosine(D_train, u0, u1):
    U = len(D_train[0])
    sumproduct = 0
    r_u0 = 0
    r_u1 = 0
    for i in range(U):
        r_u0 += D_train[u0][i] * D_train[u0][i]
        r_u1 += D_train[u1][i] * D_train[u1][i]
    sumproduct = np.dot(D_train[u0], D_train[u1])
    denominator = (r_u0**(1.0 / 2)) * (r_u1**(1.0 / 2))
    return sumproduct / denominator


def mean(u):
    return float(sum(u)) / (len(u) - np.count_nonzero(u == 0))


def pearson_sim(D_train, u0, u1):
    Item = len(D_train[0])
    mean_u = []
    numerator = 0
    S_u0 = 0
    S_u1 = 0
    for u in D_train:
        mean_u.append(mean(u))
    for i in range(Item):
        #     print(D_train[u0][i])
        if (D_train[u0][i] != 0) and (D_train[u1][i] != 0):
            # print(D_train[u0][i] ,"-", mean_u[u0])
            tmp0 = D_train[u0][i] - mean_u[u0]
            tmp1 = D_train[u1][i] - mean_u[u1]
            numerator += tmp0 * tmp1
            S_u0 += tmp0 ** 2
            S_u1 += tmp1 ** 2

    avg = (S_u0 * S_u1)**(1.0 / 2)
    return numerator / avg


def KNN_item_base_sin(D_train, K, u, item):
    U = len(D_train)
    sim = {}
    array = []
    numerator = 0
    denominator = 0
    for i in range(U):
        if i != u:
            sim[i] = sim_cosine(D_train, i, u)
            array.append((i, sim[i]))
    array.sort(key=sort_sim, reverse=True)

    for i in range(K):
        numerator += array[i][1] * D_train[array[i][0]][item]
        denominator += array[i][1]
    print(numerator, denominator)
    return numerator / denominator


def KNN_item_base_pea(D_train, K, u, item):
    U = len(D_train)
    sim = {}
    array = []
    mean_u = []

    for user in D_train:
        mean_u.append(mean(user))

    numerator = 0
    denominator = 0
    for i in range(U):
        if i != u:
            sim[i] = pearson_sim(D_train, i, u)
            array.append((i, sim[i]))
    # print("array = ", array)
    array.sort(key=sort_sim, reverse=True)

    for i in range(K):
        print(array[i][1], D_train[array[i][0]][item])
        numerator += array[i][1] * \
            (D_train[array[i][0]][item] - mean_u[array[i][0]])
        denominator += np.abs(array[i][1])
    # print(numerator, denominator)
    return numerator / denominator


# print(KNN_item_base_sin(D_train , 2, 3, 1))
print(KNN_item_base_pea(D_train, 2, 3, 0))