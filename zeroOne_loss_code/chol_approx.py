import numpy as np
import time
from scipy.sparse import csc_matrix
import timeit


def Q_kj(mu, Var, mean_S, n, k, j):  # sparse matrix

    sumq = Var[k, j] - mu[0, j] * mean_S[0, k] - mu[0, k] * mean_S[0, j] + n * mu[0, k] * mu[0, j]

    return sumq / (n - 1)


def chol_approx(S, epsilon, ra):
    mu = S.mean(0)
    [n, d] = S.shape
    mean_S = csc_matrix.sum(S, 0)
    Var = S.transpose().dot(S)
    G = np.zeros((d, ra))
    Qjj = np.array([])

    for i in range(d):
        start_time = timeit.default_timer()
        Qjj.append(Q_kj(mu, Var, mean_S, n, i, i))
        end_time = timeit.default_timer()
        print('The time used is')
        print(end_time - start_time)

    epsilon = epsilon * sum(Qjj)

    # tmpV=np.zeros((d,1))
    # tmp=0
    perm = list(range(d))
    ind = 0

    for i in range(ra):
        #print('current iter is %d' % i)
        for j in range(i, d):
            # print (perm[j])
            # print (Qjj[perm[j]])
            G[j, i] = Qjj[perm[j]]
            for m in range(i):
                G[j, i] = G[j, i] - G[j, m] * G[j, m]

        sump = np.sum(G[i:d, i])

        if ind == 0 and sump > epsilon and G[i, i] == 0:
            G[i:d, i] = np.zeros((d - i, ))
            continue
        else:
            ind += 1

        if sump > epsilon:
            val = G[i:d, i].max()
            idx = G[i:d, i].argmax()

            if abs(val) < 1e-8:
                break

            idx = idx + i

            if ind == 1:
                idx = i

            temp = perm[i]
            perm[i] = perm[idx]
            perm[idx] = temp

            for m in range(i):
                tmpW = G[idx, m]
                G[idx, m] = G[i, m]
                G[i, m] = tmpW

            if ind == 1:
                G[i, i] = np.sqrt(G[i, i])
            else:
                G[i, i] = np.sqrt(val)

            for m in range(i + 1, d):
                start_time = timeit.default_timer()
                G[m, i] = Q_kj(mu, Var, mean_S, n, perm[m], perm[i])

                # for j in range(i):
                #     G[m, i] = G[m, i] - G[m, j] * G[i, j]
                G[m, i] -= np.dot(G[m, 0:i], G[i, 0:i])
                end_time = timeit.default_timer()
                print('The time used are')
                print(end_time - start_time)

            G[i + 1:d, i] = G[i + 1:d, i] / G[i, i]

        else:
            k = i - 1
            break
    per = np.argsort(perm)
    return G[per, :]
