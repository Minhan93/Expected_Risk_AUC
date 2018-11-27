import numpy as np
import scipy.io as sio
import os


def computeAUC(param, data, wnew):

    counter = 0
    func_step = 0
    sum_p = 0

    X_p = data.X_test_p
    X_n = data.X_test_n

    N_p = data.N_test_p
    N_n = data.N_test_n

    N = N_p + N_n

    # output of linear classifier f(w) = w' * x for positive data
    w_p = np.dot(X_p, wnew)

    # output of linear classifier f(w) = w' * x for negative data
    w_n = np.dot(X_n, wnew)

    list_merge = np.append(w_p, w_n)
    list_sort = np.sort(list_merge)
    list_idx = np.argsort(list_merge)

    for i in range(N):
        if (list_idx[N - 1 - i] <= (N_p - 1)):
            counter += 1
        else:
            func_step += counter

    fval_AUC = (float(func_step) / (N_p * N_n))

    return fval_AUC

    """
    X_p = data.X_test_p
    X_n = data.X_test_n

    N_p = data.N_test_p
    N_n = data.N_test_n

    # output of linear classifier f(w) = w' * x for positive data
    w_p = np.dot(X_p,wnew)


    # output of linear classifier f(w) = w' * x for negative data
    w_n = np.dot(X_n,wnew)

    # compute the value of the step function
    matrix_w_n = np.tile(w_n,(N_p,1))
    matrix_w_n = np.transpose(matrix_w_n)
    matrix_step = np.zeros((N_p,N_n))

    for i in range(N_n):
        matrix_step[:,i] = w_p - matrix_w_n[i,:]
        idx_neg = np.where(matrix_step[:,i]<0)[0]
        matrix_step[idx_neg ,i] = 0

    func_step = np.count_nonzero(matrix_step)

    # compute AUC function value
    fval_AUC = float (func_step) / (N_p*N_n)

    return fval_AUC
    """
