
import numpy as np
import scipy.io as sio
import os


def computeAccuracy(param, data, wnew):
    # N_p = data.N_test
    # N_n = data.N_test

    N_p = data.N_test_p
    N_n = data.N_test_n

    N = N_p + N_n
    print('N is %d' % N)

    # extract positive and negative samples
    X_p = data.X_test_p
    X_n = data.X_test_n

    # compute number of missclassified positive points
    # w_p = np.dot(wnew, X_p.transpose())
    # idx_p = np.where(w_p>0)[0]
    # w_p[idx_p] = 0
    # err_p = np.count_nonzero(w_p)

    # # compute number of missclassified negative points
    # w_n = np.dot(wnew, X_n.transpose())
    # idx_n = np.where(w_n<0)[0]
    # w_n[idx_n] = 0
    # err_n = np.count_nonzero(w_n)
    w_p = np.dot(wnew, X_p.transpose())
    idx_p = np.where(w_p < 0)[0]
    err_p = len(idx_p)

    # compute number of missclassified negative points
    w_n = np.dot(wnew, X_n.transpose())
    idx_n = np.where(w_n > 0)[0]
    err_n = len(idx_n)

    # final error is
    num_error = err_p + err_n

    fval_accuracy = 1 - float(num_error) / N

    return fval_accuracy
