import numpy as np
import scipy.io as sio
import os
from numpy import linalg as la
#from chol_approx import *
import timeit
from scipy import sparse
from scipy.sparse import csc_matrix
direct = '/Users/messi/Documents/Year1/summer18/Code/RealData_ZeroOne1_ori/data/real_sim/'
os.getcwd()
os.path.exists(direct)
matfile = sio.loadmat(direct + 'tra_DFO_' + str(1) + '.mat', squeeze_me=True, struct_as_record=False)
#matfile = sio.loadmat(direct + 'segment' + '.mat', squeeze_me=True, struct_as_record=False)

""" convert sparse arrays to numpy arrays """

X_train_p = matfile['data_tra_DFO'].X  # .toarray()      # positive training
X_train_n = matfile['data_tra_DFO'].Y  # .toarray()      # negative training


print(sparse.isspmatrix_csc(X_train_n))
BB = X_train_n.mean(0)
print(BB.shape)
print(sparse.issparse(BB))
print(sparse.issparse(X_train_n[1, :]))
n, d = X_train_n.shape
# X_train_n = sparse.hstack((X_train_n, np.ones((n, 1)))).tocsc()
# print(1)
# print(sparse.isspmatrix_csc(X_train_n))

X_train_p1 = matfile['data_tra_DFO'].X.toarray()      # positive training
X_train_n1 = matfile['data_tra_DFO'].Y.toarray()      # negative training
print(sparse.isspmatrix_csc(X_train_n))
CC = np.mean(X_train_n1, 0)
mu = X_train_n.mean(0)
k = 2
j = 3
S = X_train_n
[n, d] = S.shape
print(S.shape)
mean_S = csc_matrix.sum(S, 0)
print(mean_S.shape)
print(type(mean_S))
print(type(S[:, 2]))

start_time = timeit.default_timer()
S[:, k].T * S[:, j]
end_time = timeit.default_timer()
print('1', end_time - start_time)

start_time = timeit.default_timer()
csc_matrix(mu[0, j] * np.ones((n,))) * S[:, k]
end_time = timeit.default_timer()
print('2', end_time - start_time)


start_time = timeit.default_timer()
WW = S.transpose().dot(S)
end_time = timeit.default_timer()
print('3', end_time - start_time)
print(WW.shape)
print(type(WW))

start_time = timeit.default_timer()
3.623 * 0.13451341
end_time = timeit.default_timer()
print('3', end_time - start_time)


mean_p = X_train_p.mean(0)
mean_n = X_train_n.mean(0)
frac_mean = mean_p.dot(mean_n.transpose()) / (la.norm(mean_n))**2
initw = 1 * (mean_p - frac_mean * mean_n)

norm_mu_p = la.norm(mean_p)
norm_mu_n = la.norm(mean_n)
ave_means = float(norm_mu_p + norm_mu_n) / 2
norm_w = la.norm(initw)

initw *= (float(1) / norm_w)
print(initw.shape)
print(type(initw))

# start_time = timeit.default_timer()
# sum(S[:, k])
# end_time = timeit.default_timer()
# print('3', end_time - start_time)


def Q_kj(S, mu, k, j):  # sparse matrix
    [n, d] = S.shape
    # print((- mu[k] * np.ones((n,))).shape)
    # print((S[:, j] - mu[j] * np.ones((n,))).shape)

#    start_time = timeit.default_timer()

    sumq = WW[k, j] - mu[0, j] * mean_S[0, k] - mu[0, k] * mean_S[0, j] + n * mu[0, k] * mu[0, j]

    #S[:, k].transpose().dot(S[:, j])
    # end_time = timeit.default_timer()
    # print("A")
    # print(end_time - start_time)
    return sumq / (n - 1)


# start_time = timeit.default_timer()
# AAA = Q_kj(X_train_n, BB, k, j)
# end_time = timeit.default_timer()
# print(end_time - start_time)


def Q_kj2(S, mu, k, j):  # sparse matrix
    [n, d] = S.shape
    # print((- mu[k] * np.ones((n,))).shape)
    # print((S[:, j] - mu[j] * np.ones((n,))).shape)
    sumq = (S[:, k].T * S[:, j]) - csc_matrix(mu[0, j] * np.ones((n,))) * S[:, k] - csc_matrix(mu[0, k] * np.ones((n,))) * S[:, j]
    sumq = sumq.toarray() + n * mu[0, j] * mu[0, k]
    return sumq / (n - 1)


# start_time = timeit.default_timer()
# AAA = Q_kj2(X_train_n, BB, k, j)
# end_time = timeit.default_timer()
# print(end_time - start_time)


def Qkj(S, mu, k, j):  # numpy matrix
    [n, d] = S.shape
    # print((- mu[k] * np.ones((n,))).shape)
    # print((S[:, j] - mu[j] * np.ones((n,))).shape)
    sumq = np.dot(S[:, k] - mu[k] * np.ones((n,)), S[:, j] - mu[j] * np.ones((n,)))
    return sumq / (n - 1)

# S = X_train_n
# [n, d] = S.shape
# mu = BB


# print(S[:, 2].T * S[:, 1])
# print(csr_matrix(mu[0, 2] * np.ones((n,))) * S[:, 1])
# print(csr_matrix(mu[0, 1] * np.ones((n,))) * S[:, 2])
# print(n * mu[0, 1] * mu[0, 2])

for i in range(10):  # use Sparse matrix
    for j in range(4):
        start_time = timeit.default_timer()
        AAA = Q_kj(X_train_n, BB, i, j)
        end_time = timeit.default_timer()
        print(end_time - start_time)


print("\n")
for i in range(10):    # use numpy array
    for j in range(4):
        start_time1 = timeit.default_timer()
        AAA = Q_kj2(X_train_n, BB, i, j)
        end_time1 = timeit.default_timer()
        print(end_time1 - start_time1)

# print(CC.shape)
print("\n")
for i in range(10):    # use numpy array
    for j in range(4):
        start_time1 = timeit.default_timer()
        AAA = Qkj(X_train_n1, CC, i, j)
        end_time1 = timeit.default_timer()
        print(end_time1 - start_time1)
