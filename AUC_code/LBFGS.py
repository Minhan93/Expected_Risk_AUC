# '''
# Created on Nov 4, 2016

# @author: hivaghanbari
# '''

# import numpy as np
# import scipy.io as sio
# import os


# class LBFGS:

#     def __init__(obj, param, data):

#         obj.m = param.memory_lbfgs    # LBFGS memory
#         obj.p = param.dim             # dimension
#         obj.L = np.array([])          # m*m matrix(M_k)
#         obj.D = np.array([])          # m*m diagonal matrix(D_k)
#         obj.S = np.array([])          # p*m matrix (S_k)
#         obj.T = np.array([])          # p*m matrix (T_k)
#         obj.control = np.array([])

#     """ update scalar gamma and two matrices Q and Q^ (update H) """
#     def compute(obj):
#         if (obj.D.size != 0) and (obj.D[-1] != 0):    # skip first iteration and the condition that s_{k-1}*t_{k-1}=0
#             obj.gama = np.dot(obj.T[:, -1], obj.T[:, -1])
#             obj.Q = np.hstack((obj.gama * obj.S, obj.T))
#             M_1 = np.hstack((obj.gama * np.dot(np.transpose(obj.S), obj.S), obj.L))

#             if (np.size(obj.D, 0) == 1):
#                 M_2 = np.hstack((np.transpose(obj.L), -obj.D))
#             else:
#                 M_2 = np.hstack((np.transpose(obj.L), -np.diagflat(obj.D)))

#             obj.control = np.vstack((M_1, M_2))
#             obj.control = obj.control + 1e-14 * np.identity(obj.control.shape[0])  # control singularity
#             obj.Q_bar = np.linalg.solve(obj.control, np.transpose(obj.Q))              # d = -H^{-1} * df
#         else:
#             obj.gama = 1
#             obj.Q = np.zeros((obj.p + 1, 2 * obj.m))
#             obj.Q_bar = np.zeros((2 * obj.m, obj.p + 1))

#     """ update matrix S,T, L and D for each new iteration """
#     def update(obj, s, y):
#         if (np.size(obj.S) != 0):
#             m_0 = np.size(obj.S, 1)
#         else:
#             m_0 = 0
#         if (np.dot(s, y) > 1e-14):
#             if m_0 < obj.m:
#                 if (m_0 == 0):
#                     obj.S = np.reshape(s, (obj.p + 1, 1))
#                     obj.T = np.reshape(y, (obj.p + 1, 1))
#                 else:
#                     obj.S = np.hstack((obj.S, np.reshape(s, (obj.p + 1, 1))))
#                     obj.T = np.hstack((obj.T, np.reshape(y, (obj.p + 1, 1))))

#                 """ add one row to the bottom of L """
#                 if (m_0 == 0):
#                     obj.L = np.array([])
#                 else:
#                     L_tmp = np.dot(obj.S[:, -1], obj.T[:, 0:-1])
#                     obj.L = np.vstack((obj.L, L_tmp))

#                 """ add one column to the rightmost of L """
#                 L_tmp = np.zeros((m_0 + 1, 1))

#                 if (m_0 == 0):
#                     obj.L = L_tmp
#                 else:
#                     obj.L = np.hstack((obj.L, L_tmp))

#                 if (m_0 == 0):
#                     obj.D = np.array([[np.dot(obj.S[:, -1], obj.T[:, -1])]])
#                 else:
#                     obj.D = np.vstack((obj.D, np.dot(obj.S[:, -1], obj.T[:, -1])))
#             else:

#                 """ delete the oldest pair """
#                 obj.S = np.delete(obj.S, 0, 1)
#                 obj.T = np.delete(obj.T, 0, 1)

#                 """ add the newest pair """
#                 obj.S = np.hstack((obj.S, np.reshape(s, (obj.p + 1, 1))))
#                 obj.T = np.hstack((obj.T, np.reshape(y, (obj.p + 1, 1))))

#                 """ delete the first row """
#                 obj.L = np.delete(obj.L, 0, 0)

#                 """ delete the leftmost column """
#                 obj.L = np.delete(obj.L, 0, 1)

#                 """ add one row to the bottom of L """
#                 L_tmp = np.dot(obj.S[:, -1], obj.T[:, 0:-1])
#                 obj.L = np.vstack((obj.L, L_tmp))

#                 """ add one column to the rightmost of L """
#                 L_tmp = np.zeros((obj.m, 1))
#                 obj.L = np.hstack((obj.L, L_tmp))

#                 """ delete the oldest pair """
#                 obj.D = np.delete(obj.D, 0, None)

#                 """ add the newest """
#                 obj.D = np.hstack((obj.D, np.dot(obj.S[:, -1], obj.T[:, -1])));


############

#   @File name: LBFGS.py
#   @Author:    Minhan Li
#   @Email: mil417@lehigh.com

#   @Create date:   2018-09-02 20:24:37

# @Last modified by:   Minhan Li
# @Last modified time: 2018-09-07T18:02:41-04:00

#   @Description:
#   @Example:

############

import numpy as np
import scipy.io as sio
import os


class LBFGS:

    def __init__(self, param, data):

        self.m = param.memory_lbfgs    # LBFGS memory
        self.p = param.dim + 1             # dimension
        self.S = np.zeros((self.p, self.m), dtype=np.float)  # p*m matrix, store s vector
        self.Y = np.zeros((self.p, self.m), dtype=np.float)  # p*m matrix, store y vector
        self.YS = np.zeros((self.m, 1), dtype=np.float)    # m*1 matrix, store ys product
        self.lbfgs_start = 1           # index indicates the start
        self.lbfgs_end = 0             # index indicates the end
        self.Hdiag = 1

    def lbfgsAdd(self, s, y):
        ys = np.dot(s, y)
        self.skipped = 0
        if ys > 1e-10:
            if self.lbfgs_end < self.m:
                self.lbfgs_end = self.lbfgs_end + 1
                if self.lbfgs_start != 1:
                    if self.lbfgs_start == self.m:
                        self.lbfgs_start = 1
                    else:
                        self.lbfgs_start += 1
            else:
                self.lbfgs_start = min(2, self.m)
                self.lbfgs_end = 1
            self.S[:, self.lbfgs_end - 1] = s
            self.Y[:, self.lbfgs_end - 1] = y
            self.YS[self.lbfgs_end - 1] = ys
            self.Hdiag = ys / np.dot(y, y)
        else:
            self.skipped = 1

    def lbfgsProd(self, g):
        if self.lbfgs_start == 1:
            ind = list(range(self.lbfgs_end))
            nCor = self.lbfgs_end - self.lbfgs_start + 1
        else:
            ind = list(range(self.lbfgs_start - 1, self.m)) + list(range(self.lbfgs_end))
            nCor = self.m
        al = np.zeros((nCor, 1), dtype=np.float)
        be = np.zeros((nCor, 1), dtype=np.float)
        d = -g

        for j in range(len(ind)):
            i = ind[-j - 1]
            al[i] = np.dot(self.S[:, i], d) / self.YS[i]
            d = d - al[i] * self.Y[:, i]
        d = self.Hdiag * d

        for i in ind:
            be[i] = np.dot(self.Y[:, i], d) / self.YS[i]
            d = d + self.S[:, i] * (al[i] - be[i])
        return d
