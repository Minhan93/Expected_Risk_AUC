import numpy as np
import scipy.io as sio
import os


class LBFGS:

    def __init__(self, param, data):

        self.m = param.memory_lbfgs    # LBFGS memory
        self.p = param.dim + 1             # dimension
        self.S = np.zeros((self.p, self.m), dtype=np.float)  # p*m matrix, store s vectors
        self.Y = np.zeros((self.p, self.m), dtype=np.float)  # p*m matrix, store y vectors
        self.YS = np.zeros((self.m, 1), dtype=np.float)    # m*1 matrix, store ys products
        self.lbfgs_start = 1           # index indicates the start
        self.lbfgs_end = 0             # index indicates the end
        self.control = np.array([])
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
