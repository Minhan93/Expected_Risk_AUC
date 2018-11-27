import numpy as np
import scipy.io as sio
import os
from numpy import linalg as la
import timeit
from scipy import sparse
from scipy.sparse import csc_matrix


class dataReader:
    def __init__(self, param, run):

        direct = '/Users/messi/Documents/Year1/summer18/Code/RealData_ZeroOne1_ori/data/'
        direct = direct + param.name_data + '/'
        os.getcwd()
        os.path.exists(direct)

        print('run:')
        print(run)

        matfile = sio.loadmat(direct + 'tra_DFO_' + str(run) + '.mat', squeeze_me=True, struct_as_record=False)

        """ convert sparse arrays to numpy arrays """
        if param.name_data in param.name_list and param.method == 'cdf':
            self.X_train_p = matfile['data_tra_DFO'].X  # .toarray()      # positive training
            self.X_train_n = matfile['data_tra_DFO'].Y  # .toarray()      # negative training
        else:
            self.X_train_p = matfile['data_tra_DFO'].X.toarray()      # positive training
            self.X_train_n = matfile['data_tra_DFO'].Y.toarray()      # negative training
        """
        self.X_train_p = matfile['data_training'].X.toarray()      # positive training
        self.X_train_n = matfile['data_training'].Y.toarray()      # negative training
        """

        [self.N_train_p, d] = self.X_train_p.shape
        [self.N_train_n, d] = self.X_train_n.shape

        self.N_train = self.N_train_p + self.N_train_n
        # print('size is ', self.N_train)
        param.lmd = float(1) / self.N_train

        self.prob_p = float(self.N_train_p) / self.N_train
        self.prob_n = float(self.N_train_n) / self.N_train

        print('P_+: {}'.format(self.prob_p))
        print('P_-: {}'.format(self.prob_n))

        matfile = sio.loadmat(direct + 'test_AUC_' + str(run) + '.mat', squeeze_me=True, struct_as_record=False)

        """ convert sparse arrays to numpy arrays """

        self.X_test_p = matfile['data_test_AUC'].X.toarray()    # positive test
        self.X_test_n = matfile['data_test_AUC'].Y.toarray()    # negative test
        """
        self.X_test_p = matfile['data_test'].X.toarray()    # positive test
        self.X_test_n = matfile['data_test'].Y.toarray()    # negative test
        """

        [self.N_test_p, d] = self.X_test_p.shape
        [self.N_test_n, d] = self.X_test_n.shape

        self.N_test = self.N_test_p + self.N_test_n

        print('dimension: {} and size of data is: {}'.format(d, self.N_test + self.N_train))

        param.dim = d

        # intercept
        if param.name_data in param.name_list and param.method == 'cdf':
            self.X_train_p = sparse.hstack((self.X_train_p, np.ones((self.N_train_p, 1)))).tocsc()
            self.X_train_n = sparse.hstack((self.X_train_n, np.ones((self.N_train_n, 1)))).tocsc()
        else:
            self.X_train_p = np.append(self.X_train_p, np.ones((self.N_train_p, 1)), 1)
            self.X_train_n = np.append(self.X_train_n, np.ones((self.N_train_n, 1)), 1)
        print('type os X_train_p is', type(self.X_train_p))
        self.X_test_p = np.append(self.X_test_p, np.ones((self.N_test_p, 1)), 1)
        self.X_test_n = np.append(self.X_test_n, np.ones((self.N_test_n, 1)), 1)

        self.initw = np.random.rand(1, param.dim + 1).ravel()
        # self.initw *= 1e-1  #duke

        #self.initw = np.ones(param.dim+1)
        #self.initw = np.zeros(param.dim+1)
        # approximate mean and covariance
        if (param.method == 'cdf'):
            start_time_mom = timeit.default_timer()

            if param.name_data not in param.name_list:
                self.cov_pp = np.cov(self.X_train_p, y=None, rowvar=False, bias=False,
                                     ddof=None)

                self.cov_nn = np.cov(self.X_train_n, y=None, rowvar=False, bias=False,
                                     ddof=None)
            else:
                [n1, d] = self.X_train_p.shape
                [n2, d] = self.X_train_n.shape
                self.cov_pp = self.X_train_p.transpose().dot(self.X_train_p) / (n1 - 1)

                self.cov_nn = self.X_train_n.transpose().dot(self.X_train_n) / (n2 - 1)

                # self.G_pp = chol_approx(self.X_train_p, 0.1, param.ra)
                # self.G_nn = chol_approx(self.X_train_n, 0.1, param.ra)

            end_time_mom = timeit.default_timer()
            self.soltime_time_mom = end_time_mom - start_time_mom

            print('approximate moments time: {}'.format(self.soltime_time_mom))

            """
            warm starting
            """
            """
            frac_mean = float(np.dot(self.mean_p, self.mean_n)) / (la.norm(self.mean_n))**2
            w_orth_n = 1*(self.mean_p - frac_mean * self.mean_n)
            w_orth_n *= (float(1)/la.norm(w_orth_n))

            frac_mean = float(np.dot(self.mean_p, self.mean_n)) / (la.norm(self.mean_p))**2
            w_orth_p = -1*(self.mean_n - frac_mean * self.mean_p)
            w_orth_p *= (float(1)/la.norm(w_orth_p))

            self.initw =  w_orth_n + w_orth_p


            norm_mu_p = la.norm(self.mean_p)
            norm_mu_n = la.norm(self.mean_n)
            ave_means = float(norm_mu_p + norm_mu_n)/2
            norm_w = la.norm(self.initw)

            self.initw *= (float(ave_means)/norm_w)
            """

            # print self.mean_p.shape
            # print self.mean_n.shape
            self.mean_p = self.X_train_p.mean(0)
            self.mean_n = self.X_train_n.mean(0)

            # frac_mean = float(np.dot(self.mean_p, self.mean_n)) / (la.norm(self.mean_n))**2
            # self.initw = 1 * (self.mean_p - frac_mean * self.mean_n)

            # norm_mu_p = la.norm(self.mean_p)
            # norm_mu_n = la.norm(self.mean_n)
            # ave_means = float(norm_mu_p + norm_mu_n) / 2
            # norm_w = la.norm(self.initw)

            # self.initw *= (float(1) / norm_w)

            frac_mean = self.mean_p.dot(self.mean_n.transpose()) / (la.norm(self.mean_n))**2
            self.initw = 1 * (self.mean_p - frac_mean * self.mean_n)

            norm_mu_p = la.norm(self.mean_p)
            norm_mu_n = la.norm(self.mean_n)
            ave_means = float(norm_mu_p + norm_mu_n) / 2
            norm_w = la.norm(self.initw)

            self.initw *= (float(1) / norm_w)
            self.initw = np.array(self.initw).reshape(-1,)

        else:
            self.soltime_time_mom = 0

            #self.initw = np.random.rand(1,param.dim+1).ravel()


#import numpy as np
#import scipy.io as sio
#import os
#from numpy import linalg as la
#import timeit
#
#
# class dataReader:
#    def __init__(self, param, run):
#
#        #direct = '/Users/hivaghanbari/Documents/workspace_Python/SmoothAUC/dataAUC/'
#        direct = 'C:/Users/mil417/Downloads/Archive/RealData_ZeroOne1/data/'
#        direct = direct + param.name_data + '/'
#        os.getcwd()
#        os.path.exists(direct)
#
#        matfile = sio.loadmat(direct + 'tra_DFO_' + str(run) + '.mat', squeeze_me=True, struct_as_record=False)
#
#        """ convert sparse arrays to numpy arrays """
#
#        self.X_train_p = matfile['data_tra_DFO'].X.toarray()      # positive training
#        self.X_train_n = matfile['data_tra_DFO'].Y.toarray()      # negative training
#        """
#        self.X_train_p = matfile['data_training'].X.toarray()      # positive training
#        self.X_train_n = matfile['data_training'].Y.toarray()      # negative training
#        """
#
#        [self.N_train_p, d] = self.X_train_p.shape
#        [self.N_train_n, d] = self.X_train_n.shape
#
#        matfile = sio.loadmat(direct + 'test_AUC_' + str(run) + '.mat', squeeze_me=True, struct_as_record=False)
#
#        """ convert sparse arrays to numpy arrays """
#
#        self.X_test_p = matfile['data_test_AUC'].X.toarray()    # positive test
#        self.X_test_n = matfile['data_test_AUC'].Y.toarray()    # negative test
#        """
#        self.X_test_p = matfile['data_test'].X.toarray()    # positive test
#        self.X_test_n = matfile['data_test'].Y.toarray()    # negative test
#        """
#
#        [self.N_test_p, d] = self.X_test_p.shape
#        [self.N_test_n, d] = self.X_test_n.shape
#
#        param.dim = d
#
#        # intercept
#        self.X_train_p = np.append(self.X_train_p, np.ones((self.N_train_p, 1)), 1)
#        self.X_train_n = np.append(self.X_train_n, np.ones((self.N_train_n, 1)), 1)
#
#        self.X_test_p = np.append(self.X_test_p, np.ones((self.N_test_p, 1)), 1)
#        self.X_test_n = np.append(self.X_test_n, np.ones((self.N_test_n, 1)), 1)
#
#        self.initw = np.random.rand(1, param.dim + 1).ravel()
#        #self.initw = np.ones(param.dim)
#
#        # approximate mean and covariance
#        if (param.method == 'cdf'):
#
#            """
#            N = min(self.N_train_p, self.N_train_n)
#            Xp = self.X_train_p[0:N,:]
#            Xn = self.X_train_n[0:N,:]
#
#            X_train = np.hstack((Xp,Xn))
#
#
#            cov = np.cov(X_train, y=None, rowvar=False, bias=False,
#                       ddof=None, fweights=None, aweights=None)
#
#            self.cov_pn = cov[0:d, d:2*d]
#            self.cov_np = cov[d:2*d, 0:d]
#            """
#            #start_time_mom = timeit.default_timer()
#            self.mean_p = self.X_train_p.mean(0)
#            self.mean_n = self.X_train_n.mean(0)
#
#            self.cov_pp = np.cov(self.X_train_p, y=None, rowvar=False, bias=False,
#                                 ddof=None, fweights=None, aweights=None)
#            self.cov_nn = np.cov(self.X_train_n, y=None, rowvar=False, bias=False,
#                                 ddof=None, fweights=None, aweights=None)
#
#            #end_time_mom = timeit.default_timer()
#            #soltime_time_mom = end_time_mom - start_time_mom
#
#            # print 'approximate moments time: {}'.format(soltime_time_mom)
#
#            """
#            warm starting
#            """
#
#            frac_mean = float(np.dot(self.mean_p, self.mean_n)) / (la.norm(self.mean_n))**2
#            self.initw = 1 * (self.mean_p - frac_mean * self.mean_n)
#
#            norm_mu_p = la.norm(self.mean_p)
#            norm_mu_n = la.norm(self.mean_n)
#            ave_means = float(norm_mu_p + norm_mu_n) / 2
#            norm_w = la.norm(self.initw)
#
#            self.initw *= (float(1) / norm_w)

            #self.initw = (-self.mean_n + self.mean_p)
            #norm_diff = la.norm(self.mean_n - self.mean_p)
            #self.initw  *= (float(1)/norm_diff)
            #self.initw = np.random.rand(1,param.dim+1).ravel()
