import numpy as np
import random
from numpy import linalg as la
from scipy.stats import norm


class defObj:

    def __init__(obj, param, data):
        obj.max_iter = param.max_iter
        obj.gtol = param.opt_tol
        obj.inctol = param.opt_inc
        obj.N_train = data.N_train
        obj.N_train_p = data.N_train_p
        obj.N_train_n = data.N_train_n
        obj.N_test = data.N_test
        obj.p = param.dim
        obj.method = param.method
        obj.algorithm = param.algorithm
        obj.error = param.error
        obj.lmd = param.lmd
        obj.batch_size = param.batch_size

        obj.prob_p = data.prob_p
        obj.prob_n = data.prob_n

        obj.initw = data.initw
        obj.name_list = param.name_list
        obj.name_data = param.name_data
        if (param.method == 'zeroOne'):
            obj.mean_p = data.mean_p
            obj.mean_n = data.mean_n
            # if param.name_data not in obj.name_list:
            obj.cov_pp = data.cov_pp
            obj.cov_nn = data.cov_nn
            # else:
            #     obj.G_pp = data.G_pp
            #     obj.G_nn = data.G_nn

        obj.X_train_p = data.X_train_p
        obj.X_train_n = data.X_train_n

        obj.mean_z_p = 0
        obj.cov_z_p = 0
        obj.mean_z_n = 0
        obj.cov_z_n = 0
        obj.cov_mean_p = np.zeros((obj.p, 1))
        obj.cov_mean_n = np.zeros((obj.p, 1))

        obj.iters = 0
        obj.iter_back = 0
        obj.mu = param.mu_init

        obj.flag_opt = False
        obj.message_opt = ' '

        obj.initx(obj.initw)
        obj.initf()
        obj.initdf()
        obj.initH()

    def compute_moments_z(obj, xnew):

        obj.mean_z_p = obj.mean_p.dot(xnew.transpose())                # w mu_p
        obj.mean_z_n = obj.mean_n.dot(xnew.transpose())                # w mu_n
        if obj.name_data not in obj.name_list:
            obj.cov_mean_p = np.dot(xnew, obj.cov_pp)              # w cov_pp
            obj.cov_z_p = np.dot(obj.cov_mean_p, xnew)             # w cov_pp w
            obj.cov_mean_n = np.dot(xnew, obj.cov_nn)              # w cov_nn
            obj.cov_z_n = np.dot(obj.cov_mean_n, xnew)             # w cov_nn w
        else:
            obj.cov_mean_p = obj.cov_pp.dot(xnew.transpose())           # w cov_pp
            obj.cov_z_p = obj.cov_mean_p.transpose().dot(xnew.transpose())            # w cov_pp w
            obj.cov_mean_n = obj.cov_nn.dot(xnew.transpose())             # w cov_nn
            obj.cov_z_n = obj.cov_mean_n.transpose().dot(xnew.transpose())            # w cov_nn w
            # print('cov_mean_n shape', obj.cov_mean_p.shape)
            # print('cov_z shape', obj.cov_z_p.shape)

    # compute function value
    def evalf(obj, xnew):

        if obj.method == 'zeroOne':
            obj.compute_moments_z(xnew)

            frac_p = float(obj.mean_z_p) / np.sqrt(obj.cov_z_p)
            fval_p = norm.cdf(frac_p)

            frac_n = float(obj.mean_z_n) / np.sqrt(obj.cov_z_n)
            fval_n = norm.cdf(frac_n)

            fval = obj.prob_p + obj.prob_n * fval_n - obj.prob_p * fval_p + 0.001 * (1 - (np.linalg.norm(xnew))**2)**2
            # print('faval type', type(fval))
        elif obj.method == 'logReg':

            # extract positive and negative samples
            X_p = obj.X_train_p
            X_n = obj.X_train_n

            obj.e_wx_p = np.exp(np.multiply(1, np.dot(X_p, xnew)))
            # obj.e_wx_p += 1e-30   #segment

            fval_p = np.sum(np.log(1 + 1. / obj.e_wx_p))

            obj.e_wx_n = np.exp(np.multiply(-1, np.dot(X_n, xnew)))
            # obj.e_wx_n += 1e-30  #segment

            fval_n = np.sum(np.log(1 + 1. / obj.e_wx_n))

            fval_loss = float(fval_p + fval_n) / obj.N_train

            fval_reg = obj.lmd * (la.norm(xnew)**2)

            fval = fval_loss + fval_reg

        return fval

    def evaldf(obj, xnew):

        if obj.method == 'zeroOne':
            obj.compute_moments_z(xnew)
            if obj.name_data not in obj.name_list:
                frac_p_1 = float(obj.mean_z_p) / np.sqrt(obj.cov_z_p)
                numinator_p = np.sqrt(obj.cov_z_p) * obj.mean_p - frac_p_1 * obj.cov_mean_p
                frac_p_2 = float(1) / obj.cov_z_p
                coeff_p = float(1) / np.sqrt(2 * np.pi)
                df_p = coeff_p * np.exp(-0.5 * (frac_p_1**2)) * frac_p_2 * numinator_p

                frac_n_1 = float(obj.mean_z_n) / np.sqrt(obj.cov_z_n)
                numinator_n = np.sqrt(obj.cov_z_n) * obj.mean_n - frac_n_1 * obj.cov_mean_n
                frac_n_2 = float(1) / obj.cov_z_n
                coeff_n = float(1) / np.sqrt(2 * np.pi)
                df_n = coeff_n * np.exp(-0.5 * (frac_n_1**2)) * frac_n_2 * numinator_n
                obj.df = obj.prob_n * df_n - obj.prob_p * df_p - 4 * 0.001 * (1 - (np.linalg.norm(xnew))**2) * xnew

                #- xnew / (np.linalg.norm(xnew))**3
            else:
                frac_p_1 = float(obj.mean_z_p) / np.sqrt(obj.cov_z_p)
                numinator_p = np.sqrt(obj.cov_z_p) * obj.mean_p - frac_p_1 * obj.cov_mean_p
                # print('mean_p shape', obj.mean_p.shape)
                frac_p_2 = float(1) / obj.cov_z_p
                coeff_p = float(1) / np.sqrt(2 * np.pi)
                df_p = coeff_p * np.exp(-0.5 * (frac_p_1**2)) * frac_p_2 * numinator_p

                frac_n_1 = float(obj.mean_z_n) / np.sqrt(obj.cov_z_n)
                numinator_n = np.sqrt(obj.cov_z_n) * obj.mean_n - frac_n_1 * obj.cov_mean_n
                frac_n_2 = float(1) / obj.cov_z_n
                coeff_n = float(1) / np.sqrt(2 * np.pi)
                df_n = coeff_n * np.exp(-0.5 * (frac_n_1**2)) * frac_n_2 * numinator_n
                obj.df = obj.prob_n * df_n - obj.prob_p * df_p - 4 * 0.001 * (1 - (np.linalg.norm(xnew))**2) * xnew
                obj.df = np.array(obj.df).reshape(-1,)
                # print('df type', type(obj.df))
                #- xnew / (np.linalg.norm(xnew))**3

        elif obj.method == 'logReg':

            # extract positive and negative samples
            X_p = obj.X_train_p
            X_n = obj.X_train_n

            B_p = - np.divide(1, (1 + obj.e_wx_p))
            df_p = np.transpose(np.dot(np.transpose(B_p), X_p))

            B_n = - np.divide(-1, (1 + obj.e_wx_n))
            df_n = np.transpose(np.dot(np.transpose(B_n), X_n))

            frac = float(1) / obj.N_train
            df = df_p + df_n

            obj.df = frac * df + 2 * obj.lmd * xnew

        obj.normdf = np.linalg.norm(obj.df)
        return obj.df

        # obj.normdf = np.amax(obj.df)

    def generate_minibatches(obj, data):

        np.random.shuffle(obj.X_train_p)
        batch_size_p = int(obj.N_train_p * obj.batch_size)
        list_p = range(0, obj.N_train_p)
        indices_p = random.sample(list_p, batch_size_p)
        obj.X_train_p = data.X_train_p[indices_p, :]

        np.random.shuffle(obj.X_train_n)
        batch_size_n = int(obj.N_train_n * obj.batch_size)
        list_n = range(0, obj.N_train_n)
        indices_n = random.sample(list_n, batch_size_n)
        obj.X_train_n = data.X_train_n[indices_n, :]

    def getH(obj, LH):
        obj.H = np.identity(LH.p + 1) * LH.gama - np.dot(LH.Q, LH.Q_bar)  # gama_update, initially is equal to gama, every sd iter plus gama

    def initx(obj, x0):
        obj.x = obj.initw
        obj.x_prev = obj.x

    def initf(obj):
        obj.fval = obj.evalf(obj.x)
        obj.fval_prev = obj.fval

    def initdf(obj):
        obj.evaldf(obj.x)
        obj.df_prev = obj.df

    def initH(obj):
        obj.H = np.identity(obj.p + 1)
