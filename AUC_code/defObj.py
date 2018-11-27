import numpy as np
from numpy import linalg as la
from scipy.stats import norm
import timeit


class defObj:

    def __init__(obj, param, data):
        obj.max_iter = param.max_iter
        obj.gtol = param.opt_tol
        obj.inctol = param.opt_inc
        obj.p = param.dim

        obj.method = param.method

        obj.initw = data.initw

        obj.X_p = data.X_train_p
        obj.X_n = data.X_train_n

        obj.N_p = data.N_train_p
        obj.N_n = data.N_train_n

        obj.algorithm = param.algorithm
        obj.name_list = param.name_list
        obj.name_data = param.name_data

        if (param.method == 'cdf'):
            obj.mean_p = data.mean_p
            obj.mean_n = data.mean_n

            obj.cov_pp = data.cov_pp
            obj.cov_nn = data.cov_nn

            obj.mean_hat = obj.mean_n - obj.mean_p
            obj.cov_hat = obj.cov_pp + obj.cov_nn

            # obj.mean_z = 0
            # obj.cov_z = 0

        obj.H = np.identity(obj.p + 1)
        obj.iters = 0
        obj.iter_back = 0
        obj.mu = param.mu_init

        obj.flag_opt = False
        obj.message_opt = ' '

        obj.initx(obj.initw)
        obj.initf()
        obj.initdf()

    def compute_moments_z(obj, xnew):
        if obj.name_data not in obj.name_list:
            obj.mean_z = np.dot(xnew, obj.mean_hat)               # w mu_hat
            obj.cov_mean_z = np.dot(xnew, obj.cov_hat)            # w cov_hat
            obj.cov_z = np.dot(obj.cov_mean_z, xnew)              # w cov_hat w
        else:
            obj.mean_z = obj.mean_hat.dot(xnew.transpose())           # w cov_pp          # w cov_pp w
            obj.cov_mean_z = obj.cov_hat.dot(xnew.transpose())             # w cov_nn
            obj.cov_z = obj.cov_mean_z.transpose().dot(xnew.transpose())            # w cov_nn w

    # compute function value
    def evalf(obj, xnew):

        if obj.method == 'hinge':

            start_time_f = timeit.default_timer()
            out_p = np.dot(obj.X_p, xnew)  # N_p-by-1 vector
            out_n = np.dot(obj.X_n, xnew)  # N_n-by-1 vector
            loss = 0
            sum_p = 0
            num_neg = 0
            xpre = 0

            out_n += 1
            out_all = np.hstack((out_p, out_n))  # check size
            sort_all = np.sort(out_all)
            idx_all = np.argsort(out_all)
            N_all = obj.N_n + obj.N_p
            for i in range(N_all):
                if idx_all[N_all - 1 - i] > obj.N_p - 1:  # meaning this is from negtive
                    sum_p += num_neg * (xpre - sort_all[N_all - 1 - i])
                    xpre = sort_all[N_all - 1 - i]
                    num_neg += 1
                else:
                    sum_p += num_neg * (xpre - sort_all[N_all - 1 - i])
                    loss += sum_p
                    xpre = sort_all[N_all - 1 - i]

            fval = (float(loss) / (obj.N_p * obj.N_n))
            end_time_f = timeit.default_timer()
            soltime_time_f = end_time_f - start_time_f

            # print 'fval is: {} and time for computing fval is: {}'.format(fval, soltime_time_f)

        else:

            obj.compute_moments_z(xnew)
            frac = float(obj.mean_z) / np.sqrt(obj.cov_z)
            fval = - norm.cdf(frac) + 0.001 * (1 - (np.linalg.norm(xnew))**2)**2

        return fval

    def evaldf(obj, xnew):
        if obj.method == 'hinge':

            start_time_df = timeit.default_timer()
            obj.df = np.zeros(obj.p + 1)
            out_p = np.dot(obj.X_p, xnew)  # N_p-by-1 vector
            out_n = np.dot(obj.X_n, xnew)  # N_n-by-1 vector
            loss = 0
            sum_p = 0
            num_neg = 0
            dfpre = np.zeros(obj.p + 1)

            out_n += 1
            out_all = np.hstack((out_p, out_n))  # check size
            #sort_all = np.sort(out_all)
            idx_all = np.argsort(out_all)
            N_all = obj.N_n + obj.N_p
            for i in range(N_all):
                curr_idx = idx_all[N_all - 1 - i]
                if curr_idx > obj.N_p - 1:  # meaning this is from negtive
                    sum_p += num_neg * (dfpre - obj.X_n[curr_idx - obj.N_p])
                    dfpre = obj.X_n[curr_idx - obj.N_p]
                    num_neg += 1
                else:
                    sum_p += num_neg * (dfpre - obj.X_p[curr_idx])
                    obj.df += sum_p
                    dfpre = obj.X_p[curr_idx]

            obj.df = obj.df / (obj.N_p * obj.N_n)
            end_time_df = timeit.default_timer()
            soltime_time_df = end_time_df - start_time_df

            obj.normdf = np.linalg.norm(obj.df)

        else:
            obj.compute_moments_z(xnew)
            frac_1 = float(obj.mean_z) / np.sqrt(obj.cov_z)

            numinator = np.sqrt(obj.cov_z) * obj.mean_hat - frac_1 * obj.cov_mean_z
            frac_2 = float(1) / obj.cov_z

            coeff = float(1) / np.sqrt(2 * np.pi)

            obj.df = coeff * np.exp(-0.5 * (frac_1**2)) * frac_2 * numinator

            obj.df *= -1
            obj.df -= 0.001 * 4 * (1 - (np.linalg.norm(xnew))**2) * xnew
            if obj.name_data in obj.name_list:
                obj.df = np.array(obj.df).reshape(-1,)

        obj.normdf = np.linalg.norm(obj.df)
        return obj.df

    def initx(obj, x0):
        obj.x = obj.initw
        obj.x_prev = obj.x

    def initf(obj):
        obj.fval = obj.evalf(obj.x)
        obj.fval_prev = obj.fval

    def initdf(obj):
        obj.evaldf(obj.x)
        obj.df_prev = obj.df
