from defObj import *
from numpy.linalg import inv
import csv
from computeAUC import computeAUC
import os
from LBFGS import *
from line_search_wolfe2 import *


def solver(param, data):

    obj = defObj(param, data)
    lowH = LBFGS(param, data)
    obj.x_prev = 0
    obj.df_prev = 0
    obj.normdf0 = np.linalg.norm(obj.df)

    direct = 'Users/messi/Documents/Year1/summer18/plot'
    file_name = direct + param.name_data + '_' + param.method + '.csv'
    if not os.path.exists(direct):
        os.makedirs(direct)

    with open(file_name, 'w') as fp:
        a = csv.writer(fp, delimiter=',')
        row_new = [['iter', 'AUC']]
        a.writerows(row_new)

    for iter_out in range(obj.max_iter):

        AUC_test = computeAUC(param, data, obj.x)
        row_new = [[iter_out, AUC_test]]
        with open(file_name, 'a') as fp:
            a = csv.writer(fp, delimiter=',')
            a.writerows(row_new)

        if iter_out == 0:
            obj.d = -obj.df
            obj.s = -obj.df
        else:
            if obj.algorithm == ('LBFGS', 'PCG'):
                # print('first', (obj.x - obj.x_prev).shape)
                # print('second', (obj.df - obj.df_prev).shape)
                lowH.lbfgsAdd(obj.x - obj.x_prev, obj.df - obj.df_prev)

            """
            if obj.fval < 1e-05:
                print 'Terminated due to small fval value'
                break
            """

            if obj.algorithm == 'LBFGS':
                obj.d = lowH.lbfgsProd(obj.df)

            if obj.algorithm == 'PCG':

                obj.s = lowH.lbfgsProd(obj.df)

                beta = np.dot(obj.df, (obj.s - obj.s_prev)) / np.dot(obj.df_prev, obj.s_prev)
                obj.d = obj.s + beta * obj.d

        # computeStep(obj)
        atuple = line_search_wolfe2(obj.evalf, obj.evaldf, obj.x, obj.d, c1=0.0001, c2=0.9)
        alpha = atuple[0]
        #print('alpha is %f')
        # print(alpha)
        obj.x = obj.x + alpha * obj.d
        obj.fval = obj.evalf(obj.x)

        obj.evaldf(obj.x)

        isoptimal(obj)
        if obj.flag_opt:
            print(obj.message_opt)
            break
        printStates(obj, iter_out)
        updateStates(param, obj)

    return obj.x, iter_out


def printStates(obj, iter):
    if obj.method == 'hinge':
        print('Iteration: {} Objective Function: {} Decrease: {} df_norm: {}'.format(iter, obj.fval, obj.fval_prev - obj.fval, obj.normdf))
    else:
        print('Iteration: {} Objective Function: {} Decrease: {} df_norm: {}'.format(iter, obj.fval, obj.fval_prev - obj.fval, obj.normdf))


def isoptimal(obj):
    if (obj.normdf < obj.gtol * obj.normdf0):
        obj.flag_opt = True
        obj.message_opt = 'Terminated due to small gradient norm'
    elif (abs(obj.fval_prev - obj.fval) < obj.inctol):
        obj.flag_opt = True
        obj.message_opt = 'Terminated due to small change in objective function value'


def updateStates(param, obj):
    obj.iters = obj.iters + 1

    if obj.iters == 1:
        obj.normdf0 = obj.normdf

    obj.fval_prev = obj.fval
    obj.x_prev = obj.x
    obj.df_prev = obj.df
    obj.s_prev = obj.s

    obj.mu = param.mu_init


def evalq(obj):
    qval = obj.fval + np.dot(obj.df, obj.d) + 0.5 * obj.mu * np.dot(obj.d, obj.d)
    return qval


def computeStep(obj):
    obj.sd_iters = 0

    # sufficientDecrease(obj)

    while True:
        sufficientDecrease(obj)

        if obj.sd_flag:
            break

        obj.mu = 2 * obj.mu

        obj.sd_iters = obj.sd_iters + 1

    obj.iter_back = obj.iter_back + obj.sd_iters


def sufficientDecrease(obj):
    obj.sd_flag = False

    if obj.algorithm == 'LBFGS':
        obj.d = np.linalg.solve(obj.H, -obj.df)   # d = -H^{-1} * df
    else:
        obj.d = - obj.df

    qval = evalq(obj)
    xtrial = obj.x + (float(1) / obj.mu) * obj.d

    fval = obj.evalf(xtrial)
    f_trial = fval

    # print 'fval is: {}'.format(obj.fval)
    # print 'ftrial is: {}'.format(f_trial)

    difference = f_trial - (obj.fval + (float(1) / obj.mu) * 1e-4 * np.dot(obj.df, obj.d))
    if (difference <= 0):
        obj.sd_flag = True

        obj.x = xtrial
        obj.fval = fval
