from defObj import *
from LBFGS import *
from numpy.linalg import inv
import csv
from computeAccuracy import computeAccuracy
from line_search_wolfe2 import *
import scipy.optimize as sop
import scipy


def solver(param, data):

    obj = defObj(param, data)
    lowH = LBFGS(param, data)
    obj.x_prev = 0
    obj.df_prev = 0
    obj.normdf0 = np.linalg.norm(obj.df)

    """ PLOT """
    """
    direct = '/home/hig213/RealData_ZeroOne/plot/'
    file_name = direct+ param.name_data + '_' + param.method + '.csv'
    with open(file_name, 'w') as fp:
        a = csv.writer(fp, delimiter=',')
        row_new = [['iter','Accuracy']]
        a.writerows(row_new)
    """
    idx = 0
    for iter_out in range(obj.max_iter):

        """ PLOT """
        """
        accuracy_test = computeAccuracy(param,data,obj.x)
        row_new = [[iter_out,accuracy_test]]
        with open(file_name, 'ab') as fp:
                a = csv.writer(fp, delimiter=',')
                a.writerows(row_new)
        """

        # if obj.algorithm == 'LBFGS':
        #     obj.d = lowH.lbfgsProd(obj.df)
        #     #print (obj.d)
        #     #print ('lbfgs_end is %d',lowH.lbfgs_end)

        # updateStates(param, obj)
        # # print(obj.x)
        # computeStep(obj)
        # # atuple = line_search_wolfe2(obj.evalf, obj.evaldf, obj.x, obj.d, c1=0.0001, c2=0.9,amax=4)
        # # alpha = atuple[0]
        # # print('alpha is %f' % alpha)
        # # #print('atuple is', atuple)
        # # obj.x = obj.x + alpha * obj.d
        # # obj.fval = obj.evalf(obj.x)

        # obj.evaldf(obj.x)

        # if obj.algorithm == 'LBFGS':
        #     lowH.lbfgsAdd(obj.x - obj.x_prev, obj.df - obj.df_prev)     # update matrix S, T, D and M

        if iter_out == 0:
            obj.d = -obj.df
            obj.s = -obj.df
        else:
            if obj.algorithm == 'SGD':
                obj.generate_minibatches(data)

            if obj.algorithm == ('LBFGS', 'PCG'):
                # print('first', (obj.x - obj.x_prev).shape)
                # print('second', (obj.df - obj.df_prev).shape)
                lowH.lbfgsAdd(obj.x - obj.x_prev, obj.df - obj.df_prev)
                if lowH.skipped == 1:
                    idx += 1
            print('skipped number is %d' % idx)

            # print ('x is ',obj.x)
            # print ('fval is ',obj.)

            if obj.algorithm == 'LBFGS':
                obj.d = lowH.lbfgsProd(obj.df)

            if obj.algorithm == 'PCG':

                obj.s = lowH.lbfgsProd(obj.df)

                beta = np.dot(obj.df, (obj.s - obj.s_prev)) / np.dot(obj.df_prev, obj.s_prev)
                obj.d = obj.s + beta * obj.d

        atuple = line_search_wolfe2(obj.evalf, obj.evaldf, obj.x, obj.d, c1=0.0001, c2=0.9)
        alpha = atuple[0]
        #print('alpha is %f')
        # print(alpha)
        obj.x = obj.x + alpha * obj.d
        obj.fval = obj.evalf(obj.x)

        obj.evaldf(obj.x)
        printStates(obj, iter_out)
        isoptimal(obj)
        if obj.flag_opt:
            print(obj.message_opt)
            break

        #printStates(obj, iter_out)
        updateStates(param, obj)

    return obj.x, iter_out


def printStates(obj, iter):
    print('Iteration: {} Objective Function: {} Increase: {} df_norm: {}'.format(iter, obj.fval, obj.fval_prev - obj.fval, obj.normdf))


def isoptimal(obj):
    if (obj.normdf < obj.gtol * obj.normdf0):
        obj.flag_opt = True
        obj.message_opt = 'Terminated due to small gradient norm'
    elif (abs(obj.fval_prev - obj.fval) <= obj.inctol):  # inctol is set to 0
        obj.flag_opt = True
        obj.message_opt = 'Terminated due to small change in objective function value'


def updateStates(param, obj):

    obj.fval_prev = obj.fval
    obj.x_prev = obj.x
    obj.df_prev = obj.df
    obj.s_prev = obj.s

    obj.mu = param.mu_init


"""
def evalq(obj):
    qval = obj.fval + np.dot(obj.df,obj.d) + 0.5 * obj.mu * np.dot(obj.d,obj.d)
    return qval
"""


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

    #qval = evalq(obj)
    xtrial = obj.x + (float(1) / obj.mu) * obj.d

    fval = obj.evalf(xtrial)
    f_trial = fval

    """ check the Wolfe conditions"""
    difference = f_trial - (obj.fval + (float(1) / obj.mu) * 1e-4 * np.dot(obj.df, obj.d))

    if obj.algorithm == 'SGD':
        difference = 0

    if (difference <= 0):
        obj.sd_flag = True
        obj.x = xtrial
        obj.fval = fval
