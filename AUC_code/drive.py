import csv
import numpy as np
import math
from solver import *
from dataReader import dataReader
from computeAUC import computeAUC
import scipy.io as sio
import os
import timeit


class def_param:
    def __init__(self):

        self.max_iter = 100
        self.opt_tol = 1e-7
        self.opt_inc = 1e-15
        self.mu_init = 1
        self.num_run = 5

        self.method_list = ['cdf']
        #self.method = 'cdf'
        self.algorithm = 'PCG'
        self.memory_lbfgs = 20

        self.list_data = ['diabetes']
#        'fourclass', 'svm1', 'diabetes', 'shuttle',
#                          'vowel', 'magic', 'poker', 'letter',
#                          'segment', 'svm3', 'ijcnn', 'german',
#                          'satimage', 'sonar', 'a9a', 'colon', 'w8a',  'mnist', 'gisette', 'w8a', 'a9a', 'mnist',
        #self.list_data = ['w8a', 'a9a', 'mnist']
        #self.list_data = ['w8a']
        # self.list_data = ['gisette']

        # """
        # self.list_data = ['fourclass','diabetes','vowel','segment',
        #                     'german','sonar','svm1','svm3']
        #                     """
        # self.list_data = ['satimage', 'letter', 'poker']
        # #self.list_data = ['magic','shuttle','ijcnn','a9a']\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

        # #self.list_data = ['w8a','mnist']
        #self.list_data = ['colon']
        # self.list_data = ['segment']
        self.name_list = ['rcv1', 'real_sim']


def main():
    param = def_param()

    for method in param.method_list:
        param.method = method
        for name in param.list_data:
            param.name_data = name

            """
            creat result file
            """
            direct = '/Users/messi/Documents/Year1/summer18/results_auc/result_' + param.method + '_test2/'
            file_name = direct + 'result_' + param.name_data + '.csv'
            if not os.path.exists(direct):
                os.makedirs(direct)
            auc_final = 0
            auc_init = 0
            time_final = 0
            with open(file_name, 'w') as fp:
                a = csv.writer(fp, delimiter=',')
                row_new = [['run', 'auc_init', 'auc_final', 'num. iters', 'soltime', 'moment_time']]
                a.writerows(row_new)
            m_accuracy_init = 0
            m_accuracy = 0
            m_num_iters = 0
            m_sol_time = 0
            m_moment = 0

            for run in range(param.num_run):

                data = dataReader(param, run + 1)

                start_time = timeit.default_timer()
                w_final, num_iters = solver(param, data)
                end_time = timeit.default_timer()
                soltime_time = end_time - start_time

                print('solution time is: {}'.format(soltime_time))

                fval_auc_init = computeAUC(param, data, data.initw)
                print('auc value with initial w is: {}'.format(fval_auc_init))

                fval_auc = computeAUC(param, data, w_final)
                sol_time = soltime_time
                print('auc value for this run is : {} with solution time :{}'.format(fval_auc, sol_time))

                row_new = [[run, fval_auc_init, fval_auc, num_iters, sol_time, data.soltime_time_mom]]
                m_accuracy_init += fval_auc_init / param.num_run
                m_accuracy += fval_auc / param.num_run
                m_num_iters += num_iters / param.num_run
                m_sol_time += sol_time / param.num_run
                m_moment += data.soltime_time_mom / param.num_run
                auc_init += fval_auc_init
                auc_final += fval_auc
                time_final += sol_time
                if run == 0:
                    row_new_0 = row_new

                """
                write result file
                """
                with open(file_name, 'a', newline="") as fp:
                    a = csv.writer(fp, delimiter=',')
                    a.writerows(row_new)
                if run == param.num_run - 1:
                    with open(file_name, 'a', newline="") as fp:
                        a = csv.writer(fp, delimiter=',')
                        row_new = [['average', m_accuracy_init, m_accuracy, m_num_iters, m_sol_time, m_moment]]
                        a.writerows(row_new)

            print('this is ', param.name_data)
            print('initial auc is: {}'.format(float(auc_init) / param.num_run))
            print('final auc is: {}'.format(float(auc_final) / param.num_run))
            print('final solution time is: {}'.format(float(time_final) / param.num_run))


if __name__ == '__main__':
    main()
