import csv
import numpy as np
import math
from solver import *
from dataReader import dataReader
from computeAccuracy import computeAccuracy
import scipy.io as sio
import os
import timeit

from numpy import linalg as la


class def_param:
    def __init__(self):

        self.num_run = 2
        self.lmd = 0.001
        self.max_iter = 500
        self.opt_tol = 1e-6
        self.opt_inc = 0
        self.mu_init = 1
        self.error = 0
        self.memory_lbfgs = 50
        self.batch_size = 0.2
        self.ra = 50
        #self.ra_list = [50, 100, 200, 300]

        """ Choose Method """
        # self.method = 'zeroOne'     #'logReg',
        self.method_list = ['logReg']

        """ Choose Algorithm """
        # self.algorithm = 'GD'
        # self.algorithm = 'SGD'
        self.algorithm = 'LBFGS'

        # self.list_data = ['fourclass', 'svm1', 'diabetes', 'shuttle',
        #                   'vowel', 'magic', 'poker', 'letter',
        #                   'segment', 'svm3', 'ijcnn', 'german',
        #                   'satimage', 'sonar', 'a9a',
        #                   'w8a', 'mnist', 'colon', 'gisette', 'covtype']

        # self.list_data = ['w8a','connect','mnist']
        # self.list_data = ['diabetes', 'letter','ijcnn']
        # self.list_data = ['mnist']
        # self.list_data = ['diabetes']
        self.list_data = ['diabetes']
        self.name_list = ['rcv1', 'real_sim']


def main():
    param = def_param()
    accuracy_final = 0
    accuracy_initial = 0
    time_final = 0

    for method in param.method_list:
        param.method = method
        for name in param.list_data:
            param.name_data = name

            """
            create result file
            """
            direct = '/Users/messi/Documents/summer18/results/result_' + param.method + '_test111/'
            print(direct)
            file_name = direct + 'result_' + param.name_data + '.csv'
            if not os.path.exists(direct):
                os.makedirs(direct)
            with open(file_name, 'w') as fp:
                a = csv.writer(fp, delimiter=',')
                #row_new = [['run', 'Accuracy_init', 'Accuracy_final', 'num. iters', 'soltime']]
                row_new = [['run', 'Accuracy_init', 'Accuracy_final', 'num. iters', 'soltime', 'moment_time']]
                a.writerows(row_new)
            m_accuracy_init = 0
            m_accuracy = 0
            m_num_iters = 0
            m_sol_time = 0
            m_moment = 0

            for run in range(16, 18):

                # run = 15
                data = dataReader(param, run + 1)

                print(run)
                print(param.method)
                print(param.name_data)

                print('Optimization process:')
                start_time = timeit.default_timer()
                w_final, num_iters = solver(param, data)
                end_time = timeit.default_timer()
                soltime_time = end_time - start_time

                accuracy_init = computeAccuracy(param, data, data.initw)
                print('Accuracy with initial w is: {}'.format(accuracy_init))

                accuracy = computeAccuracy(param, data, w_final)
                sol_time = soltime_time
                print('Accuracy after minimization is: {} with solution time :{}'.format(accuracy, sol_time))
                #row_new = [[run, accuracy_init, accuracy, num_iters, sol_time]]
                row_new = [[run, accuracy_init, accuracy, num_iters, sol_time, data.soltime_time_mom]]
                """
                write result file
                """
                m_accuracy_init += accuracy_init / param.num_run
                m_accuracy += accuracy / param.num_run
                m_num_iters += num_iters / param.num_run
                m_sol_time += sol_time / param.num_run
                m_moment += data.soltime_time_mom / param.num_run
                with open(file_name, 'a', newline="") as fp:
                    a = csv.writer(fp, delimiter=',')
                    a.writerows(row_new)
                if run == param.num_run - 1:
                    with open(file_name, 'a', newline="") as fp:
                        a = csv.writer(fp, delimiter=',')
                        row_new = [['average', m_accuracy_init, m_accuracy, m_num_iters, m_sol_time, m_moment]]
                        a.writerows(row_new)
                accuracy_initial += accuracy_init
                accuracy_final += accuracy

                time_final += sol_time

            print('Total initial accuracy is: {}'.format(float(accuracy_initial) / param.num_run))
            print('FINAL accuracy is: {} with solution time: {}'.format(float(accuracy_final) / param.num_run, float(time_final) / param.num_run))
            print(w_final)


if __name__ == '__main__':
    main()
