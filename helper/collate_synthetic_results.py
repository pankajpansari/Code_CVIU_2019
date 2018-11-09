#!/usr/bin/env python

import sys
import math
import numpy as np
import numpy.random as npr
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def good_bad_potts(bound_ratio_good, bound_ratio_bad):
    weights = [1, 2, 5, 10]

    for j in range(len(weights)):
        bound_diff = 0
        bound_diff2 = 0
        print "weight = " + str(weights[j])
        for i in range(100):
            file_submod_good = '/home/pankaj/SubmodularInference/data/output/synthetic/alternate_extension/potts/good/submodular_log_w_' + str(weights[j]) + '.000000_1_unary_' + str(i) + '.txt'
#            file_submod_good_2= '/home/pankaj/SubmodularInference/data/output/synthetic/submodular_potts/submodular_log_w_' + str(weights[j]) + '_unary_' + str(i) + '.txt'
            file_trw = '/home/pankaj/SubmodularInference/data/output/synthetic/trw_potts/trw_unary_' + str(i) + '_w_' + str(weights[j]) + '.txt'
            file_submod_bad = '/home/pankaj/SubmodularInference/data/output/synthetic/alternate_extension/potts/bad/submodular_log_w_' + str(weights[j]) + '.000000_0_unary_' + str(i) + '.txt'

            temp = np.loadtxt(file_submod_good)
            bound_submod_good = temp[-1, 1]

            temp = np.loadtxt(file_submod_bad)
            bound_submod_bad = temp[-1, 1]

            temp = np.loadtxt(file_trw)
            bound_trw = temp[1]

#            print (bound1 - bound2)/abs(bound2)
            bound_diff += (bound_submod_good - bound_trw)/abs(bound_trw)
            bound_diff2 += (bound_submod_bad - bound_trw)/abs(bound_trw)

        bound_ratio_good[j] = bound_diff/100
        bound_ratio_bad[j] = bound_diff2/100

def good_bad_tree1(bound_ratio_good, bound_ratio_bad):
    weights = [1, 2, 5, 10]

    for j in range(len(weights)):
        bound_diff = 0
        bound_diff2 = 0
        print "weight = " + str(weights[j])
        for i in range(100):
            file_submod_good = '/home/pankaj/SubmodularInference/data/output/synthetic/submodular_tree/submodular_log_w_' + str(weights[j]) + '_unary_' + str(i) + '_rhst_20_1.txt'
            file_trw = '/home/pankaj/SubmodularInference/data/output/synthetic/trw_tree/trw_unary_' + str(i) + '_w_' + str(weights[j]) + '_rhst_20_1.txt'
            file_submod_bad = '/home/pankaj/SubmodularInference/data/output/synthetic/alternate_extension/tree1/bad/submodular_log_w_' + str(weights[j]) + '.000000_0_unary_' + str(i) + '_synthetic_tree_20_1.txt'

            temp = np.loadtxt(file_submod_good)
            bound_submod_good = temp[-1, 1]

            temp = np.loadtxt(file_submod_bad)
            bound_submod_bad = temp[-1, 1]

            temp = np.loadtxt(file_trw)
            bound_trw = temp[1]

            bound_diff += (bound_submod_good - bound_trw)/abs(bound_trw)
            bound_diff2 += (bound_submod_bad - bound_trw)/abs(bound_trw)

        bound_ratio_good[j] = bound_diff/100
        bound_ratio_bad[j] = bound_diff2/100


def good_bad_tree2(bound_ratio_good, bound_ratio_bad):
    weights = [1, 2, 5, 10]


    for j in range(len(weights)):
        bound_diff = 0
        bound_diff2 = 0
        print "weight = " + str(weights[j])
        for i in range(100):
            file_submod_good = '/home/pankaj/SubmodularInference/data/output/synthetic/submodular_tree2/submodular_log_w_' + str(weights[j]) + '.000000_unary_' + str(i) + '_synthetic_tree_20_2.txt'
            file_trw = '/home/pankaj/SubmodularInference/data/output/synthetic/trw_tree2/trw_unary_' + str(i) + '_w_' + str(weights[j]) + '_rhst_20_2.txt'
            file_submod_bad = '/home/pankaj/SubmodularInference/data/output/synthetic/alternate_extension/tree2/bad/submodular_log_w_' + str(weights[j]) + '.000000_0_unary_' + str(i) + '_synthetic_tree_20_2.txt'

            temp = np.loadtxt(file_submod_good)
            bound_submod_good = temp[-1, 1]

            temp = np.loadtxt(file_submod_bad)
            bound_submod_bad = temp[-1, 1]

            temp = np.loadtxt(file_trw)
            bound_trw = temp[1]

            bound_diff += (bound_submod_good - bound_trw)/abs(bound_trw)
            bound_diff2 += (bound_submod_bad - bound_trw)/abs(bound_trw)

        bound_ratio_good[j] = bound_diff/100
        bound_ratio_bad[j] = bound_diff2/100


def main():

    bound_ratio_good_potts = np.zeros(4)
    bound_ratio_bad_potts = np.zeros(4)

    bound_ratio_good_tree = np.zeros(4)
    bound_ratio_bad_tree = np.zeros(4)

    bound_ratio_good_tree2 = np.zeros(4)
    bound_ratio_bad_tree2 = np.zeros(4)

    good_bad_potts(bound_ratio_good_potts, bound_ratio_bad_potts)
    good_bad_tree1(bound_ratio_good_tree, bound_ratio_bad_tree)
    good_bad_tree2(bound_ratio_good_tree2, bound_ratio_bad_tree2)


    weights = [1, 2, 5, 10]

    potts_good, = plt.plot(weights, bound_ratio_good_potts, color = 'b', marker = 'o', markersize = 8, label='Potts Optimal Extension')
    potts_bad, = plt.plot(weights, bound_ratio_bad_potts, color = 'b', marker = 'o', ls = '--', markersize = 8, label='Potts Alternate Extension')
    tree1_good, = plt.plot(weights, bound_ratio_good_tree, color = 'r', marker = 'D', markersize = 8, label='Hierarchical Tree 1 Optimal Extension')
    tree1_bad, = plt.plot(weights, bound_ratio_bad_tree, color = 'r', marker = 'D' , ls = '--', markersize = 8, label='Hierarchical Tree 1 Alternate Extension')
    tree2_good, = plt.plot(weights, bound_ratio_good_tree2, color = 'g', marker = 's', markersize = 8, label='Hierarchical Tree 2 Optimal Extension')
    tree2_bad, = plt.plot(weights, bound_ratio_bad_tree2, color = 'g', marker = 's', ls = '--', markersize = 8, label='Hierarchical Tree 2 Alternate Extension')

    plt.legend(loc=1, handlelength = 6)
    plt.axhline(0, ls='--')
    plt.xlabel('Weight')
    plt.ylabel('(Submodular bound - TRW bound)/|TRW bound|')
    plt.savefig('synthetic_upper_bound_comparison.pdf')
    plt.show()

if __name__ == '__main__':
    main()
