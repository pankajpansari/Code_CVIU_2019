#!/usr/bin/env python

import sys
import math
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import os
import glob

def main():

    str1 = '/Stereo_special_mf_1.640000_12.990000_50.840000_1.510000_1.100000/view1_abs_diff_16_mf_1.640000_12.990000_50.840000_1.510000_1.100000.txt'
    str2 = '/Stereo_special_submod_1.640000_12.990000_50.840000_1.510000_1.100000/view1_abs_diff_16_submod_1.640000_12.990000_50.840000_1.510000_1.100000.txt'
    y = glob.glob('/home/pankaj/SubmodularInference/code/denseCRF/data/new_stereo_pairs/*')
    e_ratio_sum = 0
    t_ratio_sum = 0
    for x in y:
        print x
        file1 = x + str1
        file2 = x + str2

        f1 = open(file1, "r")
        last_line = f1.readlines()[-1]
        last_line = last_line.strip()
        p1 = last_line.split('\t')
        energy1 = float(p1[1])
        time1 = float(p1[0])
        f1.close()

        f1 = open(file2, "r")
        last_line = f1.readlines()[-1]
        last_line = last_line.strip()
        p1 = last_line.split('\t')
        energy2 = float(p1[1])
        time2 = float(p1[0])
        f1.close()

        e_ratio = energy2/energy1
        t_ratio = time2/time1
        print e_ratio
        e_ratio_sum += e_ratio
        t_ratio_sum += t_ratio

    print "average ratio = " , e_ratio_sum/len(y), t_ratio_sum/len(y)
if __name__ == '__main__':
    main()
