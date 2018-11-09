#!/usr/bin/env python

# import modules used here -- sys is a very standard one
import sys
import numpy as np
import os
import subprocess
import shutil
import matplotlib.pyplot as plt

# Gather our code in a main() function
def synth_exp_tree():
    weight_list = [1, 2, 5, 10]
    log_dir ='/home/pankaj/SubmodularInference/data/working/25_10_2017/synthetic_tree_submod_tree2/'
    tree_file = 'synthetic_tree_20_2.txt' 
    f = open('/home/pankaj/scripts/perl_jobs/jobsQueued.txt', 'a')
    for weight in weight_list:
        for i in range(100):
            command = '/home/pankaj/SubmodularInference/code/denseCRF/build/examples/inference_sparse_synthetic_tree unary_' + str(i) + '.txt ' + tree_file + ' ' + str(weight) + ' 100 20 ' + log_dir + '\n'
            f.write(command)
    f.close()


def synth_exp():
    weight_list = [0]
    log_dir ='/home/pankaj/SubmodularInference/data/working/04_10_2017/submodular_w_0/'
    f = open('/home/pankaj/scripts/perl_jobs/jobsQueued.txt', 'a')
    for weight in weight_list:
        for i in range(100):
            command = '/home/pankaj/SubmodularInference/code/denseCRF/build/examples/inference_sparse_synthetic unary_' + str(i) + '.txt ' + str(weight) + ' 100 20 ' + log_dir + '\n'
            f.write(command)
    f.close()

def msrc_exp():
    imskip_list = [20, 10, 5, 1]
    dirw = r'/home/pankaj/SubmodularInference/data/working/28_02_2017'
    dirdata = r'/home/pankaj/SubmodularInference/code/denseCRF/data'
    for imskip in imskip_list:
        result_dir = dirw + '/submod_std_down_' + str(imskip) + '_bin_' + str(500)
        command='./inference ' + dirdata + '/2_14_s.bmp ' +  dirdata + '/2_14_s.c_unary submod ' + result_dir  + ' MSRC ' + str(imskip) 
        shutil.rmtree(result_dir)
        os.makedirs(result_dir)
        print command
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        process.wait()
    print "hello"

def good_bad_comparison_Potts_exp():
    weight_list = [1, 2, 5, 10]
    log_dir ='/home/pankaj/SubmodularInference/data/working/02_12_2017/'
    f = open('/home/pankaj/scripts/perl_jobs/jobsQueued.txt', 'w')
    for weight in weight_list:
        for i in range(20, 100):
            command = '/home/pankaj/SubmodularInference/code/denseCRF/build_synthetic/examples/inference_sparse_synthetic unary_' + str(i) + '.txt ' + str(weight) + ' 100 20 ' + log_dir + ' 0\n'
            f.write(command)
            command = '/home/pankaj/SubmodularInference/code/denseCRF/build_synthetic/examples/inference_sparse_synthetic unary_' + str(i) + '.txt ' + str(weight) + ' 100 20 ' + log_dir + ' 1\n'
            f.write(command)
    f.close()

def good_bad_comparison_tree1_exp():
    weight_list = [1, 2, 5, 10]
    log_dir ='/home/pankaj/SubmodularInference/data/working/02_12_2017/'
    tree_file = 'synthetic_tree_20_1.txt'
    f = open('/home/pankaj/scripts/perl_jobs/jobsQueued.txt', 'a')
    for weight in weight_list:
        for i in range(100):
            command = '/home/pankaj/SubmodularInference/code/denseCRF/build_synthetic_tree/examples/inference_sparse_synthetic_tree unary_' + str(i) + '.txt ' + tree_file + ' ' + str(weight) + ' 100 20 ' + log_dir + ' 0\n'
            f.write(command)
#            command = '/home/pankaj/SubmodularInference/code/denseCRF/build_synthetic_tree/examples/inference_sparse_synthetic_tree unary_' + str(i) + '.txt ' + tree_file + ' ' + str(weight) + ' 100 20 ' + log_dir + ' 1\n'
#            f.write(command)
    f.close()

def good_bad_comparison_tree2_exp():
    weight_list = [1, 2, 5, 10]
    log_dir ='/home/pankaj/SubmodularInference/data/working/02_12_2017/'
    tree_file = 'synthetic_tree_20_2.txt'
    f = open('/home/pankaj/scripts/perl_jobs/jobsQueued.txt', 'a')
    for weight in weight_list:
        for i in range(100):
            command = '/home/pankaj/SubmodularInference/code/denseCRF/build_synthetic_tree/examples/inference_sparse_synthetic_tree unary_' + str(i) + '.txt ' + tree_file + ' ' + str(weight) + ' 100 20 ' + log_dir + ' 0\n'
            f.write(command)
#            command = '/home/pankaj/SubmodularInference/code/denseCRF/build_synthetic_tree/examples/inference_sparse_synthetic_tree unary_' + str(i) + '.txt ' + tree_file + ' ' + str(weight) + ' 100 20 ' + log_dir + ' 1\n'
#            f.write(command)
    f.close()


def main():
    good_bad_comparison_Potts_exp()
    good_bad_comparison_tree1_exp()
    good_bad_comparison_tree2_exp()

if __name__ == '__main__':
    main()
