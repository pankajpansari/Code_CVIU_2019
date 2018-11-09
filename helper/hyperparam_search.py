#!/usr/bin/env python

# import modules used here -- sys is a very standard one
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import subprocess

random.seed(200)
dirw = '/home/pankaj/SubmodularInference/data/working/24_10_2017/'
# Gather our code in a main() function
def print_jobs_tsukuba(image, option):

    spc_std_end = 10 
    spc_potts_end = 150 
    bil_colstd_end = 70 
    bil_spcstd_end = 10 
    bil_potts_end =  20 

    results_path = dirw + 'stereo_' + image + '/'
    path_to_executable = '/home/pankaj/SubmodularInference/code/denseCRF/build/examples/inference_tree'
    path_image = '/home/pankaj/SubmodularInference/code/denseCRF/data/stereo/' + image + '_left.png'
    path_unary = '/home/pankaj/SubmodularInference/code/denseCRF/data/stereo/confFileStereo_' + image + '.txt'
    path_tree = '/home/pankaj/SubmodularInference/data/input/trees/short/tsukuba_tree1.txt'
    dataset_name = 'Stereo_special'
    method = 'submod_tree'

    f = open('/home/pankaj/scripts/perl_jobs/jobsQueued.txt', option)

    for a in range(100):
        spc_std = random.uniform(0, spc_std_end)
        spc_potts = random.uniform(0, spc_potts_end)
        bil_spcstd = random.uniform(0, bil_spcstd_end)
        bil_colstd = random.uniform(0, bil_colstd_end)
        bil_potts = random.uniform(0, bil_potts_end)

        f.write(path_to_executable + " " + path_image + " " + path_unary + " " + path_tree + " " + method + " " + results_path + " " + dataset_name + " " + str(spc_std) + " " + str(spc_potts) + " " + str(bil_spcstd) + " " + str(bil_colstd) + " " + str(bil_potts) + "\n") 

    f.close()



def print_jobs_house(image, option):
    spc_std_end = math.log(3e6, 10)
    spc_potts_end = math.log(3e6, 10)
    bil_colstd_end = math.log(3e6, 10)
    bil_spcstd_end = math.log(3e6, 10)
    bil_potts_end = math.log(3e6, 10)

    results_path = dirw + 'denoising_' + image + '/'
#    path_to_executable = '/home/pankaj/SubmodularInference/code/denseCRF/build/examples/inference_denoising'
    path_to_executable = '/home/pankaj/SubmodularInference/code/denseCRF/build/examples/inference'
    path_image = '/home/pankaj/SubmodularInference/code/denseCRF/data/denoising/' + image + '-input.png'
    path_unary = '/home/pankaj/SubmodularInference/code/denseCRF/data/denoising/confFileInpainting_' + image + '.txt'
    dataset_name = 'Denoising'
    method = 'mf'

    f = open('/home/pankaj/scripts/perl_jobs/jobsQueued.txt', option)

    for a in range(100):
        spc_std = random.uniform(2, spc_std_end)
        spc_potts = random.uniform(2, spc_potts_end)
        bil_spcstd = random.uniform(2, bil_spcstd_end)
        bil_colstd = random.uniform(2, bil_colstd_end)
        bil_potts = random.uniform(2, bil_potts_end)

        f.write(path_to_executable + " " + path_image + " " + path_unary + " " + method + " " + results_path + " " + dataset_name + " " + str(math.pow(10, spc_std)) + " " + str(math.pow(10, spc_potts)) + " " + str(math.pow(10, bil_spcstd)) + " " + str(math.pow(10, bil_colstd)) + " " + str(math.pow(10, bil_potts)) + "\n") 

    f.close()

def print_jobs_house_uniform(image, option):

    spc_std_end = 4 
    spc_potts_end = 150 
    bil_colstd_end = 80 
    bil_spcstd_end = 5 
    bil_potts_end =  15 

    results_path = dirw + 'denoising_' + image + '/'
#    path_to_executable = '/home/pankaj/SubmodularInference/code/denseCRF/build/examples/inference_denoising'
    path_to_executable = '/home/pankaj/SubmodularInference/code/denseCRF/build/examples/inference'
    path_image = '/home/pankaj/SubmodularInference/code/denseCRF/data/denoising/' + image + '-input.png'
    path_unary = '/home/pankaj/SubmodularInference/code/denseCRF/data/denoising/confFileInpainting_' + image + '.txt'
    dataset_name = 'Denoising'
    method = 'submod'

    f = open('/home/pankaj/scripts/perl_jobs/jobsQueued.txt', option)

    for a in range(100):
        spc_std = random.uniform(0, spc_std_end)
        spc_potts = random.uniform(0, spc_potts_end)
        bil_spcstd = random.uniform(0, bil_spcstd_end)
        bil_colstd = random.uniform(0, bil_colstd_end)
        bil_potts = random.uniform(0, bil_potts_end)

        f.write(path_to_executable + " " + path_image + " " + path_unary + " " + method + " " + results_path + " " + dataset_name + " " + str( spc_std) + " " + str( spc_potts) + " " + str( bil_spcstd) + " " + str( bil_colstd) + " " + str( bil_potts) + "\n") 

    f.close()


def print_jobs_penguin(image, option):
    spc_std_end = math.log(100, 10)
    spc_potts_end = math.log(100, 10)
    bil_colstd_end = math.log(100, 10)
    bil_spcstd_end = math.log(100, 10)
    bil_potts_end = math.log(100, 10)

    results_path = dirw + 'denoising_' + image + '/'
    path_to_executable = '/home/pankaj/SubmodularInference/code/denseCRF/build/examples/inference_denoising'
    path_image = '/home/pankaj/SubmodularInference/code/denseCRF/data/denoising/' + image + '-input.png'
    path_unary = '/home/pankaj/SubmodularInference/code/denseCRF/data/denoising/confFileInpainting_' + image + '.txt'
    dataset_name = 'Denoising'
    method = 'mf'

    f = open('/home/pankaj/scripts/perl_jobs/jobsQueued.txt', option)

    for a in range(20):
        spc_std = random.uniform(0, spc_std_end)
        spc_potts = random.uniform(0, spc_potts_end)
        bil_spcstd = random.uniform(0, bil_spcstd_end)
        bil_colstd = random.uniform(0, bil_colstd_end)
        bil_potts = random.uniform(0, bil_potts_end)

        f.write(path_to_executable + " " + path_image + " " + path_unary + " " + method + " " + results_path + " " + dataset_name + " " + str(math.pow(10, spc_std)) + " " + str(math.pow(10, spc_potts)) + " " + str(math.pow(10, bil_spcstd)) + " " + str(math.pow(10, bil_colstd)) + " " + str(math.pow(10, bil_potts)) + "\n") 

    f.close()


def main():
#    print_jobs_penguin('penguin', 'w')
#    print_jobs_house_uniform('house', 'w')
    print_jobs_tsukuba('tsukuba', 'w')

if __name__ == '__main__':
    main()
