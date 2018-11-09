#!/usr/bin/env python

# import modules used here -- sys is a very standard one
import sys
import numpy as np
import os
import subprocess
import shutil
import matplotlib.pyplot as plt

# Gather our code in a main() function
def stereo_exp():
    log_dir ='/home/pankaj/SubmodularInference/data/working/28_11_2017/'
    f = open('/home/pankaj/scripts/perl_jobs/jobsQueued.txt', 'w')
    for t in os.listdir('/home/pankaj/SubmodularInference/code/denseCRF/data/new_stereo_pairs/'):
        i = '/home/pankaj/SubmodularInference/code/denseCRF/data/new_stereo_pairs/' + t
        print i
        command1 = '/home/pankaj/SubmodularInference/code/denseCRF/build/examples/inference ' + i + '/view1_abs_diff_16.png ' + i + '/abs_diff_16_unary.txt mf ' + i + ' Stereo_special 1.64 12.99 50.84 1.51 1.10 \n'
        command2 = '/home/pankaj/SubmodularInference/code/denseCRF/build/examples/inference ' + i + '/view1_abs_diff_16.png ' + i + '/abs_diff_16_unary.txt submod ' + i + ' Stereo_special 1.64 12.99 50.84 1.51 1.10 \n'
#        command2 = '/home/pankaj/SubmodularInference/code/denseCRF/build/examples/inference ' + i + '/view1.png ' + i + '/unary.txt submod ' + i + ' Stereo_special 1.64 12.99 50.84 1.51 1.10 \n'
        f.write(command1)
        f.write(command2)
    f.close()

def main():
    stereo_exp()

if __name__ == '__main__':
    main()
