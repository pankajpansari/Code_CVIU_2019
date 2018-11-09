#!/usr/bin/env python

# import modules used here -- sys is a very standard one
import sys
import numpy as np
import matplotlib.pyplot as plt
import os

# Gather our code in a main() function
def main():
    file_list = os.listdir('/home/pankaj/SubmodularInference/data/input/msrc/fine_annot')
    for item in file_list:
        print item
    file_list_new = []
    validation_len = 18
    tempfilename = '/home/pankaj/SubmodularInference/data/input/msrc/split/temp.txt'
    tempfile = open(tempfilename, 'w')
    for i in range(0,len(file_list)):
        file_name = file_list[i]
        start = file_name.find('_GT')
        file_list_new.append(file_name[:start]  + '.bmp')

    for i in range(0,len(file_list)):
        tempfile.write(file_list_new[i] + '\n')
    
    tempfile.close()

    for j in range(1, 6):
        vstart = (j - 1)*validation_len
        vend = j*validation_len
        vset = file_list_new[vstart:vend]
        tset = file_list_new[0:vstart] + file_list_new[vend:] 
        vfilename =  '/home/pankaj/SubmodularInference/data/input/msrc/split/Validation' + str(j) + '.txt'
        tfilename =  '/home/pankaj/SubmodularInference/data/input/msrc/split/Train' + str(j) + '.txt'
        print vfilename
        print tfilename
        vfile = open(vfilename, 'w')
        tfile = open(tfilename, 'w')
        for i in range(0,len(vset)):
            vfile.write(vset[i] + '\n')

        for i in range(0,len(tset)):
            tfile.write(tset[i] + '\n')

        vfile.close()
        tfile.close()
     

if __name__ == '__main__':
    main()
