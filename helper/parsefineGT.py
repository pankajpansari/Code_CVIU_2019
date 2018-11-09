import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import re

def main():
    filename = '/home/pankaj/SubmodularInference/data/input/msrc/split/FineTest.txt'
    newfilename = '/home/pankaj/SubmodularInference/data/input/msrc/split/FineTestTrue.txt'
    f = open(filename, 'rU')
    fTrue = open(newfilename, 'w')
    for line in f:
        start = line.find('_GT')
        fTrue.write(line[:start]  + '.bmp\n')
        
    f.close()
    fTrue.close()

if __name__ == '__main__':
    main()
