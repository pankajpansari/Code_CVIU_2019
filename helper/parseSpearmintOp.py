#!/usr/bin/env python

# import modules used here -- sys is a very standard one
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from tabulate import tabulate

method_list = ['MF5', 'Submod', 'ProxLP-Acc']
nSamples = 7 

def findBestSample(currentPath):
    file_list = os.listdir(currentPath)
    file_list = sorted(file_list)
    file_list = file_list[0:10]
     
    dictAcc = {}
    for item in file_list:
        f = open(currentPath + '/' + item, 'rU')
        searchStr = 'Percentage of pixels correctly labelled overall:'  
        for line in f:
            if line.find(searchStr) != -1:
                match = re.search('\d+\.\d+',line)
                dictAcc[item] = float(match.group())
        f.close()

    index = dictAcc.values().index(max(dictAcc.values()))
    print '        |     ' + str(dictAcc.values()[index])
    return dictAcc.values()[index]
     
def countSamples(currentPath):
    print len([name for name in os.listdir(currentPath)])

def getAccuracies():
    print '\n# of samples = ' + str(nSamples) + '\n \n'
    for method in method_list:
        print 'Method = ' + method
        print 'Fold     |  Best accuracy'
        accuracySum = 0
        for fold in range(5):
            print fold + 1,
            results_path = '/home/pankaj/SubmodularInference/code/denseCRF/examples/AllParameters-' + method + '/fold_' + str(fold + 1) + '/output'
            accuracy = findBestSample(results_path)
            accuracySum += accuracy 
        print "Average best accuracy = " + str(accuracySum/5) + '\n\n'


def getSampleCount():
    print 'Total # of samples computed \n'
    for method in method_list:
       print 'Method = ' + method
       print 'Fold     |   #Samples'
       for fold in range(5):
           print str(fold + 1) + '\t \t',
           results_path = '/home/pankaj/SubmodularInference/code/denseCRF/examples/AllParameters-' + method + '/fold_' + str(fold + 1) + '/output'
           countSamples(results_path)
       print 
    
def main():
    getSampleCount()
    print('-'*40)
    getAccuracies()   

if __name__ == '__main__':
    main()
