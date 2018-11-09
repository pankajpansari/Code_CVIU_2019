#!/usr/bin/env python

# import modules used here -- sys is a very standard one
import sys
import os 
import numpy as np
import matplotlib.pyplot as plt

# Gather our code in a main() function
def generateLongFileTree(L, M):
    parts = {}
    nodeNum = 0
    for i in range(0, L):
        parts[nodeNum] = [i]
        nodeNum = nodeNum + 1     

    for i in range(0, M):
#       print "i = " + str(i)
       parts[nodeNum] = range(0, i + 1)
       nodeNum = nodeNum + 1
       for j in range(i + 1, L, M):
           parts[nodeNum] = range(j, min(j + M, L))
           nodeNum = nodeNum + 1
#           print 
#    print parts
    nMeta = len(parts)

    print nMeta, L
    #print path to label
    for i in range(0, L):
        print i,
        for j in parts.keys():
            if i in parts[j]:
                print j,
        print
   
   #print list of leaves
    for j in parts.keys():
        print j,
        for t in parts[j]:
            print t,
        print
#        print [t for t in parts[j]] 
    
    #print edge len
    for i in range(0, L):
        print i, 0

    for i in range(L, nMeta):
        print i, 0.5

def generateShortFilesTree(nlabel, M):
    dir_name = "/home/pankaj/SubmodularInference/data/input/tests/trees/truncated_l1_L_" + str(nlabel) + "_M_" + str(M) + "/"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    labels = range(0, nlabel)
    print labels

    #partition leaves/labels
    for i in range(0, M):
        label_list = []
        filename = dir_name + "/tree_" + str(i) + ".txt"
        print filename
        f = open(filename, 'w')
        if i > 0:
            label_list.append(labels[0:i])
        labels2 = labels[i:]    
        for j in range(0, len(labels2), M):
            label_list.append(labels2[j:min(j + M, nlabel)])

        nvertices = nlabel + len(label_list) + 1

        #print to file

        f.write(str(nvertices) + '\n')
        #for root
        f.write( str(nlabel) + " 0.5" + " ")
        f.write( ' '.join(map(str, range(nlabel + 1, nlabel + 1 + len(label_list)))) + '\n')

        for item in range(1,  len(label_list) + 1):
            f.write( str(nlabel + item) + " 0" + " ")
            f.write(  ' '.join(map(str, label_list[item - 1])) + '\n')
        f.close()

def main():
    generateLongFileTree(256, 10)

if __name__ == '__main__':
    main()
