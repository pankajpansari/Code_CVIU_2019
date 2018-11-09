#!/usr/bin/env python

from __future__ import division
# import modules used here -- sys is a very standard one
import sys
import numpy as np
import matplotlib.pyplot as plt
import testSubmodular

# Gather our code in a main() function
def main():
   
    #read image
    imgf = open('../data/img1.txt', 'r')
    numLine = imgf.read()
    numList = map(float, numLine.split())
    height = int(numList[0])
    width = int(numList[1])
    img = np.zeros((height, width, 3))
    counter = 2
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(3):
                img[i, j, k] = numList[counter]
                counter = counter + 1
    imgf.close()
#    print img

    #read unary
    unaryf = open('../data/unary1.txt', 'r')
    numLine = unaryf.read()
    numList = map(float, numLine.split())
    M = int(numList[0])
    N = int(numList[1])
    unaries = np.zeros((M, N));
    counter = 2
    for i in range(unaries.shape[0]):
        for j in range(unaries.shape[1]):
            unaries[i, j] = numList[counter]
            counter = counter + 1
    unaryf.close()
#    print unaries

    #read Q
    Qf = open('../data/Q1.txt', 'r')
    numLine = Qf.read()
    numList = map(float, numLine.split())
    M = int(numList[0])
    N = int(numList[1])
    Q = np.zeros((M, N));
    counter = 2
    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            Q[i, j] = numList[counter]
            counter = counter + 1
    Qf.close()
#    print Q     

#    #random image
#    height = 2
#    width = 2
#    M = 3 
#    N = height * width 
# 
#    img = np.random.rand(height, width, 3)*100
#    char_img = np.zeros(height*width*3)
#    imgf = open('img1.txt', 'w')
#    imgf.write(str(height) + " " + str(width) + " ")
#    for i in range(img.shape[0]):
#        for j in range(img.shape[1]):
#            for k in range(3):
#                imgf.write("{:.5f}".format(img[i, j, k]) + " ")
#                char_img[i*img.shape[1] + j + k] = img[i, j, k]
#    imgf.close()
#    
#    #random unary
#    unaries = np.random.rand(M, N)*100;
#    unaryf = open('unary1.txt', 'w')
#    unaryf.write(str(M) + " " + str(N) + " ")
#    for i in range(unaries.shape[0]):
#        for j in range(unaries.shape[1]):
#            unaryf.write("{:.5f}".format(unaries[i, j]) + " ")
#    unaryf.close()
#
#    #random Q
#    Q = np.random.rand(M, N)*100;
#    Qf = open('Q1.txt', 'w')
#    Qf.write(str(M) + " " + str(N) + " ")
#    for i in range(Q.shape[0]):
#        for j in range(Q.shape[1]):
#            Qf.write("{:.5f}".format(Q[i, j]) + " ")
#    Qf.close()
#
    #get negative gradient

    featureArr = testSubmodular.getFeatureArr(img)
#    print Q
    for k in range(1):
        negGrad = testSubmodular.getNegGrad(unaries)
        Qs = testSubmodular.greedyAlgorithm(featureArr, unaries, negGrad) 
        step = float(2/(k + 3))
        Q = Q + step*(Qs - Q) 
   #     print "step = " + str(step)
   #     print "Q"
   #     print Q
   #     print "Qs"
   #     print Qs
        print "After iteration " + str(k) + " obj function =  " + str(testSubmodular.getObj(Q))
#    print unaries
#    print Qs
#    print testSubmodular.computeGaussianWeight([2, 3, 1, 4, 7], [8, 14, 9, 2, 1])
#    nIter = 1000
#    objArr = np.zeros(nIter)
#    for i in range(nIter):
#        Q = Q + 1*testSubmodular.getNegGrad(Q)
#        objArr[i] = testSubmodular.getObj(Q)
##        print "At iteration " + str(i) + ": " + str(testSubmodular.getObj(Q)) 
#
#    plt.plot(objArr)
#    plt.show()

if __name__ == '__main__':
    main()
