#!/usr/bin/env python

#python plotMarginals.py

# import modules used here -- sys is a very standard one
import sys
import math 
import numpy as np
import csv
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
import seaborn as sns

# Gather our code in a main() function
def main():
    f = open(sys.argv[1], 'rb')
    marg = []
    try:
        reader = csv.reader(f)
        for row in reader:
            a = [float(i) for i in row]
            marg.append(a)
    finally:
        f.close()
    x = marg[0]
    a1 = int(x[0])
    a2 = int(x[1])
    marg = marg[1:]
    t = np.array(marg)
    print t.shape
    entropy = np.zeros(a1*a2) 
    count = 0
    epsilon = 1e-8
    for col in range(t.shape[1]):
        prob = t[:, col]
        entropy[col] = - np.sum(prob * np.log(prob + epsilon))
    entropy_mat = entropy.reshape(a1, a2)
    entropy_plot = 255.0/12 * entropy_mat 
    plt.imshow(entropy_rev)
#    plt.show()
    plt.savefig(sys.argv[2])
#    for column in t:
#        entropyVal = 0
#        for p in column:
#            entropyVal = entropyVal + (-p * math.log(p + epsilon, 2))
#        entropy_vec[count] = entropyVal
#        count = count + 1
#    b = np.amax(t, axis = 1)
##    entropy_mat = entropy_vec.reshape(a1, a2)
##    trace = go.Heatmap(entropy_vec)
##    data=[trace]
##    py.iplot(data, filename='basic-heatmap')
##    print b 
##    plt.imshow(b, cmap = 'hot', interpolation='nearest')
##    plt.show()
##    sns.heatmap(entropy_mat, cmap ='binary', xticklabels=False, yticklabels=False)
#    print a1, a2
#    b = b.reshape(a1, a2)
#    print b.shape 
##    sns.heatmap(b, cmap ='binary', xticklabels=False, yticklabels=False)
#    sns.heatmap(b, xticklabels=False, yticklabels=False)
#    plt.show()

if __name__ == '__main__':
    main()
