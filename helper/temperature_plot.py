#!/usr/bin/env python

import sys
import math
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

def main():
    mu = 0
    variance = 1

    sigma = math.sqrt(variance)
    x = np.linspace(mu-3*variance,mu+3*variance, 100)
    plt.plot(x,mlab.normpdf(x, mu, sigma), label = 'T1')
    plt.text(0, 0.735, 'T1', fontsize = 18) 

    variance = 0.5
    sigma = math.sqrt(variance)
    x = np.linspace(mu-5*variance,mu+5*variance, 1000)
    plt.plot(x,mlab.normpdf(x, mu, sigma), label = 'T2')
    plt.text(0, 0.58, 'T2', fontsize = 18) 

    variance = 0.3
    sigma = math.sqrt(variance)
    x = np.linspace(mu-10*variance,mu+10*variance, 1000)
    plt.plot(x,mlab.normpdf(x, mu, sigma), label = 'T3')
    plt.text(0, 0.42, 'T3', fontsize = 18) 


    plt.text(1.2, 0.7, 'T1 < T2 < T3', fontsize = 18) 

#    plt.tick_params(
#        axis='x',          # changes apply to the x-axis
#        which='both',      # both major and minor ticks are affected
#        bottom='off',      # ticks along the bottom edge are off
#        top='off',         # ticks along the top edge are off
#        labelbottom='off') # labels along the bottom edge are off

    plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off') # labels along the bottom edge are off
 
    plt.xlabel('x')
    plt.ylabel('P(x)')
    plt.savefig('/home/pankaj/SubmodularInference/paper/figures/temperature.png')
    plt.show()

    print 'Hello there', sys.argv[1]

if __name__ == '__main__':
    main()
