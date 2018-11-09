#!/usr/bin/env python

# import modules used here -- sys is a very standard one
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Gather our code in a main() function
def main():
#    print 'Hello there', sys.argv[1]
     a = np.loadtxt(sys.argv[1])
     b = np.loadtxt(sys.argv[2])
     plt.plot(b, a, linestyle='-') 
#     b = np.loadtxt(sys.argv[2])
#     pp = PdfPages(sys.argv[2])
#     print(a)
#     plt.plot(a, marker='o', linestyle='-')
#     plt.plot(a[0:50, 0], linestyle='-')
#     plt.plot(b[0:50, 0], linestyle='-')
#     plt.plot(a[0:], a[1:], linestyle='-')
#
#     plt.xlabel("Time(s)")
#     plt.ylabel("Value of objective function")
#     plt.title("Objective function in Frank Wolfe")
     plt.show()
#     pp.savefig()
#     plt.close()

#     plt.plot(a[1:100, 1], linestyle='-')
#     plt.xlabel("Iterations")
#     plt.ylabel("Value of objective function")
#     plt.title("Objective function in Frank Wolfe (Zoomed In)")
#    
#     pp.savefig()
#     pp.close()
#     plt.close()
#     print a
     
    # Command line args are in sys.argv[1], sys.argv[2] ...
    # sys.argv[0] is the script name itself and can be ignored

# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
    main()
