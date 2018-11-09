#!/usr/bin/env python

# import modules used here -- sys is a very standard one
import sys
import numpy as np
import matplotlib.pyplot as plt

# Gather our code in a main() function
def write_unary():
    height = 200
    width = 300
    f = open('trial.txt', 'w') 
    f.write(str(height*width) + " " + str(2))
    f.write("1")
    f.write("1")
    for i in range(height):
        for j in range(width):
            f.write("100 0\n")
    f.close()

def main():
    write_unary() 
    print "Hello, world!"

if __name__ == '__main__':
    main()
