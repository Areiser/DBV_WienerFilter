# -*- coding: utf-8 -*-
"""
Spyder Editor

Dies ist eine temporäre Skriptdatei.
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

def dft(inputImage):
    dimension = inputImage.shape
    M = dimension[0]
    N = dimension[1]
    
    OutputFT = inputImage
    
    for m in range(M):
        print (m, " ")
        for n in range (N):
            print(n,  " ")
            sumReal = 0.0
            for k in range(M):
                print(k, " ")
                for l in range(N):
                    print(m, " ", n, " ", k, " ", l, " ")
                    constant = ((2*math.pi*k*m)/M + (2*math.pi*l*N)/N)
                    sumReal += inputImage[k][l] * math.cos(constant)
                    
            OutputFT[m][n] = sumReal
            
    return OutputFT

imgRGB = cv2.imread('input3.jpg')
img = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2GRAY)

Output = dft(img)

print(Output)