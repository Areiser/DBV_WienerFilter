# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 15:28:43 2019

@author: Olivier
"""


import cv2
import numpy as np
from matplotlib import pyplot as plt


imgRGB = cv2.imread('input2.jpg')
img = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2GRAY)
#cv2.imshow('Original', img)

sigma = 1

dimension= img.shape
noise =  np.random.normal(0, sigma, size=(dimension[0], dimension[1]))
noise = noise.reshape(img.shape[0],img.shape[1]).astype('uint8')

print (noise)
size = 150

# generating the kernel
kernel_motion_blur = np.zeros((size, size))
kernel_motion_blur[int((size-1)), :] = np.ones(size)
kernel_motion_blur = kernel_motion_blur / size


# applying the kernel to the input image
imageWithMotionBlur = cv2.filter2D(img, -1, kernel_motion_blur)
output2 = cv2.filter2D(img, -1, kernel_motion_blur2)

#Adding the noise to the blurred image
output3 =  cv2.add(imageWithMotionBlur, noise)
noisy_output = cv2.cvtColor(output3, cv2.COLOR_GRAY2RGB)


#Displaying the different Images
print("Original Image")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
plt.show()

print("\n\n\nImage with Motion Blur")
plt.imshow(cv2.cvtColor(imageWithMotionBlur, cv2.COLOR_GRAY2RGB))
plt.show()

print("\n\n\nMotion Blur + Noise")
plt.imshow(noisy_output)
plt.show()
#plt.imshow(output3)


#cv2.imshow('Motion Blur2', output2)
#cv2.waitKey(30000)