#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 09:22:41 2020

@author: soonmi
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

image = np.array(Image.open('IMG_2247.jpg').convert('L'))
print(type(image))
print(image.shape)
print(image.size)


U, s, VT = np.linalg.svd(image)

x_axis = np.array(range(0,image.shape[1]))
plt.scatter(x_axis,s, 3)
plt.axis([0,500,0,500000])
plt.xlabel('Modes')
plt.ylabel('Singular Value')
plt.title('Singular Values')
plt.show()

x_axis = np.array(range(0,image.shape[1]))
plt.scatter(x_axis,s, 3)
plt.axis([0,100,0,500000])
plt.xlabel('Modes')
plt.ylabel('Singular Value')
plt.title('Singular Values')
plt.show()


print(U.shape, s.shape, VT.shape)

S = np.zeros((image.shape[0], image.shape[1]))

for i in range (image.shape[1]):
    S[i][i] = s[i]

#print(S)

n_modes = np.array([5, 10, 20, 40, 54, 87, 220])

for i in range(n_modes.size):
    n_mode = n_modes[i]

    S1 = S[:, :n_mode]
    VT1 = VT[:n_mode,:]
    
    A = U.dot(S1.dot(VT1))
    
    #print(A)
    plt.matshow(A)
    plt.title('modes: ' + str(n_modes[i]))

plt.matshow(image)
plt.title('Original Picture')

#print(VT1.shape, S1.shape)
print('Size of Original Singular Value Matrix: ' + str(S.shape[1]) + ' by ' +str(S.shape[1]))
print('Size of New Singular Value Matrix: ' + str(S1.shape[1]) + ' by ' + str(S1.shape[1]))
print('Compression Ratio: ' + str(float(S.shape[1])/float(S1.shape[1])))

new_image = plt.imsave('new_image_87.jpg', A)