import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import scipy.ndimage as ndimage
import scipy.signal as signal


# Caricamento del dataset MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#shaping data (an image for each line)
x_train = x_train.reshape(x_train.shape[0], -1)
y_train = y_train.reshape(y_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
y_test = y_test.reshape(y_test.shape[0], -1)


# Normalizzazione dei dati
x_train, x_test = x_train / 255.0, x_test / 255.0


'''
#Visualizzare l'immagine numero i del dataframe df
i, df = 6, x_train
single_image = df[1, :].reshape(28,28)
plt.imshow(single_image, cmap='gray')
plt.show()


#apply gaussian filter n times and show it at the end, for that single image
n = 20
convolved_image = single_image
kernel = ndimage.gaussian_filter(np.ones([3,3]), sigma=1)
for i in range(n):
    convolved_image = signal.convolve2d(convolved_image, kernel, mode='same')

plt.imshow(convolved_image, cmap='gray')
plt.show()
'''

#creating a gaussian filter 
size = 5
sigma = 0.5
kernel = signal.gaussian(size, sigma) #1D
kernel = np.outer(kernel, kernel) #2D

conv_x_train, conv_x_test = np.empty_like(x_train), np.empty_like(x_test)
n = 25

#appling the gaussian filter to x_train n times    
for i in range(x_train.shape[0]):
    single_image = x_train[i, :].reshape(28,28)

    for j in range(n):
        single_image = signal.convolve2d(single_image, kernel, mode='same')
    
    conv_x_train[i, :] = single_image.reshape(1,784)


#appling the gaussian filter to x_test n times
for i in range(x_test.shape[0]):
    single_image = x_test[i, :].reshape(28,28)

    for j in range(n):
        single_image = signal.convolve2d(single_image, kernel, mode='same')
    
    conv_x_test[i, :]= single_image.reshape(1,784)


single_image = x_train[1, :].reshape(28,28)
plt.imshow(single_image, cmap='gray')
plt.show()


single_image = conv_x_train[1, :].reshape(28,28)
plt.imshow(single_image, cmap='gray')
plt.show()


#BUILDING THE RELATIVE TRANSITION MATRIX

#creating an outer frame of zeros
#funcion that makes the frame of zeros n times
#def add_frame(matrix):
#    n = len(matrix)
#    new_matrix = [[0] * (n + 2) for i in range(n + 2)]
#    for i in range(n):
#        for j in range(n):
#            new_matrix[i + 1][j + 1] = matrix[i][j]
#    return new_matrix
#
#for i in range(size//2):
#    kernel = add_frame(kernel)


#creating the transition matrix 
dim = x_train.shape[1]
transition_matrix = np.empty((dim,dim))

for j in range(dim):
    
    for i in range(dim):
        if(j % 28 <= 1):
            shiftx = 2 - (j % 28)
        elif(j % 28 >=26):
            shiftx = 

        row = i % 28 + shiftx
        column = i // 28 + shifty

        transition_matrix[i,j] = kernel[row,column]

