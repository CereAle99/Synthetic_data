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


#creating a gaussian filter 
kernel = ndimage.gaussian_filter(np.ones([3,3]), sigma=1)


# Creazione della matrice di transizione con distribuzione gaussiana
mean = 0
std = 1
transition_matrix = np.random.normal(mean, std, (28, 28))

'''
#uniform transition matrix
transition_matrix = np.random.uniform(size=(28,28))
'''

# Normalizzazione della matrice di transizione
transition_matrix = np.abs(transition_matrix)
for i in range(transition_matrix.shape[0]):
    transition_matrix[i,:] = transition_matrix[i,:] / np.sum(transition_matrix[i,:])


'''
#check if something is wrong with visualization of matices
plt.imshow(transition_matrix, cmap='gray')
plt.show()

single_image = conv_x_train[50, :].reshape(28,28)
plt.imshow(single_image, cmap='gray')
plt.show()

single_image = transition_matrix @ single_image
plt.imshow(single_image, cmap='gray')
plt.show()
'''



conv_x_train, conv_x_test = x_train, x_test
n = 1

#appling the gaussian filter to x_train n times
for i in range(x_train.shape[0]):
    single_image = conv_x_train[i, :].reshape(28,28)

    for j in range(n):
        new_state = transition_matrix @ single_image
        single_image = new_state
    
    conv_x_train[i, :] = single_image.reshape(1,784)


#appling the gaussian filter to x_test n times
for i in range(x_test.shape[0]):
    single_image = conv_x_test[i, :].reshape(28,28)

    for j in range(n):
        new_state = transition_matrix @ single_image
        single_image = new_state
    
    conv_x_test[i, :]= single_image.reshape(1,784) 


#check if the algorithm worked
single_image = conv_x_train[50, :].reshape(28,28)
plt.imshow(single_image, cmap='gray')
plt.show()
