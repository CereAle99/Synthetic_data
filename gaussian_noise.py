import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import scipy.ndimage as ndimage
import scipy.signal as signal


# Caricamento del dataset MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], -1)
y_train = y_train.reshape(y_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
y_test = y_test.reshape(y_test.shape[0], -1)

# Normalizzazione dei dati
x_train, x_test = x_train / 255.0, x_test / 255.0


'''
# Visualizzare l'immagine numero i del dataframe df
df = x_train
single_image = df.iloc[0, :].values.reshape(28,28)
plt.imshow(single_image, cmap='gray')
plt.show()

#apply gaussian noise n times and show it at the end
n = 1000
beta = 
noise_image = single_image
for i in range(n):
    noise_image += np.random.normal(loc=0.0, scale=0.01, size=(28,28))
plt.imshow(noise_image, cmap='gray')
plt.show()
'''

#creation of the noisy file, and setting the number of iterations and the diffusion length
noise_x_train, noise_x_test = x_train, x_test
n = 1
beta = 0.01

#appling the gaussian filter to x_train n times
for i in range(x_train.shape[0]):
    single_image = noise_x_train[i, :].reshape(28,28)

    for j in range(n):
        single_image += beta * np.random.normal(loc=0.0, scale=1, size=(28,28))
    
    noise_x_train[i, :] = single_image.reshape(1,784)

'''
#appling the gaussian noise to x_test n times
for i in range(x_test.shape[0]):
    single_image = noise_x_test[i, :].reshape(28,28)

    for j in range(n):
        single_image += beta * np.random.normal(loc=0.0, scale=1, size=(28,28))
    
    noise_x_test[i, :] = single_image.reshape(1,784) 
'''

#check if the algorithm worked with the j^th image if the sample
j = 50
show_image = noise_x_train[j, :].reshape(28,28)
plt.imshow(show_image, cmap='gray')
plt.show()



#function which get the markov matrix from the gaussian noise matrix
gaussian_noise = np.random.normal(loc=0.0, scale=1, size=(28,28))
mean, std = 0, 1
def get_markov_gauss(matrix):
    markov_matrix =  (1 / ( std * np.sqrt(2*np.pi))) * np.e ** (-((matrix - mean)**2) / (2*std**2))
    return markov_matrix


