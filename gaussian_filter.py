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
y_train, y_test = y_train / 255.0, y_test / 255.0


'''
# Visualizzare l'immagine numero i del dataframe df
i, df = 6, x_train
single_image = df[0, :].reshape(28,28)
plt.imshow(single_image, cmap='gray')
plt.show()


#apply gaussian filter n times and show it at the end
n = 20
convolved_image = single_image
kernel = ndimage.gaussian_filter(np.ones([3,3]), sigma=1)
for i in range(n):
    convolved_image = signal.convolve2d(convolved_image, kernel, mode='same')

plt.imshow(convolved_image, cmap='gray')
plt.show()
'''


