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

x_train = pd.DataFrame(x_train)
y_train = pd.DataFrame(y_train)

# Normalizzazione dei dati
x_train, x_test = x_train / 255.0, x_test / 255.0

# Visualizzare l'immagine numero i del dataframe df
i, df = 6, x_train
single_image = df.iloc[0, :].values.reshape(28,28)
plt.imshow(single_image, cmap='gray')
plt.show()

#apply gaussian noise n times and show it at the end
n = 500
convolved_image = single_image
for i in range(n):
    convolved_image += np.random.normal(loc=0.0, scale=0.01, size=(28,28))
plt.imshow(convolved_image, cmap='gray')
plt.show()



#print(x_train)