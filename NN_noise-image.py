import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt 
import scipy.ndimage as ndimage
import scipy.signal as signal
from keras import layers, models
import tensorflow as tf


# Caricamento del dataset MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#shaping 3D matrices as 2D, each row is an image
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# Normalizzazione dei dati
x_train, x_test = x_train / 255.0, x_test / 255.0


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
    
#appling the gaussian noise to x_test n times
for i in range(x_test.shape[0]):
    single_image = noise_x_test[i, :].reshape(28,28)

    for j in range(n):
        single_image += beta * np.random.normal(loc=0.0, scale=1, size=(28,28))
    
    noise_x_test[i, :] = single_image.reshape(1,784) 

'''
#reshaping the 2D matrices back as 3D, each row is a 28x28 bidimensional image (both x and noise_x)
noise_x_train = x_train.reshape(noise_x_train.shape[0], 28, -1)
x_train = x_train.reshape(x_train.shape[0], 28, -1)
noise_x_test = x_test.reshape(noise_x_test.shape[0], 28, -1)
x_test = x_test.reshape(x_test.shape[0], 28, -1)
'''



# Creiamo il layer di ingresso per i dati di input
input_layer = layers.Input(shape=(784))

# Creiamo la struttura della rete neurale, sono tre layer nascosti e un 
# layer di output e il primo argomento sono i dnodi del layer
x = layers.Flatten()(input_layer)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dense(784, activation='softmax')(x)

# Creiamo il modello
model = models.Model(inputs=input_layer, outputs=x)

# Compiliamo il modello
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])#I use the mean squared error because it allows to have vectors of shape (784) as lables

# Eseguiamo il training del modello, il batch_size è il 
# numero di esempi che usa ogni volta per aggiornare i pesi
model.fit(noise_x_train, x_train, epochs=5, batch_size=32)

# Valutiamo il modello utilizzando i dati di test
test_loss, test_acc = model.evaluate(noise_x_test, x_test)
print('Test accuracy:', test_acc)


#check if the algorithm worked with the j^th image if the sample
j = 50
try_image = x_train[j, :] + beta * np.random.normal(loc=0.0, scale=1, size=(784))
probabilities = model.predict(try_image[np.newaxis, ...])
probabilities = probabilities.reshape(28,28)
plt.imshow(try_image.reshape(28,28), cmap='gray')
plt.show()

plt.imshow(probabilities, cmap='gray')
plt.show()

