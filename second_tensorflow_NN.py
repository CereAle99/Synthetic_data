from keras import layers, models
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Carichiamo il dataset MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalizziamo i dati
x_train = x_train / 255.0
x_test = x_test / 255.0

# Creiamo il layer di ingresso per i dati di input
input_layer = layers.Input(shape=(28, 28))

# Creiamo la struttura della rete neurale, sono tre layer nascosti e un 
# layer di output e il primo argomento sono i dnodi del layer
x = layers.Flatten()(input_layer)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dense(10, activation='softmax')(x)

# Creiamo il modello
model = models.Model(inputs=input_layer, outputs=x)

# Compiliamo il modello
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Eseguiamo il training del modello, il batch_size è il 
# numero di esempi che usa ogni volta per aggiornare i pesi
model.fit(x_train, y_train, epochs=5, batch_size=32)

# Valutiamo il modello utilizzando i dati di test
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

#we use the model on a brand new image
file_path = "/Users/aless/OneDrive/Desktop/immagine.txt"
data = np.genfromtxt(file_path, delimiter=';') #the file must contain just a row with 784 elements
data = data.reshape(28,28)
data = np.nan_to_num(data)
#show the new image and print the NN's guess and the probability
plt.imshow(data, cmap='gray')
plt.show()
probabilities = model.predict(data[np.newaxis, ...])
class_idx = np.argmax(probabilities[0])
print(class_idx, "  p: ", probabilities[0,class_idx])