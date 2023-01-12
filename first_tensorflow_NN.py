import tensorflow as tf
import matplotlib.pyplot as plt

# Caricamento del dataset MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalizzazione dei dati
x_train, x_test = x_train / 255.0, x_test / 255.0

# Creazione del modello
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

# Compilazione del modello
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])



# Valutazione del modello
model.evaluate(x_test, y_test, verbose=2)



# Addestramento del modello
history = model.fit(x_train, y_train, epochs=5)

# Estrazione dei dati di history
acc = history.history['accuracy']
loss = history.history['loss']

# Creazione del grafico
plt.figure(figsize=[8,6])
plt.plot(loss,'r',linewidth=2.0)
plt.plot(acc,'b',linewidth=2.0)
plt.legend(['Training Loss', 'Training Accuracy'],fontsize=18)
plt.xlabel('Epochs',fontsize=16)
plt.ylabel('Accuracy/Loss',fontsize=16)
plt.title('Training Accuracy and Loss',fontsize=16)
