from tensorflow.keras import layers, models
import tensorflow as tf

# Carichiamo il dataset MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalizziamo i dati
x_train = x_train / 255.0
x_test = x_test / 255.0

# Creiamo il layer di ingresso per i dati di input
input_layer = layers.Input(shape=(28, 28))

# Creiamo la struttura della rete neurale
x = layers.Flatten()(input_layer)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dense(10, activation='softmax')(x)

# Creiamo il modello
model = models.Model(inputs=input_layer, outputs=x)

# Compiliamo il modello
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Eseguiamo il training del modello
model.fit(x_train, y_train, epochs=5, batch_size=32)

# Valutiamo il modello utilizzando i dati di test
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)
