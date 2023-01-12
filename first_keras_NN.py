from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=32)

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

classes = model.predict(x_test, batch_size=128)

# Prepara i dati di input per la previsione
#input_data = np.array([[0.1, 0.2, 0.3]])

# Fai la previsione utilizzando il modello
#prediction = model.predict(input_data)

#print(prediction)