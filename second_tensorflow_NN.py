# Importiamo TensorFlow
import tensorflow as tf

# Carichiamo il dataset MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalizziamo i dati
x_train = x_train / 255.0
x_test = x_test / 255.0

# Creiamo i placeholders per i dati di input e di output
x = tf.placeholder(tf.float32, [None, 28, 28])
y = tf.placeholder(tf.int32, [None])

# Creiamo le variabili per i pesi e i bias della rete neurale
W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.random_normal([10]))

# Creiamo la struttura della rete neurale
logits = tf.matmul(tf.reshape(x, [-1, 784]), W) + b
predictions = tf.nn.softmax(logits)

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)

# Inizializziamo le variabili
init = tf.global_variables_initializer()

# Eseguiamo il training della rete neurale
with tf.Session() as sess:
    sess.run(init)
    for step in range(1000):
        _, l = sess.run([optimizer, loss], feed_dict={x: x_train, y: y_train})
        if step % 100 == 0:
            print("Loss: {}".format(l))

# Valutiamo la rete neurale utilizzando i dati di test
test_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predictions, 1), y), tf.float32))
with tf.Session() as sess:
    sess.run(init)
    print("Test accuracy: {}".format(sess.run(test_accuracy, feed_dict={x: x_test, y: y_test})))
