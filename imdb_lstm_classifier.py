
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Parameters
words = 20000
max_length = 100
embed_size = 128

# Load dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=words)
x_train = pad_sequences(x_train, maxlen=max_length)
x_test = pad_sequences(x_test, maxlen=max_length)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(words, embed_size, input_shape=(x_train.shape[1],)),
    tf.keras.layers.LSTM(units=128, activation='tanh'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

model.summary()

# Compile & train
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=128)

# Evaluate
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test accuracy: {:.2f}%".format(test_accuracy * 100))
