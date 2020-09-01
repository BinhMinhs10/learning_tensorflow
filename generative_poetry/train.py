from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import os
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras import layers
from pyvi import ViTokenizer
import re

tokenizer = Tokenizer()

f = open("corpus.txt", "r")
corpus = [ViTokenizer.tokenize(re.sub(r'\d+|\.','', sentence)) for sentence in f.read().lower().split("\n") if (sentence != "")]
f.close()

tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1
# print(corpus)

input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    # generate n gram
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = 56
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding="pre"))

# label for sentence is last word
xs = input_sequences[:, :-1]
labels = input_sequences[:, -1]
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

# Define a simple sequential model
model = tf.keras.Sequential([
    layers.Embedding(total_words, 240, input_length=max_sequence_len-1),
    layers.Bidirectional(layers.LSTM(150)),
    layers.Dense(total_words, activation="softmax")
])


# checkpoint_path = "training_1/cp.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)
#
# # Create a callback that saves the model's weights
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                  save_weights_only=True,
#                                                  verbose=1)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(xs, ys, epochs=35, verbose=1)

if not os.path.isdir('model'):
    os.mkdir('model')
model.save('model/gen_poetry.h5')




