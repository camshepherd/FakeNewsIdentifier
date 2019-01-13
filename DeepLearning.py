import spacy
import tensorflow as tf
import numpy as np
import itertools
import time
from nltk.corpus import stopwords
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers import Embedding
from keras.preprocessing import text
from keras.preprocessing import sequence
from keras import *
import pandas as pd

nlp = spacy.load('en_core_web_lg')

class DeepIdentifier:
    def do_RNN_stuff(self, epochs, neurons, dropout, activation_type, num_training, vocab_size, patience, max_length=1000):
        the_stopwords = stopwords.words("english")
        start = time.time()
        num_cores = 6
        num_CPU = 1
        num_GPU = 1
        config = tf.ConfigProto(intra_op_parallelism_threads=num_cores, \
                                inter_op_parallelism_threads=num_cores, allow_soft_placement=False, \
                                device_count={'CPU': num_CPU, 'GPU': num_GPU})
        session = tf.Session(config=config)
        backend.set_session(session)

        raw_data = pd.read_csv("news_ds.csv")
        training_data = raw_data.iloc[0:num_training]
        t = text.Tokenizer(vocab_size)

        t.fit_on_texts(training_data["TEXT"])


        matrix = np.ndarray(shape=(vocab_size,300))
        tokens = []
        ref = 0
        thingy = sorted([x for x in t.word_counts if x[0] not in the_stopwords],key=t.word_counts.get, reverse=True)

        for word in itertools.islice(thingy,vocab_size):
            matrix[ref] = np.array([nlp(word)[0].vector])
            ref += 1


        # integer encode the documents
        encoded_docs = t.texts_to_sequences(training_data["TEXT"])
        padded_docs = sequence.pad_sequences(encoded_docs, maxlen=max_length, padding='post')

        e = Embedding(vocab_size, 300, weights=np.array([matrix]), input_length=max_length, trainable=False)
        model = Sequential()
        model.add(e)
        callback1 = callbacks.EarlyStopping(monitor='acc',
                                      min_delta=0,
                                      patience=patience,
                                      verbose=0, mode='max')
        callback2 = callbacks.ModelCheckpoint("superthing.txt")

        model.add(SimpleRNN(neurons))
        model.add(Dropout(dropout))
        model.add(Dense(1, activation=activation_type))


        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

        model.fit(padded_docs, training_data["LABEL"], epochs=epochs, verbose=0, callbacks = [callback1,callback2])

        testing_data = raw_data.iloc[num_training + 1:]

        t.fit_on_texts(testing_data["TEXT"])

        encoded_docs2 = t.texts_to_sequences(testing_data["TEXT"])
        padded_testing_docs = sequence.pad_sequences(encoded_docs2, maxlen=max_length, padding='post')

        loss, accuracy = model.evaluate(padded_testing_docs, testing_data["LABEL"], verbose=0)
        print(
            "epochs: %d, neurons: %d, dropout: %f, activation: %s, num_training: %d, vocab_size: %d, patience: %d,  max_length: %d" % (
            epochs, neurons, dropout, activation_type, num_training, vocab_size, patience, max_length))

        print('Accuracy: %f' % (accuracy * 100))
        end = time.time()
        print("Total time = %d" % (end - start))


    def do_LSTM_stuff(self, epochs, neurons, dropout, activation_type, num_training, vocab_size, patience, max_length=1000):
        the_stopwords = stopwords.words("english")
        start = time.time()
        num_cores = 6
        num_CPU = 1
        num_GPU = 1
        config = tf.ConfigProto(intra_op_parallelism_threads=num_cores, \
                                inter_op_parallelism_threads=num_cores, allow_soft_placement=False, \
                                device_count={'CPU': num_CPU, 'GPU': num_GPU})
        session = tf.Session(config=config)
        backend.set_session(session)

        raw_data = pd.read_csv("news_ds.csv")
        training_data = raw_data.iloc[0:num_training]
        t = text.Tokenizer(vocab_size)

        t.fit_on_texts(training_data["TEXT"])


        matrix = np.ndarray(shape=(vocab_size,300))
        tokens = []
        ref = 0
        thingy = sorted([x for x in t.word_counts if x[0] not in the_stopwords],key=t.word_counts.get, reverse=True)

        for word in itertools.islice(thingy,vocab_size):
            matrix[ref] = np.array([nlp(word)[0].vector])
            ref += 1



        encoded_docs = t.texts_to_sequences(training_data["TEXT"])
        padded_docs = sequence.pad_sequences(encoded_docs, maxlen=max_length, padding='post')

        e = Embedding(vocab_size, 300, weights=np.array([matrix]), input_length=max_length, trainable=False)
        model = Sequential()
        model.add(e)
        callback1 = callbacks.EarlyStopping(monitor='acc',
                                      min_delta=0,
                                      patience=patience,
                                      verbose=0, mode='max')
        callback2 = callbacks.ModelCheckpoint("superthing.txt")

        model.add(LSTM(neurons))
        model.add(Dropout(dropout))
        model.add(Dense(1, activation=activation_type))


        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

        model.fit(padded_docs, training_data["LABEL"], epochs=epochs, verbose=0, callbacks = [callback1,callback2])

        testing_data = raw_data.iloc[num_training + 1:]

        t.fit_on_texts(testing_data["TEXT"])

        encoded_docs2 = t.texts_to_sequences(testing_data["TEXT"])
        padded_testing_docs = sequence.pad_sequences(encoded_docs2, maxlen=max_length, padding='post')

        loss, accuracy = model.evaluate(padded_testing_docs, testing_data["LABEL"], verbose=0)
        print(
            "epochs: %d, neurons: %d, dropout: %f, activation: %s, num_training: %d, vocab_size: %d, patience: %d,  max_length: %d" % (
            epochs, neurons, dropout, activation_type, num_training, vocab_size, patience, max_length))

        print('Accuracy: %f' % (accuracy * 100))
        end = time.time()
        print("Total time = %d" % (end - start))




def sentence_to_word2vec(sentence):
    doc = nlp(sentence)
    returningSentence = []
    for token in doc:
        returningSentence.append(token.vector)
    return np.array(returningSentence)


if __name__ == "__main__":
    thing = DeepIdentifier()
    thing.do_RNN_stuff(epochs=50, neurons=120, dropout=0.5, num_training=1000, activation_type='tanh', vocab_size=10000,
             patience=6)
    thing.do_LSTM_stuff(epochs=50, neurons=120, dropout=0.5, num_training=1000, activation_type='tanh', vocab_size=10000,
             patience=6)