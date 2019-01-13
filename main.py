import ShallowIdentification
import DeepLearning

if __name__ == "__main__":
    shallow = ShallowIdentification.ShallowIdentifier()
    deep = DeepLearning.DeepIdentifier()
    shallow.do_tf_stuff(1, 1)
    shallow.do_tfidf_stuff(1, 1)

    deep.do_RNN_stuff(epochs=50, neurons=120, dropout=0.5, num_training=1000, activation_type='tanh', vocab_size=10000,
                       patience=6)
    deep.do_LSTM_stuff(epochs=50, neurons=120, dropout=0.5, num_training=1000, activation_type='tanh',
                        vocab_size=10000,
                        patience=6)
