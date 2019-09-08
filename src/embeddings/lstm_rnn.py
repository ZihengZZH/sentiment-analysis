import os
import json
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.utils import plot_model
from keras.utils import np_utils


class LSTMRNN(object):
    def __init__(self, docs_train, docs_test, dataset):
        self.config = json.load(open('./config.json', 'r'))['lstm_rnn']
        self.docs_train = docs_train
        self.docs_test = docs_test
        self.hparams = self.config['hparams_set']
        self._tokenize()
    
    def _tokenize(self):
        token = Tokenizer(num_words=self.hparams['max_length'], oov_token='<OOV>')
        token.fit_on_texts(self.docs_train)
        self.X_train = pad_sequences(token.texts_to_sequences(self.docs_train))
        self.X_test = pad_sequences(token.texts_to_sequences(self.docs_test))
        # self.X_train = token.texts_to_matrix(self.docs_train, mode='count')
        # self.X_test = token.texts_to_matrix(self.docs_train, mode='count')
        # summarize what was learned
        # print(token.word_counts)
        # print(token.document_count)
        # print(token.word_index)
        # print(token.word_docs)
    
    def build_model(self):
        self.model = Sequential()
        self.model.add(Embedding(self.hparams['max_length'],
                        self.hparams['embed_dim'],
                        input_length=self.X_train.shape[1]))
        self.model.add(SpatialDropout1D(0.1))
        self.model.add(LSTM(self.hparams['out_dim'], 
                            dropout=0.2,
                            recurrent_dropout=0.2))
        self.model.add(Dense(2, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy',
                            optimizer='rmsprop',
                            metrics=['accuracy'])
        print(self.model.summary())
        plot_model(self.model, 
                    to_file='./images/embeddings_lstm.png', 
                    show_layer_names=True)
    
    def train_model(self, y_train):
        y_train = np_utils.to_categorical(y_train)
        self.model.fit(self.X_train, y_train, 
                        epochs=self.hparams['epochs'], 
                        batch_size=self.hparams['batch_size'], 
                        verbose=self.hparams['verbose'])
    
    def infer_embeddings(self):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass
