from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from tqdm.keras import TqdmCallback
import numpy as np
import pickle
import time
import psutil
from utils import *

''' Parent class for all the CNN models '''
class CNNModel:
    def __init__(self, X_train, y_train, X_test, y_test, max_words, num_class):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.y_predict = None
        self.y_prob = None
        self.max_words = max_words
        self.num_class = num_class
        self.model = None
        gpu_management() # gpu set up for tensorflow

    def runner(self):
        ''' Total runner function '''
        print("Model:\t " + self.name)
        print("")
        self.tokenize()
        self.padding()

        start_time = time.time()
        self.train()
        print("Train:\t %s seconds" % round((time.time() - start_time), 4))
        print("Train CPU(%):\t", psutil.cpu_percent())
        print("Train RAM(GB):\t", round(psutil.virtual_memory()[3] * 1e-9, 2))  # physical memory usage
        print('Train RAM(%):\t', psutil.virtual_memory()[2])
        print("")

        name_str = self.name.replace(' ', '_').lower()
        file_path = f"./model/{name_str}/1/"
        self.model.save(filepath=file_path, save_format='tf')
        print("\nModel saved at: ./model/1/" + name_str + '\n')

        start_time = time.time()
        self.predict_prob()
        print("Predict: %s seconds" % round((time.time() - start_time), 4))
        print("Predict CPU(%):\t", psutil.cpu_percent())
        print("Predict RAM(GB):", round(psutil.virtual_memory()[3] * 1e-9, 2))  # physical memory usage
        print('Predict RAM(%):\t', psutil.virtual_memory()[2])
        print("")

        self.predict()
        self.print_report()
        self.print_confusion()

    def tokenize(self):
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(self.X_train)
        self.X_train = tokenizer.texts_to_sequences(self.X_train)
        self.X_test = tokenizer.texts_to_sequences(self.X_test)

        name_str = self.name.replace(' ', '_').lower()
        with open('./model/tokenizer_' + name_str + '.pickle', 'wb') as handle: # save tokenizer
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def padding(self):
        self.X_train = sequence.pad_sequences(self.X_train, maxlen=self.max_words)
        self.X_test = sequence.pad_sequences(self.X_test, maxlen=self.max_words)

    def train(self):
        self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), epochs=50, batch_size=32, 
                        verbose=0, callbacks=[TqdmCallback(verbose=1)])

    def predict_prob(self):
        self.y_prob = self.model.predict(self.X_test)
    
    def print_report(self):
        scores = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))

        if self.y_test.ndim != 1: # shape revert required for multi-class model
            print(classification_report(np.argmax(self.y_test, axis=1, out=None), self.y_predict))
        else: 
            print(classification_report(self.y_test, self.y_predict))

    def print_confusion(self):
        if self.y_test.ndim != 1: # shape revert required for multi-class model
            y_test = np.argmax(self.y_test, axis=1, out=None)
        else:
            y_test = self.y_test

        label = np.sort(np.unique(y_test)).astype(int).astype(str)
        draw_matrix(y_test, self.y_predict.tolist(), 'Confusion Matrix' , label).show()

''' 
Below is the child class under CNN models 
1. Binary classifier model
2. Multiclass classifier model
'''
class CNNBinary(CNNModel):
    def __init__(self, X_train, y_train, X_test, y_test, max_words, num_class):
        CNNModel.__init__(self, X_train, y_train, X_test, y_test, max_words, num_class)
        self.model = self.build_cnn()
        self.name = 'Binary CNN'

    def build_cnn(self) -> Sequential:
        model = Sequential()
        model.add(Embedding(5000, 32, input_length=self.max_words))
        model.add(Conv1D(32, 3, padding='same', activation='relu'))
        model.add(MaxPooling1D())
        model.add(Flatten())
        model.add(Dense(250, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(1, activation='sigmoid')) # binary
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model

    def predict(self):
        ''' 
        Convert predicted probability into the class prediction
        Simply compare whether the probability is above or below the threshold (default=0.5)
        '''
        self.y_predict = np.where(self.y_prob > 0.5, 1,0)

class CNNMulti(CNNModel):
    def __init__(self, X_train, y_train, X_test, y_test, max_words, num_class):
        CNNModel.__init__(self, X_train, y_train, X_test, y_test, max_words, num_class)
        self.model = self.build_cnn()
        self.name = 'Multi CNN'
        self.y_train = to_categorical(y_train, self.num_class)
        self.y_test = to_categorical(y_test, self.num_class)

    def build_cnn(self) -> Sequential:
        model = Sequential()
        model.add(Embedding(5000, 32, input_length=self.max_words))
        model.add(Conv1D(32, 3, padding='same', activation='relu'))
        model.add(MaxPooling1D())
        model.add(Flatten())
        model.add(Dense(250, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(self.num_class, activation='softmax')) # multi
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model

    def predict(self):
        ''' 
        Convert predicted probability into the class prediction
        It performs argmax since it was a multiclass prediction
        '''
        self.y_predict = np.argmax(self.y_prob, axis=1)