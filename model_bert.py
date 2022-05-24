import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from tqdm.keras import TqdmCallback
import numpy as np
import time
import psutil
from utils import *

''' Parent class for all the CNN models '''
class BERTModel:
    def __init__(self, X_train, y_train, X_test, y_test, num_class):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.y_predict = None
        self.y_prob = None
        self.num_class = num_class
        self.model = None
        gpu_management() # gpu set up for tensorflow
        self.bert_preprocess = None
        self.bert_encoder = None
        self.text_input = None
        self.outputs = None
        self.download_bert()
        self.bert_model()

    def runner(self):
        ''' Total runner function '''
        print("Model:\t " + self.name)
        print("")

        start_time = time.time()
        self.train()
        print("Train:\t %s seconds" % round((time.time() - start_time), 4))
        print("Train CPU(%):\t", psutil.cpu_percent())
        print("Train RAM(GB):\t", round(psutil.virtual_memory()[3] * 1e-9, 2))  # physical memory usage
        print('Train RAM(%):\t', psutil.virtual_memory()[2])
        print("")

        name_str = self.name.replace(' ', '_').lower()
        file_path = f"./model/{name_str}/1/"
        self.model.save(file_path, include_optimizer=False)

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

    def download_bert(self):
        self.bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
        self.bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

    def bert_model(self):
        self.text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        preprocessed_text = self.bert_preprocess(self.text_input)
        self.outputs = self.bert_encoder(preprocessed_text)

    def train(self):
        self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), epochs=50, batch_size=128, 
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
Below is the child class under BERT transformer pre-trained models 
1. Binary classifier model
2. Multiclass classifier model
'''
class BERTBinary(BERTModel):
    def __init__(self, X_train, y_train, X_test, y_test, num_class):
        BERTModel.__init__(self, X_train, y_train, X_test, y_test, num_class)
        self.model = self.build_cnn()
        self.name = 'Binary BERT'

    def build_cnn(self) -> tf.keras.Model:
        net = tf.keras.layers.Conv1D(32, (2), activation='relu')(self.outputs["sequence_output"])
        net = tf.keras.layers.GlobalMaxPool1D()(net)
        net = tf.keras.layers.Flatten()(net)
        net = tf.keras.layers.Dense(250, activation="relu")(net)
        net = tf.keras.layers.Dropout(0.1)(net)
        net = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(net)
        model = tf.keras.Model(inputs=[self.text_input], outputs = [net])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model

    def predict(self):
        ''' 
        Convert predicted probability into the class prediction
        Simply compare whether the probability is above or below the threshold (default=0.5)
        '''
        self.y_predict = np.where(self.y_prob > 0.5, 1,0)

class BERTMulti(BERTModel):
    def __init__(self, X_train, y_train, X_test, y_test, num_class):
        BERTModel.__init__(self, X_train, y_train, X_test, y_test, num_class)
        self.model = self.build_cnn()
        self.name = 'Multi BERT'
        self.y_train = to_categorical(y_train, self.num_class)
        self.y_test = to_categorical(y_test, self.num_class)

    def build_cnn(self) -> tf.keras.Model:
        net = tf.keras.layers.Conv1D(32, (2), activation='relu')(self.outputs["sequence_output"])
        net = tf.keras.layers.GlobalMaxPool1D()(net)
        net = tf.keras.layers.Flatten()(net)
        net = tf.keras.layers.Dense(250, activation="relu")(net)
        net = tf.keras.layers.Dropout(0.1)(net)
        net = tf.keras.layers.Dense(self.num_class, activation="softmax", name='classifier')(net)
        model = tf.keras.Model(inputs=[self.text_input], outputs = [net])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model

    def predict(self):
        ''' 
        Convert predicted probability into the class prediction
        It performs argmax since it was a multiclass prediction
        '''
        self.y_predict = np.argmax(self.y_prob, axis=1)