from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
import psutil
from utils import *
import time

''' Parent class for all the sklearn models '''
class SklearnModel:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = None
        self.y_predict = None
        self.y_prob = None

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

        start_time = time.time()
        self.predict()
        print("Predict: %s seconds" % round((time.time() - start_time), 4))
        print("Predict CPU(%):\t", psutil.cpu_percent())
        print("Predict RAM(GB):", round(psutil.virtual_memory()[3] * 1e-9, 2))  # physical memory usage
        print('Predict RAM(%):\t', psutil.virtual_memory()[2])
        print("")

        self.predict_prob()
        self.print_report()
        self.print_confusion()

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        self.y_predict = self.model.predict(self.X_test)

    def predict_prob(self):
        self.y_prob = self.model.predict_proba(self.X_test)[:,1]

    def print_report(self):
        print(classification_report(self.y_test, self.y_predict))

    def print_confusion(self):
        label = np.sort(self.y_test.unique()).astype(int).astype(str)
        draw_matrix(self.y_test, self.y_predict, self.name + 'Confusion Matrix' , label).show()

''' 
Below is the child class under sklearn models 
1. Logistic Regression
2. Naive Bayes model
3. Support Vector Machine (SVM)
4. Gradient Boosting
'''
class Logistic(SklearnModel):
    def __init__(self, X_train, y_train, X_test, y_test):
        SklearnModel.__init__(self, X_train, y_train, X_test, y_test)
        self.model = LogisticRegression(solver='saga', penalty='l2')
        self.name = 'Logistic Regression'

class NaiveBayes(SklearnModel):
    def __init__(self, X_train, y_train, X_test, y_test):
        SklearnModel.__init__(self, X_train, y_train, X_test, y_test)
        self.model = MultinomialNB()
        self.name = 'Naive Bayes'

class SVM(SklearnModel):
    def __init__(self, X_train, y_train, X_test, y_test):
        SklearnModel.__init__(self, X_train, y_train, X_test, y_test)
        self.model = svm.SVC(probability=True)
        self.name = 'SVM'

class GradientBoosting(SklearnModel):
    def __init__(self, X_train, y_train, X_test, y_test):
        SklearnModel.__init__(self, X_train, y_train, X_test, y_test)
        self.model = GradientBoostingClassifier()
        self.name = 'Gradient Boosting'
