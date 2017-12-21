import pickle
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from numpy import *
class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier
        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.weak_classifier = weak_classifier
        self.n_weakers_limit = n_weakers_limit
        self.weak_classifier_list = []
        self.alpha_list = []

    def is_good_enough(self):
        '''Optional'''
        pass

    def fit(self,X,y):
        '''Build a boosted classifier from the training set (X, y).

        Returns:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        m = X.shape[0];
        weight = np.ones(m)/m
        for i in range(self.n_weakers_limit):
            model = self.weak_classifier(max_depth=1).fit(X,y,sample_weight=weight)
            self.weak_classifier_list.append(model)
            error = 0.
            for j in range(m):
                if model.predict(X[j].reshape(1,-1)) != y[j]:
                    error += weight[j]
            if error > 0.5:
                break
            alpha = 0.5 * np.log((1-error)/error)
            self.alpha_list.append(alpha)
            expon = 0
            for j in range(m):
                expon += weight[j] * (np.exp(-alpha * y[j] * model.predict(X[j].reshape(1,-1))))
            for j in range(m):
                weight[j] = weight[j] * (np.exp(-alpha * y[j] * model.predict(X[j].reshape(1,-1)))) / expon
        self.save(self.alpha_list , "./alpha_list.pkl")
        self.save(self.weak_classifier_list,"./weak_classifier_list.pkl")
        print("train success")
    

    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        pass

    def predict(self, X, y,threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''

        weak_classifier_list = self.load("./weak_classifier_list.pkl")
        alpha_list = self.load('./alpha_list.pkl')
        class_y = 0
        m = X.shape[0]
        y_pred = array(zeros(m))
        error_list = []
        for i in range(len(weak_classifier_list)):
            y_pred += alpha_list[i] * weak_classifier_list[i].predict(X)
            class_y = sign(y_pred)
            class_y_errors = (class_y != y)
            loss_rate = class_y_errors.sum()/m
            print(i," the total error rate in predicting the validation set is ",loss_rate)
            error_list.append(loss_rate)
            if loss_rate <= 0.005:break
        iter_list = []
        for i in range(len(error_list)):
            iter_list.append(i)
        plt.figure()
        plt.plot(iter_list,error_list,linestyle = '-',color = 'black',linewidth = 2.0, label = 'test')
        plt.xlabel('iter_times')  
        plt.ylabel('error_rate')
        plt.legend()
        plt.show()
        return class_y

    @staticmethod
    def save(model, filename):
        with open(filename, "wb+") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
