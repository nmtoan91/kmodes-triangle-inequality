import os
import os.path
import sys
from sys import platform
sys.path.append(os.path.join(os.getcwd(), "Measures"))
import operator
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from Measures import *
import pandas as pd
import TUlti as tulti
from sklearn.model_selection import ShuffleSplit, cross_val_score
from datetime import date

class kNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, k=5):
        self.__name__ = 'kNNClassifier'
        self.k = k

    def initMetric2(self, module_name, class_name):
        self.module_name = module_name
        self.class_name = class_name
        module = __import__(self.module_name, globals(), locals(), ['object'])
        class_ = getattr(module, self.class_name)
        return class_()

    def fit(self, X, y, **kwargs):
        self.X_ = X
        self.y_ = y
        self.k = kwargs['k']
        self.measure = self.initMetric2(kwargs['module_name'], kwargs['class_name'])
        return self

    def predict2(self, X, y=None):
        result = []
        #self.measure.setUp(self.X_, self.y_) 
        self.measure.setUp(np.concatenate((self.X_,X)) , self.y_) 
        

        for x in X:
            neighbors = self.getNeighbors2(x)
            result.append(self.getResponse2(neighbors))
        return np.array(result)

    def getNeighbors2(self, X):
        distances = []
        length = len(X)

        for i in range(len(self.X_)):
            dist = self.measure.calculate(X, self.X_[i])
            distances.append((self.y_[i], dist))
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        if len(self.X_[:, 0]) < self.k:
            self.k = len(self.X_)
        for i in range(self.k):
            neighbors.append(distances[i][0])
        return neighbors

    def getResponse2(self, neighbors):
        classVotes = {}
        for i in range(len(neighbors)):
            response = neighbors[i]
            if response in classVotes:
                classVotes[response] += 1
            else:
                classVotes[response] = 1
        sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
        return sortedVotes[0][0]

    def score(self, X, y=None):
        #for i in range(len(X[0])):
            #if max(X[:,i]) != max(self.X_[:,i]):
                #return 0;
            
        score = 0
        predict = self.predict2(X)
        for i in range(len(predict)):
            if predict[i] == y[i]:
                score += 1
        return (float(score) / len(predict)) * 100

    def get_params2(self, deep=True):
        return {'k': self.k}

df_list = {}
cols = MeasureManager.MEASURE_LIST


def main():
    for line in MeasureManager.DATASET_LIST:
        MeasureManager.CURRENT_DATASET = line
        filepath = 'F:\\DATASET\\ANN_CATEGORICAL\\' + line 
        if platform == "linux" or platform == "linux2":
            filepath = '/home/s1620409/DATASET/ANN_CATEGORICAL/' + line

        db = pd.read_csv(filepath).replace(np.nan, 'nan', regex=True)._values;
        DB = tulti.NormalizeDB(db)
        data = DB['DB']
        target = DB['labels_']
        n_jobs =1;
        if platform == "linux" or platform == "linux2":
            n_jobs=8
        for class_name in MeasureManager.MEASURE_LIST:
            MeasureManager.CURRENT_MEASURE = class_name
            module_name = class_name
            clf = kNNClassifier()
            cv = ShuffleSplit(n_splits=10)
            score = cross_val_score(clf, X=data, y=target, cv=cv,
                                    fit_params={'module_name': module_name, 'class_name': class_name, 'k': 7},
                                    verbose=10,n_jobs=n_jobs)
            print(score)
            mean = 0
            for i in score:
                mean += i
                value = repr(float(mean) / len(score))
            print('Mean Accuracy of measure ' + class_name + ' for data set ' + line + ' : ' + value)
            if class_name in df_list:
                df_list[class_name].append(value)
            else:
                df_list[class_name] = [value];

        df = pd.DataFrame(df_list, columns = MeasureManager.MEASURE_LIST)
        name_rules = {i: MeasureManager.DATASET_LIST[i] for i in range(len(MeasureManager.DATASET_LIST)) }
        df = df.rename(index=name_rules)
        print(df)
        df.to_csv('kNNClassifier_result_' + str(date.today())+ '.csv')
if __name__ == "__main__":
    main()