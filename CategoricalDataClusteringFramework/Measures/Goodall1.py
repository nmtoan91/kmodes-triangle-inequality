from Measure import Measure
import math
from MathUtils import *
from scipy.stats import entropy
import numpy as np
from collections import Counter
import timeit

class Goodall1(Measure):
    def __init__(self):
        self.name = "Goodall1"
    def calculate(self, instance1, instance2):
        distance = 0
        length = len(instance1)
        for x in range(length):
            distance =distance+self.distMatrix[x][instance1[x]][instance2[x]]
        return distance/length

    def setUp(self, X, y):
        start = timeit.default_timer()
        self.X_ = X
        if self.LoaddistMatrixAuto()==False:
            self.max = []
            for i in range(len(X[0])):
                self.max.append(max(X[:,i]))
            self.computeProbabilities2Matrix(X)
            self.GeneratedistMatrix()
            self.SavedistMatrix()
        return timeit.default_timer() - start
    def CalcdistanceArrayForDimension(self,d,i,j):
        #matrix_tmp = []; 
        if i!=j:
            dist_sum_all=0;
            for j in range(self.max[d]+1):
                if self.probabilities2[d][j] >= self.probabilities2[d][i]:
                    dist_sum_all = dist_sum_all+ self.probabilities2[d][j];
            return dist_sum_all
        return 0
    def computeProbabilities2Matrix(self,X):
        self.probabilities2 = [];
        n = len(X)
        d = len(X[0])
        #frequency2 =[]
        self.max = []
        for i in range(d):
            sdb = X[:,i];
            self.max.append(np.max(np.unique(sdb)))
            frequency = Counter(sdb)
            frequency=frequency.values();
            frequency = np.array(list(frequency))#/selWf.N
            tmp = [i*(i-1)/(n*(n-1)) for i in frequency ]
            self.probabilities2.append(tmp)