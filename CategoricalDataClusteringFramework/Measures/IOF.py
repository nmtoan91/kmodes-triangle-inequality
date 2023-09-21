from Measure import Measure
from MathUtils import *
import math
from collections import Counter
import timeit
class IOF(Measure):
    def __init__(self):
        self.name = "IOF"
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
            self.computeFrequencyMatrix(X)
            self.GeneratedistMatrix()
            self.SavedistMatrix()
        return  timeit.default_timer() - start
    def computeFrequencyMatrix(self,X):
        self.probabilities2 = [];
        n = len(X)
        d = len(X[0])
        self.max = []
        self.frequency= []
        for i in range(d):
            sdb = X[:,i];
            self.max.append(np.max(np.unique(sdb)))
            frequency = Counter(sdb)
            frequency=frequency.values();
            frequency = np.array(list(frequency))#/selWf.N
            self.frequency.append(frequency)
    def CalcdistanceArrayForDimension(self,d,i,j):
        if i!=j:
            if self.frequency[d][i] and self.frequency[d][j] != 0:
                similar =  1 / (1 + math.log10(self.frequency[d][i]) * math.log10(self.frequency[d][j]))
                return 1 - similar
        return 0

