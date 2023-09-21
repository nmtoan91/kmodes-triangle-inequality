from .Measure import *
import timeit

class Overlap(Measure):
    def __init__(self,datasetName='None'):
        self.name = "Overlap"
        MeasureManager.CURRENT_DATASET = datasetName
    def calculate(self, instance1, instance2):
        distance = 0
        length = len(instance1)
        for x in range(length):
            distance =distance+self.distMatrix[x][instance1[x]][instance2[x]]
        return distance/length
    def setUp(self, X, y):
        start = timeit.default_timer()
        self.X_ = X
        self.N = len(self.X_[:, 0])
        self.max = []
        for i in range(len(X[0])):
            self.max.append(max(X[:,i]))
        D = len(X[0])
        self.distMatrix = [];
        if self.LoaddistMatrixAuto()==False:
            self.GeneratedistMatrix()
            self.SavedistMatrix()
        return timeit.default_timer()- start
    def CalcdistanceArrayForDimension(self,k,xi,xj):
        if xi == xj:
            return 0
        else:
            return 1
