from Measure import Measure
from MathUtils import *
import timeit

class Eskin(Measure):
    def __init__(self):
        self.name = "ESkin"
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
            self.A = generateAllDomain(self.X_)
            self.GeneratedistMatrix()
            self.SavedistMatrix()
        return   timeit.default_timer()-start
    def CalcdistanceArrayForDimension(self,d,i,j):
        if i!=j:
            nk = len(self.A[d])
            return 2 / math.pow(nk, 2)
        return 0
    # calculate the similarity between two instance xi and xj at attribute k
    def dist(self, xi, xj, k):
        if xi == xj:
            d = 0
        else:
            nk = len(self.A[k])
            d = 2 / math.pow(nk, 2)
        return d
