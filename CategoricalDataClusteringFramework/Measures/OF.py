from Measure import Measure
from MathUtils import *
import timeit

class OF(Measure):
    def __init__(self):
        self.name = "OF"
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
            D = len(X[0])
            self.distMatrix = [];
            for d in range(D):
                matrix2D = [] # 2D array for 1 dimension
                for i in range(self.max[d]+1):
                    matrix1D = [] # 1D array for 1 dimension
                    for j in range(self.max[d]+1): 
                        matrix_tmp = self.CalcdistanceArrayForDimension(d,i,j)
                        matrix1D.append(matrix_tmp)
                    matrix2D.append(matrix1D)
                self.distMatrix.append(matrix2D)
            self.SavedistMatrix()
        return timeit.default_timer() - start
    def CalcdistanceArrayForDimension(self,k,xi,xj):
        similar = 0
        if xi == xj:
            similar = 1
        else:
            fki = freq(self.X_[:, k], xi)
            fkj = freq(self.X_[:, k], xj)
            N = len(self.X_)

            if fki != 0 and fkj != 0:
                similar = 1 / (1 + (N / fki).log10() * (N / fkj).log10())
        return float(1-similar)

