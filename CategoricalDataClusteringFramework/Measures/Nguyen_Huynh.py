from Measure import Measure
from MathUtils import *
import timeit

class Nguyen_Huynh(Measure):
    def __init__(self):
        self.name = "Nguyen_Huynh"
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
            self.freq_table = countFreq(self.X_)
            self.N = len(self.X_[:, 0])
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
        return timeit.default_timer()- start
    # calculate the similarity between two instance xi and xj at attribute k
    def CalcdistanceArrayForDimension(self,k,xi,xj):
        if xi == xj:
            similar = 1
        else:
            Pxi = prob(self.X_[:, k], xi)
            Pxj = prob(self.X_[:, k], xj)
            if Pxi != 0 and Pxj != 0:
                similar = 2 * (Pxi + Pxj).log10() / (Pxi.log10() + Pxj.log10())
            else:
                similar = 0
        return float(1-similar)

