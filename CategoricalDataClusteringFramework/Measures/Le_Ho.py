from Measure import Measure
from MathUtils import *
import numpy as np
import timeit
from decimal import *
class Le_Ho(Measure):
    def __init__(self):
        self.name = "Le_Ho"
    def calculate(self, instance1, instance2):
        distance = 0
        length = len(instance1)
        for x in range(length):
            distance =distance+self.distMatrix[x][instance1[x]][instance2[x]]
        return distance/length
    def ComputeconditionProMatrix(self,X):
        self.conditionProMatrix ={}
        D = len(X[0])
        for d in range(D):
            Y_ = X[:,d]
            Y_unique, Y_freq = np.unique(Y_, return_counts=True)
            for j in range(D):
                if d !=j:
                    X_ = X[:,j]
                    X_unique, X_freq = np.unique(X_, return_counts=True)
                    conditionMatrix = ProconditionMatrixYX(Y_,X_,Y_unique, X_unique)
                    self.conditionProMatrix[(d,j)] = conditionMatrix
    def setUp(self, X, y):
        start = timeit.default_timer()
        self.X_ = X
        #Compute conditional probatilityes
        if self.LoaddistMatrixAuto()==False:
            self.A = generateAllDomain(self.X_)
            self.max = []
            for i in range(len(X[0])):
                self.max.append(max(X[:,i]))
            D = len(X[0])
            self.distMatrix = [];
            self.ComputeconditionProMatrix(X)
            self.GeneratedistMatrix()
            self.SavedistMatrix()
        return timeit.default_timer() - start
    def CalcdistanceArrayForDimension(self,k,xi,xj):
        d = 0
        length = len(self.X_[0, :])

        if xi != xj:
            for n in range(length):
                if n != k:
                    domAn = self.A[n]
                    conditionMatrix = self.conditionProMatrix[(n,k)]
                    for m in domAn:
                        # calculate conditional probability of m given xi
                        Pni = Decimal(conditionMatrix[m][xi])
                        #Pni = condProb(self.X_[:, n], self.X_[:, k], m, xi)
                        #Pni = Decimal(Pni_t)
                        # calculate conditional probability of m given xj
                        Pnj = Decimal(conditionMatrix[m][xj])
                        #Pnj = condProb(self.X_[:, n], self.X_[:, k], m, xj)
                        if Pni != 0 and Pnj != 0:
                            d += Pni * (Pni / Pnj).log10()/Decimal(2).log10()
                            + Pnj * (Pnj / Pni).log10()/Decimal(2).log10()
        return float(d)
