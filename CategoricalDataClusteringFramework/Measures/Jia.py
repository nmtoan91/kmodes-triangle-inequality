from Measure import Measure
from MathUtils import *
import timeit

class Jia(Measure):
    def __init__(self):
        self.name = "Jia"
    # calculate distance between 2 instances following DM3 algorithm
    def calculate(self, instance1, instance2):
        distance = 0
        length = len(instance1)
        self.xi = instance1
        self.xj = instance2
        self.W = self.weigth(instance1, instance2)
        for x in range(length):
            distance += self.W[x] * self.dist3(instance1[x], instance2[x], x)
        return distance / self.W.sum()

    def setUp(self, X, y):
        start = timeit.default_timer()
        self.X_ = X
        self.A = generateAllDomain(self.X_)
        self.Sr = self.generateRelativeSet(beta=0.1)
        self.freq_table = countFreq(self.X_)
        self.N = len(self.X_[:, 0])
        return timeit.default_timer() - start
    # calculate the similarity between 2 instances xi, xj at attribute k in DM3 algorithm
    def dist3(self, xi, xj, r):
        d = 0
        w = 0
        if xi != xj:
            for i in range(len(self.Sr[r])):
                l = self.Sr[r][i]
                R_rl = self.RMatrix[r][l]
                p_ir_il = self.condEqualProb(self.X_[:, r], self.X_[:, l], xi, self.xi[l])
                p_jr_jl = self.condEqualProb(self.X_[:, r], self.X_[:, l], xj, self.xj[l])
                d += R_rl * (p_ir_il + p_jr_jl)
        return d

    # calculate the similarity between two instance xi and xj at attribute k in DM1 algorithm
    def dist1(self, xi, xj, k):
        if xi == xj:
            d = 0
        else:
            d = self.origin(xi, xj, k)
        return d

    # calculate the origin distance
    def origin(self, xi, xj, k):
        pxi_xi = self.equalProb(k, xi)
        pxj_xj = self.equalProb(k, xj)
        org = pxi_xi + pxj_xj
        return org

    # calculate the probability of two values of an attribute is the same
    def equalProb(self, Ak, x):
        N = len(Ak)
        fx = Decimal(freq(Ak, x))
        px_x = (fx / N) * ((fx - 1) / (N - 1))
        return px_x

    # calculate joint probability of 2 couples of 2 values appear the same
    def condEqualProb(self, Ar, Al, xr, xl):
        N = len(Ar)
        frl = Decimal(condFreq(Ar, Al, xr, xl))
        prl_rl = (frl / N) * ((frl - 1) / (N - 1))
        return prl_rl

    # calculate weigth for attribute Ak in DM2
    def getWeigth2(self, k):
        total = self.W.sum()
        w = self.W[k] / total
        return w

    # calculate weigth for all attributes
    def weigth(self, instance1, instance2):
        W = []
        for i in range(len(instance1)):
            ps = 0
            pf = 0
            for j in range(len(self.A[i])):
                ps += self.equalProb(self.X_[:, i], self.A[i][j])
            if instance1[i] == instance2[i]:
                pf = 1 - ps
                w_i = pf
            else:
                w_i = ps
            W.append(w_i)
        return np.array(W)

    # create relative matrix based on Jia
    def generateRelativeSet(self, beta):
        Sk = []
        self.RMatrix = interRedundMatrix(self.X_)
        if beta != 0:
            beta = self.RMatrix.mean()
        length = len(self.RMatrix[:, 1])
        for i in range(length):
            temp = []
            for j in range(length):
                if j != i:
                    if self.RMatrix[i][j] > beta:
                        temp.append(j)
            Sk.append(temp)
        return np.array(Sk)
