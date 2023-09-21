from Measure import Measure
from MathUtils import *
import timeit

class New_Lin(Measure):
    def __init__(self):
        self.name = "New_Lin"
    def calculate(self, instance1, instance2):
        distance = 0
        length = len(instance1)
        for x in range(length):
            distance =distance+self.distMatrix[x][instance1[x]][instance2[x]]
        return distance/length

    def setUp(self, X, y):
        start= timeit.default_timer()
        self.X_ = X
        self.y_ = y
        if self.LoaddistMatrixAuto()==False:
            print("Generating disMatrix for New_Lin")
            self.Sk = np.array(self.generateRelativeSet(beta=0.1))
            self.A = generateAllDomain(self.X_)
            self.GeneratedistMatrix()
            self.SavedistMatrix()
        return timeit.default_timer() - start
    
    def getRelativeAttributes(self, k):
        return self.Sk[k]

    # calculate the similarity between two instance xi and xj at attribute k
    def CalcdistanceArrayForDimension(self,k,xi,xj):
        similar = 0
        Sk = np.array(self.getRelativeAttributes(k))
        # Need to add the case that attributes has no highly independence attributes
        if Sk.size == 0:
            if xi == xj:
                similar = 1
            else:
                Pxi = prob(self.X_[:, k], xi)
                Pxj = prob(self.X_[:, k], xj)
                if Pxi != 0 and Pxj != 0:
                    similar = (2 * (Pxi + Pxj).log10()) / (Pxi.log10() + Pxj.log10())
        else:
            card_Sk = len(Sk)
            card_Al = 0
            if xi == xj:
                similar = 1
            else:
                for p in range(len(Sk)):
                    for q in range(len(self.A[Sk[p]])):
                        Pxi_a = condProb(self.X_[:, k], self.X_[:, p], xi, self.A[Sk[p]][q])
                        Pxj_a = condProb(self.X_[:, k], self.X_[:, p], xj, self.A[Sk[p]][q])
                        card_Al = len(self.A[Sk[p]])
                        if Pxi_a != 0 and Pxj_a != 0:
                            similar += (Decimal(1) / card_Sk) * (Decimal(1) / card_Al) * (
                                2 * (Pxi_a + Pxj_a).log10() / (Pxi_a.log10() + Pxj_a.log10()))
        return float(1 - similar)
    # create relative matrix based on Jia
    def generateRelativeSet(self, beta):
        Sk = []
        RMatrix = interRedundMatrix(self.X_)
        length = len(RMatrix[:, 1])
        for i in range(length):
            temp = []
            for j in range(length):
                if j != i:
                    if RMatrix[i][j] > beta:
                        temp.append(j)
            Sk.append(temp)
        return np.array(Sk,dtype=object)
