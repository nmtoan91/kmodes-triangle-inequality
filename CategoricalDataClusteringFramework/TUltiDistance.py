import numpy as np
from collections import Counter
import math 
class TSimilarity:
    def __init__(self, DB):
        
        self.frequency = [];
        self.probabilities = [];
        self.probabilities2 = [];
        self.N = DB['n'] 
        self.d = DB['d'] 
        self.db = DB['DB']
        self.max = DB['max']
        self.nk = [i +1 for i in self.max]
       
        for i in range(self.d):
            sdb = DB['DB'][:,i];
            n = DB['max'][i]
            frequency = Counter(sdb)
            frequency=frequency.values();
            frequency = np.array(list(frequency))#/self.N
            self.frequency.append(frequency)
            self.probabilities.append(frequency/self.N)
            tmp = [i*(i-1)/(self.N*(self.N-1)) for i in frequency ]
            self.probabilities2.append(tmp)
    #def Test_Half(v):

    def Dis_Overlap(self, x, y):
        sim=0;
        for i in range(len(x)):
            if x[i]==y[i]:
                sim = sim+1
            else:
                sim=sim+0;
        return sim/self.d
    def Dis_Eskin(self,x,y):
        sim=0
        for i in range(len(x)):
            if x[i]==y[i]:
                sim = sim+1
            else:
                sim= sim + (self.nk[i]**2)/ (self.nk[i]**2+2)
        return sim/self.d
    def Dis_IOF(self,x,y):
        sim=0
        for i in range(len(x)):
            if x[i]==y[i]:
                sim = sim+1
            else:
                sim=sim+ 1/(1+ math.log(self.frequency[i][x[i]])*math.log(self.frequency[i][y[i]]))
        return sim/self.d
    def Dis_OF(self,x,y):
        sim=0
        for i in range(len(x)):
            if x[i]==y[i]:
                sim = sim+1
            else:
                sim=sim+ 1/(1+ math.log(self.N/self.frequency[i][x[i]])*math.log(self.N/self.frequency[i][y[i]]))
        return sim/self.d
    def Dis_Lin(self,x,y):
        sim=0
        w=0;
        for i in range(len(x)):
            w = w + math.log(self.probabilities[i][x[i]]) + math.log(self.probabilities[i][y[i]]);
        w = 1/w;
        for i in range(len(x)):
            if x[i]==y[i]:
                tmp = 2*math.log(self.probabilities[i][x[i]]);
                sim = sim+tmp*w;
            else:
                tmp =  2*math.log(self.probabilities[i][x[i]] + self.probabilities[i][y[i]]);
                sim = sim + tmp*w;
        return sim

    def Dis_Lin1(self,x,y):
        sim=0
        w=0;
        for i in range(len(x)):
            for j in range(self.nk[i]):
                if self.probabilities[i][x[i]] <= self.probabilities[i][j] and self.probabilities[i][j] <= self.probabilities[i][y[i]]:
                    w = w + math.log(self.probabilities[i][j])
        w = 1/w;
        for i in range(len(x)):
            if x[i]==y[i]:
                for j in range(self.nk[i]):
                    if self.probabilities[i][x[i]] <= self.probabilities[i][j] and self.probabilities[i][j] <= self.probabilities[i][y[i]]:
                        tmp = math.log(self.probabilities[i][j]);
                        sim = sim+tmp*w;
            else:
                for j in range(self.nk[i]):
                    if self.probabilities[i][x[i]] <= self.probabilities[i][j] and self.probabilities[i][j] <= self.probabilities[i][y[i]]:
                        tmp = 2*(self.probabilities[i][j]);
                        sim = sim+tmp*w;
        return sim
    def Dis_Goodall1(self,x,y):
        sim=0
        for i in range(len(x)):
            if x[i]==y[i]:
                sum=0;
                for j in range(self.nk[i]):
                    if self.probabilities2[i][j] <= self.probabilities2[i][x[i]]:
                        sum = sum+ self.probabilities2[i][j];
                sim= sim + 1 - sum;
        return sim/self.d

    def Dis_Goodall2(self,x,y):
        sim=0
        for i in range(len(x)):
            if x[i]==y[i]:
                sum=0;
                for j in range(self.nk[i]):
                    if self.probabilities2[i][j] >= self.probabilities2[i][x[i]]:
                        sum = sum+ self.probabilities2[i][j];
                sim= sim + 1 - sum;
        return sim/self.d
    def Dis_Goodall3(self,x,y):
        sim=0
        for i in range(len(x)):
            if x[i]==y[i]:
                sim= sim + 1 - self.probabilities2[i][x[i]];
        return sim/self.d
    def Dis_Goodall4(self,x,y):
        sim=0
        for i in range(len(x)):
            if x[i]==y[i]:
                sim= sim + self.probabilities2[i][x[i]];
        return sim/self.d

    def Dis_Goodall4(self,x,y):
        sim=0
        for i in range(len(x)):
            if x[i]==y[i]:
                sim= sim + self.probabilities2[i][x[i]];
        return sim/self.d
    def Dis_Smirnov(self,x,y):
        sim=0
        w=0;
        for i in range(len(x)):
            w = w + self.nk[i];
        w = 1/w;
        for i in range(len(x)):
            if x[i]==y[i]:
                tmp = 2 + (self.N - self.frequency[i][x[i]] )/self.frequency[i][x[i]];
                for j in range(self.nk[i]):
                    if j!= x[i]:
                        tmp = tmp + self.frequency[i][j]/(self.N - self.frequency[i][j]);
                sim = sim + tmp*w;
            else:
                tmp =0;
                for j in range(self.nk[i]):
                    if j!= x[i] or j!= y[i]:
                        tmp = tmp + self.frequency[i][j]/(self.N - self.frequency[i][j]);
                sim = sim+tmp*w;
        return sim
    def Dis_Gambaryan(self,x,y):
        sim=0
        w=0;
        for i in range(len(x)):
            w = w + self.nk[i];
        w = 1/w;
        for i in range(len(x)):
            if x[i]==y[i]:
                tmp = - (self.probabilities[i][x[i]]*math.log2(self.probabilities[i][x[i]]) + (1-self.probabilities[i][x[i]])*math.log2(1-self.probabilities[i][x[i]]))
                sim = sim + tmp*w;
        return sim
    def Dis_Burnaby(self,x,y):
        sim=0
        for i in range(len(x)):
            if x[i]==y[i]:
                sim = sim+1
            else:
                tmp=0;
                for j in range(self.nk[i]):
                    tmp = tmp + 2*math.log(1-self.probabilities[i][j])
                tmp2=0;
                tmp2=math.log( self.probabilities[i][x[i]]*self.probabilities[i][y[i]] / ((1-self.probabilities[i][x[i]])*(1-self.probabilities[i][y[i]])))
                for j in range(self.nk[i]):
                    tmp2= tmp2 + 2*math.log(1-self.probabilities[i][j])
                sim = sim+tmp/tmp2;
        return sim/self.d
    def Dis_Anderberg(self,x,y):
        tmp=0;
        for i in range(len(x)):
            if x[i]==y[i]:
                tmp =tmp +  (1/self.probabilities[i][x[i]])**2 * 2/(self.nk[i]*(self.nk[i]+1))
        tmp2=0;
        tmp3=0;
        for i in range(len(x)):
            if x[i]==y[i]:
                tmp2 = tmp2 + (1/self.probabilities[i][x[i]])**2 * 2/(self.nk[i]*(self.nk[i]+1))
            if x[i]!=y[i]:
                tmp3 = (1/(2*self.probabilities[i][x[i]]*self.probabilities[i][y[i]])) * (2/(self.nk[i]*(self.nk[i]+1)))
        sim = tmp/(tmp2+tmp3) 
        return sim

