#paper: https://ieeexplore.ieee.org/ielx7/6287639/8600701/08681032.pdf?tag=1&fbclid=IwAR3hroHd8uORyROBSlfgagpqp60VJ0xVTbI0KiKjG01Rinsw5L1fAL7D9ag&tag=1
#A Hybrid MPI/OpenMP Parallelization of K-Means Algorithms Accelerated Using the Triangle Inequality
import os.path
import sys
from tqdm import tqdm
sys.path.append("..")
sys.path.append(".")
from sys import platform

from CategoricalDataClusteringFramework.Measures.MeasureManager import MeasureManager
from CategoricalDataClusteringFramework.Measures.Overlap import Overlap
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array
import timeit
from kmodes.util import get_max_value_key, encode_features, get_unique_rows, \
    decode_centroids, pandas_to_numpy



from CategoricalDataClusteringFramework.ClusteringAlgorithms.ClusteringAlgorithm import ClusteringAlgorithm


import sklearn
kModesPlain_measure = None
def overlapMetric(x1,x2):
    return kModesPlain_measure.calculate(x1.astype(np.int32),x2.astype(np.int32))
    #return 0

class kModesTriangleInequality_IEEEAccess(ClusteringAlgorithm):
    def test(self):
        print("a234 " + str(self.k))

    def DoCluster(self,seed = 41):
        self.AddVariableToPrint('seed', seed)
        global kModesPlain_measure
        if kModesPlain_measure== None: 
            kModesPlain_measure= Overlap(self.dbname)
            kModesPlain_measure.setUp(self.X, self.y)
        
        self.name = 'kModesTriangleInequality_IEEEAccess'
        start_time = timeit.default_timer()
        
        np.random.seed(seed)
        c = []
        c2 = []

        #init random clusters
        for k in range(self.k):
            center = np.zeros((self.d), int)
            mc_item = np.zeros((self.d), int)
            for d in range(self.d):
                mc_item[d] = center[d] = np.random.randint(0, self.D[d])
                 
            c.append(center)
            c2.append(mc_item)
        
        
        #Start
        a = np.ones((self.n),int)
        u = np.ones((self.n))*100000000
        l = np.zeros((self.n, self.k))
        delta = np.zeros((self.k))

        self.n_iter =10 #test
        #for iter in tqdm(range(self.n_iter)):
        for iter in range(self.n_iter):
            #6,7
            C = sklearn.metrics.pairwise_distances(c, c, metric = overlapMetric)
            s = np.ones(self.k)*100000
            for k1 in range(self.k):
                for k2 in range(self.k):
                    if k1 == k2: continue
                    if C[k1,k2] < s[k1] : s[k1] = C[k1,k2]
                s[k1]/=2
            #8
#            count = 0
#            count1 = 0
            hard_indices = []
            for i in range(self.n):
                if u[i] > s[a[i]]:
                    r = True
                    for k in range(self.k):
                        z = max(l[i,k],C[a[i],k]/2)
                        if k == a[i] or u[i] <= z: continue
                        if r :
                            u[i] = overlapMetric(self.X[i], c[a[i]])
                            r = False
                            if u[i] <= z: continue
                        l[i,k] = overlapMetric(self.X[i], c[k])
                        if l[i,k] < u[i]:
                            a[i] = k
                            u[i] = l[i,k]
                            continue
#                    count1 += 1
                    hard_indices.append(i)
                    # l[i,:] = sklearn.metrics.pairwise_distances(self.X[i].reshape(1, -1), c, metric = overlapMetric)
                    # a[i] = np.argmin(l[i,:])
                    # u[i] = l[i,a[i]]
 #               else:
 #                   count += 1
            l_hard_indices = sklearn.metrics.pairwise_distances(self.X[hard_indices], c, metric = overlapMetric)  
            l[hard_indices,:] = l_hard_indices
            a[hard_indices] = np.argmin(l_hard_indices,1)
            u[hard_indices] = [l_hard_indices[i, a[hard_indices[i]]] for i in range(len(l_hard_indices))]
            
            #22
            # Calc frequencies of categorial attributes in each cluster
            c2 = np.copy(c)
            
            frequencies = []
            for k in range(self.k):
                frequencies_k = []
                for d in range(self.d):
                    frequencies_k.append(np.zeros((self.D[d])))
                frequencies.append(frequencies_k)

            for x in range(self.n):
                minIndex = a[x]
                for d in range(self.d):
                    frequencies[minIndex][d][self.X[x][d]] += 1

            # Extract the highest frequencies attibutes
            for k in range(self.k):
                c[k] = np.argmax(frequencies[k],1)
                
            #23
            for k in range(self.k):
                delta[k] =  overlapMetric(c[k], c2[k])
            #24-26
            for i in range(self.n):
                u[i] = u[i] + delta[a[i]]
                for k in range(self.k):
                    l[i,k] = l[i,k] - delta[k]
#            print(iter, count, count1)      
#            print(a[1:20])
 
        self.time_score = (timeit.default_timer() - start_time)/ self.n_init
        print( " Time:", self.time_score)
#        print(a[1:30])
        self.labels = a
        return self.labels

#for fast test 
if __name__ == '__main__':
    
    dataPath = './DataSample/'
    #dataFile = 'SYN_100000_16_256_8_10.csv'
    dataFile = 'SYN_512_10_20_8_10.csv'
    data = pd.read_csv(dataPath+dataFile, header=None)
    X = data.to_numpy(int)
    y = X[:,X.shape[1]-1]
    X = X[:,0:X.shape[1]-1]

    alg2 = kModesTriangleInequality_IEEEAccess(X,y,dbname = dataFile)
    alg2.DoCluster()
    alg2.CalcScore()

