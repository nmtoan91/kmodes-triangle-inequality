import os.path
import sys
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

class kModesTriangleInequality(ClusteringAlgorithm):
    def test(self):
        print("a234 " + str(self.k))

    def DoCluster(self):
        global kModesPlain_measure
        if kModesPlain_measure== None: 
            kModesPlain_measure= Overlap(self.dbname)
            kModesPlain_measure.setUp(self.X, self.y)
        
        self.name = 'kModesTriangleInequality'
        start_time = timeit.default_timer()
        
        np.random.seed(41)
        centers = []
        mc = []

        #init random clusters
        for k in range(self.k):
            center = np.zeros((self.d), int)
            mc_item = np.zeros((self.d), int)
            for d in range(self.d):
                center[d] = np.random.randint(0, self.D[d])
                mc_item[d] = np.random.randint(0, self.D[d])
            centers.append(center)
            mc.append(mc_item)
        
        dxc =  sklearn.metrics.pairwise_distances(self.X, centers, metric = overlapMetric)
        

        #init parameters
        lxc = dxc
        cx = np.argmin(dxc,1)
        ux = np.min(lxc,1)
        rx = np.full((self.n), False)

        #New loop
        self.n_iter =10 #test
        for iter in range(self.n_iter):
            #1
            dcc = sklearn.metrics.pairwise_distances(centers, centers, metric = overlapMetric)
            sc = np.ones(self.k)*100000
            for k1 in range(self.k):
                for k2 in range(self.k):
                    if k1 == k2: continue
                    if dcc[k1,k2] < sc[k] : sc[k] = dcc[k1,k2]
                sc[k1]/=2


            #3
            for x in range(self.n):
                #2
                if ux[x] <= sc[cx[x]]: continue 
                
                
                for c in range(self.k):
                    if c == cx[x]: continue
                    if ux[x] <= lxc[x,c]: continue
                    if ux[x] <= 0.5*dcc[cx[x]  ,c]: continue

                    #3a
                    if rx[x]:
                        dxc[x,cx[x]]  = overlapMetric(self.X[x], centers[cx[x]])
                        rx[x] = False
                    else:
                        dxc[x,cx[x]] = ux[x]

                    #3b
                    if dxc[x,cx[x]] > lxc[x,c] or dxc[x,cx[x]] > 0.5*dcc[c,cx[x]]:
                        dxctmp = overlapMetric(self.X[x], centers[c])
                        if dxctmp < dxc[x,cx[x]]:
                            cx[x] = c 
            #4
            # Calc frequencies of categorial attributes in each cluster
            frequencies = []
            for k in range(self.k):
                frequencies_k = []
                for d in range(self.d):
                    frequencies_k.append(np.zeros((self.D[d])))
                frequencies.append(frequencies_k)

            for x in range(self.n):
                minIndex = cx [x]
                for d in range(self.d):
                    frequencies[minIndex][d][self.X[x][d]] += 1

            # Extract the highest frequencies attibutes
            for k in range(self.k):
                for d in range(self.d):
                    mc[k][d] = np.argmax(frequencies[k][d])

            #5
            for x in range(self.n):
                for c in range(self.k):
                    lxc[x,c] = np.max(lxc[x,c] - overlapMetric(centers[c], mc[c]  ),0 )
            #6
            for x in range(self.n):
                ux[x] = ux[x] + overlapMetric(mc[cx[x]],centers[cx[x]] )
                rx[x] = True
            #7
            for c in range(self.k):
                centers[c] = mc[c]


            #exit(0)
        


        self.time_score = (timeit.default_timer() - start_time)/ self.n_init
        self.labels = cx
        return self.labels


#for fast test 
if __name__ == '__main__':
    
    dataPath = 'D:/DATA/CATEGORICAL/SYN/'
    dataFile = 'SYN_512_20_20_8_10.csv'
    data = pd.read_csv(dataPath+dataFile, header=None)
    X = data.to_numpy(int)
    y = X[:,X.shape[1]-1]
    X = X[:,0:X.shape[1]-1]

    alg2 = kModesTriangleInequality(X,y)
    alg2.DoCluster()
    alg2.CalcScore()

