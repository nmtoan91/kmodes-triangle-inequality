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

class kModesBaseline(ClusteringAlgorithm):
    def test(self):
        print("a234 " + str(self.k))

    def DoCluster(self):
        global kModesPlain_measure
        if kModesPlain_measure== None: 
            kModesPlain_measure= Overlap(self.dbname)
            kModesPlain_measure.setUp(self.X, self.y)
        
        self.name = 'kModesPlain'
        start_time = timeit.default_timer()
        
        np.random.seed(41)
        centers = []

        #init random clusters
        for k in range(self.k):
            center = np.zeros((self.d), int)
            for d in range(self.d):
                center[d] = np.random.randint(0, self.D[d])
            centers.append(center)
        

        #Loop
        self.n_iter =10 #test
        for iter in tqdm(range(self.n_iter)):
            # Computer distances from items to centers
            minIndexs = np.zeros((self.n),int)
            for i in range(self.n):
                dists = sklearn.metrics.pairwise_distances(self.X[i].reshape(1, -1), centers, metric = overlapMetric)
                minIndexs[i] = np.argmin(dists,1)
 #           dists =  sklearn.metrics.pairwise_distances(self.X, centers, metric = overlapMetric)
 #           minIndexs = np.argmin(dists,1)
            
            # Calc frequencies of categorial attributes in each cluster
            frequencies = []
            for k in range(self.k):
                frequencies_k = []
                for d in range(self.d):
                    frequencies_k.append(np.zeros((self.D[d])))
                frequencies.append(frequencies_k)

            for i in range(self.n):
                minIndex = minIndexs [i]
                for d in range(self.d):
                    frequencies[minIndex][d][self.X[i][d]] += 1

            # Extract the highest frequencies attibutes
            for k in range(self.k):
                centers[k] = np.argmax(frequencies[k],1)
                
        self.time_score = (timeit.default_timer() - start_time)/ self.n_init
        print( " Time:", self.time_score)
        self.labels = minIndexs
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

    alg2 = kModesBaseline(X,y,dbname = dataFile)
    alg2.DoCluster()
    alg2.CalcScore()

