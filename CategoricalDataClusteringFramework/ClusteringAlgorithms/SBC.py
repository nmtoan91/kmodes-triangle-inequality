import os
import os.path
import sys
from sys import platform
sys.path.append(os.path.join(os.getcwd(), "Measures"))
sys.path.append(os.path.join(os.getcwd(), "LSH"))
sys.path.append(os.path.join(os.getcwd(), "../"))
sys.path.append(os.path.join(os.getcwd(), "../Dataset"))
sys.path.append(os.path.join(os.getcwd(), "../Measures"))
sys.path.append(os.path.join(os.getcwd(), "../LSH"))
import numpy as np
import pandas as pd
#from kmodes_lib import KModes
import TUlti as tulti
from collections import defaultdict
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array
import timeit
from kmodes.util import get_max_value_key, encode_features, get_unique_rows, \
    decode_centroids, pandas_to_numpy
from Measures import *
from ClusteringAlgorithm import ClusteringAlgorithm
from kmodes_lib import KModes
import TDef
from sklearn.cluster import KMeans
#from kmodes.util.dissim import matching_dissim

class SBC(ClusteringAlgorithm):
    def test(self):
        print("a234 " + str(self.k))
    def Overlap_SBC(self,x,y,n):
        sum =0
        for i in range(n):
            if x[i] == y[i]: sum +=1
        return sum/n
    def StructureData(self):
        d = len(self.X[0])
        self.X2 = np.zeros((self.n,self.n))
        for i in range(self.n):
            self.X2[i][i] =1
            for j in range(i+1,self.n):
                self.X2[i][j] =  self.Overlap_SBC(self.X[i],self.X[j],d)
                self.X2[j][i] = self.X2[i][j]

    def DoCluster(self):
        self.name = 'SBC'
        start_time = timeit.default_timer()
        self.StructureData()
        self.kmeans = KMeans(n_clusters=self.k, random_state=0,max_iter=self.n_iter, n_init = self.n_init, verbose=TDef.verbose).fit(self.X2)
        self.time_score = (timeit.default_timer() - start_time)/ self.n_init
        self.labels = self.kmeans.labels_
        self.iter = self.kmeans.n_iter_
        return self.labels

if __name__ == "__main__":
    MeasureManager.CURRENT_DATASET = 'soybean_small.csv'
    MeasureManager.CURRENT_MEASURE = 'Overlap'
    if TDef.data!='': MeasureManager.CURRENT_DATASET = TDef.data
    if TDef.measure!='': MeasureManager.CURRENT_MEASURE = TDef.measure
    if TDef.test_type == 'syn':
        DB = tulti.LoadSynthesisData(TDef.n,  TDef.d, TDef.k)
        MeasureManager.CURRENT_DATASET= DB['name']
    else:
        DB = tulti.LoadRealData(MeasureManager.CURRENT_DATASET)
    print("\n\n############## kMeanspp ###################")
    alo = SBC(DB['DB'],DB['labels_'] ,dbname=MeasureManager.CURRENT_DATASET ,k=TDef.k)
    alo.SetupMeasure(MeasureManager.CURRENT_MEASURE)
    #alo.SetupLSH(measure=MeasureManager.CURRENT_MEASURE)
    alo.DoCluster()
    alo.CalcScore()