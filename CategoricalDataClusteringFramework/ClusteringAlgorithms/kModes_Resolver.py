import os
import os.path
import sys
from sys import platform
sys.path.append(os.path.join(os.getcwd(), "Measures"))
sys.path.append(os.path.join(os.getcwd(), "LSH"))
sys.path.append(os.path.join(os.getcwd(), "../"))
sys.path.append(os.path.join(os.getcwd(), "../../"))
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
from kmodes.kmodes import KModes
from kRepresentatives import kRepresentatives
#from kmodes.util.dissim import matching_dissim

class kModes_Resolver(ClusteringAlgorithm):
    def test(self):
        print("a234 " + str(self.k))
    def DoCluster(self):
        self.name = 'kModes_Resolver'
        start_time = timeit.default_timer()
        #self.kmeans = KModes(n_clusters=self.k, random_state=0).fit(self.X)
        self.km = KModes(n_clusters=self.k, init='Huang',max_iter=self.n_iter, n_init=self.n_init, verbose=TDef.verbose)
        self.clusters = self.km.fit_predict(self.X)
        self.time_score = (timeit.default_timer() - start_time)/ self.n_init
        self.labels = self.km.labels_
        self.iter = self.km.n_iter_
        return self.labels
def main():
    MeasureManager.CURRENT_DATASET = 'soybean_small.csv'
    MeasureManager.CURRENT_MEASURE = 'DILCA'
    DB = tulti.LoadRealData(MeasureManager.CURRENT_DATASET)
    kmodes = kModes(DB['DB'],DB['labels_'] ,dbname=MeasureManager.CURRENT_DATASET)
    kmodes.SetupMeasure(MeasureManager.CURRENT_MEASURE)
    labels = kmodes.DoCluster()
    kmodes.CalcScore()

def Test(): 
    MeasureManager.CURRENT_DATASET = 'balance-scale.csv'
    MeasureManager.CURRENT_MEASURE = 'Overlap'
    if TDef.data!='': MeasureManager.CURRENT_DATASET = TDef.data
    if TDef.measure!='': MeasureManager.CURRENT_MEASURE = TDef.measure
    if TDef.test_type == 'syn':
        DB = tulti.LoadSynthesisData(TDef.n,  TDef.d, TDef.k)
        MeasureManager.CURRENT_DATASET= DB['name']
    else:
        DB = tulti.LoadRealData(MeasureManager.CURRENT_DATASET)
    print("\n\n############## kModes_Resolver ###################")
    alo = kModes_Resolver(DB['DB'],DB['labels_'] ,dbname=MeasureManager.CURRENT_DATASET ,k=TDef.k)
    alo.SetupMeasure(MeasureManager.CURRENT_MEASURE)
    alo.DoCluster()
    #alo.CalcScore()
    krepresentatives = kRepresentatives(DB['DB'],DB['labels_'] ,dbname=MeasureManager.CURRENT_DATASET ,k=TDef.k)
    krepresentatives.SetupMeasure(MeasureManager.CURRENT_MEASURE)
    krepresentatives.DoCluster(alo.labels)
    krepresentatives.CalcScore()
    

def TestDatasets(): 
    for dbname in MeasureManager.DATASET_LIST:
        DB = tulti.LoadRealData(dbname)
        MeasureManager.CURRENT_DATASET = dbname
        MeasureManager.CURRENT_MEASURE = 'Overlap'
        print("\n\n############## kModes_Resolver ###################")
        alo = kModes(DB['DB'],DB['labels_'],dbname=MeasureManager.CURRENT_DATASET )
        alo.SetupMeasure(MeasureManager.CURRENT_MEASURE)
        #alo.SetupLSH(measure=MeasureManager.CURRENT_MEASURE)
        alo.DoCluster()
        alo.CalcScore()

if __name__ == "__main__":
    TDef.InitParameters(sys.argv)
    if TDef.test_type == 'datasets':
        TestDatasets()
    else:
        Test()