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
import random
import scipy
class MOGA(ClusteringAlgorithm):
    def Overlap_SBC(self,x,y,n):
        sum =0
        for i in range(n):
            if x[i] == y[i]: sum +=1
        return sum/n
    def Overlap(self, x, y):
        sum =0
        for i in range(self.d):
            if x[i] != y[i]: sum +=1
        return sum
    def CrossOver(self):
        for pi in range(self.pop_index_cro[0],self.pop_index_cro[1]):
            i = random.randint(self.pop_index_pop[0],self.pop_index_pop[1]-1)
            j = random.randint(self.pop_index_pop[0],self.pop_index_pop[1]-1)
            for ki in range(self.k):
                for di in range(self.d):
                    self.chromosomes[pi][ki][di] = self.chromosomes[i][ki][di] if random.randint(0,1)==0 else self.chromosomes[j][ki][di]
    def Mutation(self):
        for pi in range(self.pop_index_mut[0],self.pop_index_mut[1]):
            i = random.randint(self.pop_index_pop[0],self.pop_index_pop[1]-1)
            for ki in range(self.k):
                for di in range(self.d):
                    self.chromosomes[pi][ki][di] = self.chromosomes[i][ki][di] if random.randint(0,1)==0 else random.randint(0,self.D[di]-1)
        

    def Selection(self):
        #Check duplicated objects
        for pi in range(1,self.pop_index_mut[1]):
            for pi2 in range(0,pi):
                if (self.chromosomes[pi] ==self.chromosomes[pi2]).all():
                    for ki in range(self.k):
                        for di in range(self.d):
                            if random.randint(0,1)==1: self.chromosomes[pi][ki][di] = random.randint(0,self.D[di]-1)
                    break

        for pi in range(0,self.pop_index_mut[1]):
            self.chromosome_scores[pi] = self.CalcScore_(pi)
        indexes = np.argsort(self.chromosome_scores)
        self.converged = False
        for i in range(0,self.pop_index_pop[1]):
            #if i != indexes[i]: self.converged = False
            self.chromosome_scores[i] = self.chromosome_scores[indexes[i]]
            self.chromosomes[i] = self.chromosomes[indexes[i]]
            asd=123
        return self.chromosome_scores[0]

    def CalcScore_(self,pi):
        centroids = self.chromosomes[pi]
        dist_matrix = scipy.spatial.distance.cdist(self.X, centroids, self.Overlap)
        return sum(np.min(dist_matrix,1)) 
    def Init(self): #Cao 
        self.chromosomes = np.empty((self.popsize, self.k, self.d), dtype='object')
        for pi in range(self.pop_index_pop[1]):
            for ki in range(self.k):
                for di in range(self.d):
                    self.chromosomes[pi][ki][di] =  random.randint(0,self.D[di]-1)
            #self.chromosome_scores[pi] = self.CalcScore_(pi)

    def DoCluster(self):
        self.popsize = 3*4*self.k
        self.chromosome_scores = np.zeros((self.popsize))
        self.pop_index_pop = (0,4*self.k)
        self.pop_index_cro = (4*self.k,2*4*self.k)
        self.pop_index_mut = (2*4*self.k,3*4*self.k)

        self.D = D = [len(np.unique(self.X[:,i])) for i in range(self.d) ]
        self.name = 'MOGASim'
        self.labels = np.zeros((self.n));self.iter=0;
        start_time = timeit.default_timer()
        self.Init()
        for self.iter in range(self.n_iter):
            self.CrossOver()
            self.Mutation()
            score=self.Selection()
            print("Iter:", self.iter, " score:", score)
            if self.converged: break
        self.labels = np.argmin(scipy.spatial.distance.cdist(self.X, self.chromosomes[1], self.Overlap),1)
        print(self.labels)
        #self.kmeans = KMeans(n_clusters=self.k, random_state=0,max_iter=self.n_iter, n_init = self.n_init, verbose=TDef.verbose).fit(self.X2)
        self.time_score = (timeit.default_timer() - start_time)/ self.n_init
        #self.iter = self.kmeans.n_iter_
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
    print("\n\n############## MOGASim  ###################")
    alo = MOGA(DB['DB'],DB['labels_'] ,dbname=MeasureManager.CURRENT_DATASET ,k=TDef.k)
    alo.SetupMeasure(MeasureManager.CURRENT_MEASURE)
    #alo.SetupLSH(measure=MeasureManager.CURRENT_MEASURE)
    alo.DoCluster()
    alo.CalcScore()