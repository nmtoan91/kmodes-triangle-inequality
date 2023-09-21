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
sys.path.append(os.path.join(os.getcwd(), "./ClusteringAlgorithms"))

import numpy as np
import pandas as pd
#from kmodes_lib import KModes

from collections import defaultdict
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array
import timeit
from kmodes.util import get_max_value_key, encode_features, get_unique_rows, \
    decode_centroids, pandas_to_numpy

from ClusteringAlgorithm import ClusteringAlgorithm
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import homogeneity_score
import random
import TDef
import TUlti as tulti
from Measures import *
class kCenters(ClusteringAlgorithm):
    def SetupMeasure(self, classname):
        module = __import__(classname, globals(), locals(), ['object'])
        class_ = getattr(module, classname)
        self.measure = class_()
        self.measure.setUp(self.X, self.y)
    
    def DistanceRepresentativestoAPoints(self,representatives, point):
        return [self.Distance(c, point) for c in centers]

    def CalcDistMatrix(self):
        for i in range(self.n):
            for j in range(self.k):
                 tmp=self.Distance(self.centers[j],self.X[i],j)
                 self.dist_matrix[i][j] = tmp
    def UpdateLabels(self):
        asd=123

    def UpdateLabelsFirst(self):
        self.CalcDistMatrix()
        for i in range(self.n):
            min_id = np.argmin(self.dist_matrix[i])
            self.labels[i] = min_id
            self.representatives_sum[min_id]+=1
            for ii, val in enumerate(self.X[i]):
                self.representatives_count[min_id][ii][val]+=1

    def MoveAPoint(self,i,from_id,to_id):
        if self.representatives_sum[from_id] >1:
            self.labels[i] = to_id
            self.representatives_sum[to_id]+=1
            self.representatives_sum[from_id]-=1
            for ii, val in enumerate(self.X[i]):
                self.representatives_count[to_id][ii][val]+=1
                self.representatives_count[from_id][ii][val]-=1
            return 1
        return 0

    def UpdateLabels(self):
        cost=0
        move=0
        self.CalcDistMatrix()
        for i in range(self.n):
            min_id = np.argmin(self.dist_matrix[i])
            cost+= self.dist_matrix[i][min_id]
            last_id = self.labels[i]
            if min_id!= last_id:
                move+=self.MoveAPoint(i,last_id,min_id)
        return cost,move
    def Distance2(self,representative, point,ki):
        sum=0;
        for i in range (self.d):
            sum = sum + representative[i][point[i]]
        return (self.d - sum)#/self.d
    def Distance(self,representative,point,ki):
        sum=0;
        for i in range (self.d):
            for vj in range(self.D[i]):
                if point[i] == vj:
                    tmp=  self.W[ki][i]* (1-representative[i][vj])**2
                else: tmp= self.W[ki][i]*(0-representative[i][vj])**2
                sum+= tmp
        return sum**0.5
    def UpdateLambdas(self):
        for ki in range(self.k):
            for di in range(self.d):
                    for vj in range(self.D[di]):
                        self.representatives_only[ki][di][vj] =  self.representatives_count[ki][di][vj]/self.representatives_sum[ki]

        for ki in range(self.k):
            if self.representatives_sum[ki]>1:
                tmp = (1/(self.representatives_sum[ki]-1))
            else: tmp = (1/(self.representatives_sum[ki]))

            numerator=0
            denominator=0
            for di in range(self.d):
                numerator_child=0
                for vj in range(self.D[di]):
                    numerator_child+= self.representatives_only[ki][di][vj]**2
                numerator+= 1 - numerator_child
                denominator+= numerator_child - 1/ self.D[di]
                
            self.lambdas[ki] = tmp*numerator/denominator
        asd=123

        print("Lambdas:",self.lambdas)

    def UpdateCenters(self):
        for ki in range(self.k):
            for di in range(self.d):
                right = 1 - (self.lambdas[ki]**2) /self.D[di];
                tmp= self.lambdas[ki]**2-1
                tmp2=0
                for vj in range(self.D[di]):
                    tmp2 = self.representatives_only[ki][di][vj]**2
                right = right+tmp*tmp2
                tmp/=-self.beta
                self.W[ki][di] = 10**tmp
        #Now update centers
        for ki in range(self.k):
             for di in range(self.d):
                    for vj in range(self.D[di]):
                        self.centers[ki][di][vj] = self.lambdas[ki]/self.D[di]  + (1-self.lambdas[ki])*self.representatives_only[ki][di][vj]

    def DoCluster(self, plabels=np.zeros(0)):
        self.name = "kCenters"
        #Init varibles
        X = self.X
        self.k = k = n_clusters = self.k
        self.n = n = self.X.shape[0];
        self.d = d = X.shape[1]
        self.D = D = [len(np.unique(X[:,i])) for i in range(d) ]
        self.beta = 0.5
        all_labels = []
        all_costs = []
        start_time = timeit.default_timer()
        self.dist_matrix = np.zeros((self.n, self.k))

        for init_no in range(self.n_init):
            if TDef.verbose >=1: print ('kCenters Init ' + str(init_no))
            self.random_state = check_random_state(None)

            self.lambdas = np.zeros(self.k)
            #self.membship = np.zeros((k, n), dtype=np.uint8)
            self.labels = np.zeros(self.n,dtype=int)
            self.W = np.ones((self.k, self.d))/d
            self.representatives_count = [[[0 for i in range(D[j])] for j in range(d)]for ki in range(k)]
            self.representatives_sum = np.zeros(k)
            self.centers = [[[random.uniform(0,1) for i in range(D[j])] for j in range(d)] for ki in range(k)]
            self.representatives_only = [[[random.uniform(0,1) for i in range(D[j])] for j in range(d)] for ki in range(k)]
            last_cost = float('inf')
            self.UpdateLabelsFirst();
            asd=123
            for i in range(self.n_iter):
                start_time_iter =  timeit.default_timer()
                cost,move=self.UpdateLabels()
                self.UpdateLambdas()
                self.UpdateCenters()
                if(last_cost==cost and move==0): break;
                last_cost=cost
                if TDef.verbose >=2: print ('Iter ' + str(i)," Cost:", "%.2f"%cost," Move:", move," Timelapse:", "%.2f"%(timeit.default_timer()-start_time_iter) )

            all_costs.append(cost)
            all_labels.append(self.labels)

        best = np.argmin(all_costs)
        self.labels = all_labels[best]
        self.time_score = (timeit.default_timer() - start_time)/ self.n_init
        self.scorebest = all_costs[best]
        return self.labels

def Test(): 
    MeasureManager.CURRENT_DATASET = 'soybean_small.csv' 
    MeasureManager.CURRENT_MEASURE = 'Overlap'
    if TDef.data!='': MeasureManager.CURRENT_DATASET = TDef.data
    if TDef.measure!='': MeasureManager.CURRENT_MEASURE = TDef.measure
    if TDef.test_type == 'syn':
        DB = tulti.LoadSynthesisData(TDef.n,  TDef.d, TDef.k)
        MeasureManager.CURRENT_DATASET= DB['name']
    else:
        DB = tulti.LoadRealData(MeasureManager.CURRENT_DATASET)
    print("\n\n############## kCenters ###################")
    algo = kCenters(DB['DB'],DB['labels_'],k=TDef.k ,dbname=MeasureManager.CURRENT_DATASET)
    algo.SetupMeasure(MeasureManager.CURRENT_MEASURE)
    algo.DoCluster()
    algo.CalcScore()



if __name__ == "__main__":
    TDef.InitParameters(sys.argv)
    if TDef.test_type == 'datasets':
        TestDatasets()
    else:
        Test()
