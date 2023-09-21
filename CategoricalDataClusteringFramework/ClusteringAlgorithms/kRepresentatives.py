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
class kRepresentatives(ClusteringAlgorithm):
    def SetupMeasure(self, classname):
        module = __import__(classname, globals(), locals(), ['object'])
        class_ = getattr(module, classname)
        self.measure = class_()
        self.measure.setUp(self.X, self.y)
    def test(self):
        print("a234 " + str(self.k))
    def Distance(self,representative, point):
        sum=0;
        for i in range (self.d):
            sum = sum + representative[i][point[i]]
        return self.d - sum
    def DistanceRepresentativestoAPoints(self,representatives, point):
        return [self.Distance(c, point) for c in representatives]
    def MovePoint(self, point_id, from_id, to_id ,representatives_count, representatives_sum,membship, curpoint):
        membship[to_id, point_id] = 1
        membship[from_id, point_id] = 0
        representatives_sum[to_id]+=1
        representatives_sum[from_id]-=1
        for ii, val in enumerate(curpoint):
            representatives_count[to_id][ii][val]+=1
            representatives_count[from_id][ii][val]-=1
   
        
    def UpdateLabels(self,representatives, X,representatives_sum, representatives_count,membship):
        cost = 0
        move = 0
        for ipoint, curpoint in enumerate(X):
            dist_matrix = self.DistanceRepresentativestoAPoints(representatives, curpoint)
            representative_id = np.argmin(dist_matrix)
            cost += dist_matrix[representative_id]
            if membship[representative_id, ipoint]: continue
            indices = np.argwhere(membship[:, ipoint])
            if len(indices)>0:
                old_clust = indices[0][0]
                self.MovePoint(ipoint, old_clust,representative_id, representatives_count, representatives_sum,membship,curpoint  )
                move +=1
            else: # First
                membship[representative_id, ipoint] = 1
                representatives_sum[representative_id]+=1
                for ii, val in enumerate(curpoint):
                    representatives_count[representative_id][ii][val]+=1
        #Check empty clusters
        count_empty =0
        big_cluster_id = -1
        for ki in range(self.k):
            if representatives_sum[ki] ==0 :
                #print("FOUND A EMPTY CLUSTER");
                count_empty += 1
                #if big_cluster_id ==-1:
                big_cluster_id = np.argmax([sum(mem_) for mem_ in membship])
                choices = [i for i in range(self.n) if membship[big_cluster_id][i] == 1 ]
                rindx = self.random_state.choice(choices)
                self.MovePoint(rindx, ki,big_cluster_id, representatives_count, representatives_sum,membship,self.X[rindx]  )
                move +=1
        #Calc cost
        return cost ,move, count_empty


    def UpdateRepresentatives(self,representatives,representatives_sum,representatives_count ) :  
        for ki in range(self.k):
            for di in range(self.d):
                    for vj in range(self.D[di]):
                        representatives[ki][di][vj] =  representatives_count[ki][di][vj]/representatives_sum[ki]
        return 0
    def GetLabels(self, membship):
        labels = np.empty(self.n, dtype=np.uint16)
        for ki in range(self.k):
            for i in range(self.n):
                if membship[ki][i]:
                    labels[i] = ki
        return labels
    def InitClusterRandomly(self,representatives,membship,representatives_sum,representatives_count ):
        # def InitClusters(self,representatives,representatives_sum,representatives_count):
        for ki in range(self.k):
            for i in range(self.d):
                sum_ = 0
                for j in range(self.D[i]): sum_ = sum_ + representatives[ki][i][j]
                for j in range(self.D[i]): representatives[ki][i][j] = representatives[ki][i][j]/sum_;
    def InitClusterFromLabel(self, labels,membship,representatives_sum,representatives_count ):
        for i, curpoint in enumerate(self.X):
            representative_id = labels[i]
            membship[representative_id, ipoint] = 1
            representatives_sum[representative_id]+=1
            for ii, val in enumerate(curpoint):
                representatives_count[representative_id][ii][val]+=1

    def DoCluster(self, plabels=np.zeros(0)):
        self.name = "kRepresentatives"
        #Init varibles
        X = self.X
        self.k = k = n_clusters = self.k
        self.n = n = self.X.shape[0];
        self.d = d = X.shape[1]
        self.D = D = [len(np.unique(X[:,i])) for i in range(d) ]

        all_labels = []
        all_costs = []
        start_time = timeit.default_timer()
        for init_no in range(self.n_init):
            if TDef.verbose >=1: print ('kRepresentatives Init ' + str(init_no))
            self.random_state = check_random_state(None)
            membship = np.zeros((k, n), dtype=np.uint8)
            representatives_count = [[[0 for i in range(D[j])] for j in range(d)]for ki in range(k)]
            representatives_sum = [0 for ki in range(k)]

            representatives = [[[random.uniform(0,1) for i in range(D[j])] for j in range(d)] for ki in range(k)]

            if len(plabels) == 0:
                self.InitClusterRandomly(representatives,membship,representatives_sum,representatives_count)
            else: 
                self.InitClusterFromLabel(plabels, membship, representatives_sum,representatives_count)
                self.UpdateRepresentatives(representatives,representatives_sum,representatives_count ) ;
            #Init first cluster
            last_cost = float('inf')
            for i in range(self.n_iter):
                start_time_iter = timeit.default_timer()
                self.iter = i
                cost ,move, count_empty = self.UpdateLabels(representatives, X,representatives_sum, representatives_count,membship)
                self.UpdateRepresentatives(representatives,representatives_sum,representatives_count ) ;
                #print("Cost: ", cost," ;Move: ", move, " ;Num empty: ", count_empty)
                if TDef.verbose >=2: print ('Iter ' + str(i)," Cost:", "%.2f"%cost," Move:", move, " Num empty:", count_empty," Timelapse:", "%.2f"%(timeit.default_timer()-start_time_iter) )
                if last_cost == cost and move==0:
                    break 
                last_cost = cost
            labels = self.GetLabels(membship)
            all_costs.append(cost)
            all_labels.append(labels)

        best = np.argmin(all_costs)
        labels = all_labels[best]
        self.time_score = (timeit.default_timer() - start_time)/ self.n_init
        self.labels = labels
        print("Score: ", all_costs[best] , " Time:", self.time_score)
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
    print("\n\n############## kRepresentatives ###################")
    algo = kRepresentatives(DB['DB'],DB['labels_'],k=TDef.k ,dbname=MeasureManager.CURRENT_DATASET)
    algo.SetupMeasure(MeasureManager.CURRENT_MEASURE)
    algo.DoCluster()
    algo.CalcScore()

def TestDatasets(): 
    for dbname in MeasureManager.DATASET_LIST:
        DB = tulti.LoadRealData(dbname)
        MeasureManager.CURRENT_DATASET = dbname
        MeasureManager.CURRENT_MEASURE = 'Overlap'
        print("\n\n############## kRepresentatives ###################")
        alo = kRepresentatives(DB['DB'],DB['labels_'],dbname=MeasureManager.CURRENT_DATASET ,k=TDef.k)
        alo.SetupMeasure(MeasureManager.CURRENT_MEASURE)
        alo.DoCluster()
        alo.CalcScore()

if __name__ == "__main__":
    TDef.InitParameters(sys.argv)
    if TDef.test_type == 'datasets':
        TestDatasets()
    else:
        Test()
