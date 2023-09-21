#TOANSTT Copy idea from MH-k-Kmodes

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
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import homogeneity_score
import random
from LSH import LSH as LSH
from kModes import kModes
from kRepresentatives import kRepresentatives
from TUlti import MyTable
import multiprocessing as mp


class LSHkRepresentatives(ClusteringAlgorithm):
    def SetupLSH(self, hbits=-1,k=-1,measure='DILCA' ):
        start = timeit.default_timer()
        self.lsh = LSH.LSH(self.X,self.y,measure=measure,hbits=hbits)
        self.lsh.DoHash()
        self.lsh.CorrectSingletonBucket()
        #print("SETING UP: ",self.n,self.d,self.k,hbits,'->',self.lsh.hbits, "avg=" + str(self.n/ (2**self.lsh.hbits))  )
        self.time_lsh = timeit.default_timer() - start
        self.AddVariableToPrint("Time_lsh",self.time_lsh)
        return self.time_lsh
    def SetupMeasure(self, classname):
        self.measurename = classname
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
    
    def MovePoint(self, point_id, from_id, to_id ,representatives_count, representatives_sum,membship, curpoint,labels_matrix):
        labels_matrix[point_id] = to_id
        membship[to_id, point_id] = 1
        membship[from_id, point_id] = 0
        representatives_sum[to_id]+=1
        representatives_sum[from_id]-=1 
        for ii, val in enumerate(curpoint):
            representatives_count[to_id][ii][val]+=1
            representatives_count[from_id][ii][val]-=1
    def CheckEmptyClusters(self,representatives, X,representatives_sum, representatives_count,membship,labels_matrix):
        move =0
        big_cluster_id = -1
        for ki in range(self.k):
            if representatives_sum[ki] ==0 :
                #print("FOUND A EMPTY CLUSTER");
                #count_empty += 1
                #if big_cluster_id ==-1:
                big_cluster_id = np.argmax([sum(mem_) for mem_ in membship])
                choices = [i for i in range(self.n) if membship[big_cluster_id][i] == 1 ]
                rindx = self.random_state.choice(choices)
                self.MovePoint(rindx, big_cluster_id,ki, representatives_count, representatives_sum,membship,self.X[rindx],labels_matrix  )
                move +=1
        return move
    def InitClusters(self,representatives,representatives_sum,representatives_count):
        for ki in range(self.k):
            for i in range(self.d):
                sum_ = 0
                for j in range(self.D[i]): sum_ = sum_ + representatives[ki][i][j]
                for j in range(self.D[i]): representatives[ki][i][j] = representatives[ki][i][j]/sum_;

    def DistanceRepresentativestoAPoints(self,representatives, point):
        dist_matrix = [self.Distance(c, point) for c in representatives]
        representative_id = np.argmin(dist_matrix)
        return representative_id, dist_matrix[representative_id]

    
    def UpdateLabelsInit(self,representatives, X,representatives_sum, representatives_count,membship,labels_matrix):
        cost = 0
        move = 0
        self.preferList= defaultdict(set)
        for ipoint, curpoint in enumerate(X):
            representative_id,tmp = self.DistanceRepresentativestoAPoints(representatives, curpoint)
            cost += tmp
            labels_matrix[ipoint] = representative_id
            membship[representative_id, ipoint] = 1
            representatives_sum[representative_id]+=1
            for ii, val in enumerate(curpoint):
                representatives_count[representative_id][ii][val]+=1
            self.preferList[self.lsh.hash_values[ipoint]].add(labels_matrix[ipoint])
        self.CheckEmptyClusters(representatives, X,representatives_sum, representatives_count,membship,labels_matrix)
        self.dist_matrix_tmp = [1000000000 for i in range(self.k)]
        return cost ,move, 0
    def UpdateLabelsLast(self,representatives, X,representatives_sum, representatives_count,membship,labels_matrix):
        cost = 0
        self.preferList= defaultdict(set)
        for ipoint, curpoint in enumerate(X):
            representative_id,tmp = self.DistanceRepresentativestoAPoints(representatives, curpoint)
            cost += tmp
            #labels_matrix[ipoint] = representative_id
            membship[representative_id, ipoint] = 1
        return cost ,0, 0
    def DistanceRepresentativestoAPoints_LSH2(self,item_id, point,labels_matrix,representatives):
        neighbors = self.lsh.GetNeighborsbyBucket(item_id)
        candidates = [labels_matrix[i] for i in neighbors]
        candidates_unique = np.unique(candidates)
        if(len(candidates_unique)==1): 
            candidates_unique = np.append(candidates_unique, [(candidates_unique[0] + int(random.uniform(1,self.k)))%self. k])
            
        dist_matrix = [self.Distance(representatives[i], point) for i in candidates_unique]
        temp_index = np.argmin(dist_matrix)
        return candidates_unique[temp_index], dist_matrix[temp_index]  

    def DistanceRepresentativestoAPoints_LSH1(self,item_id, point,labels_matrix,representatives):
        neighbors = self.lsh.GetNeighborsbyBucket(item_id)
        min_clust_index = labels_matrix[item_id]
        min_clust_score = 1000000000
        for i in range(self.k):
            self.dist_matrix_tmp[i] = 1000000000
        for i in range(len(neighbors)):
            clust = labels_matrix[neighbors[i]]
            if self.dist_matrix_tmp[clust] == 1000000000:
                self.dist_matrix_tmp[clust] = self.Distance(representatives[clust], point)
                if min_clust_score > self.dist_matrix_tmp[clust]:
                    min_clust_score = self.dist_matrix_tmp[clust]
                    min_clust_index = clust
        return min_clust_index, min_clust_score


    def DistanceRepresentativestoAPoints_LSH(self,item_id, point,labels_matrix,representatives):
        myset = self.preferList[self.lsh.hash_values[item_id]]
        dist_min = 1000000000
        dist_index =-1
        for i in myset:
            dist = self.Distance(representatives[i], point)
            if dist_min > dist:
                dist_min = dist 
                dist_index = i
        return dist_index, dist_min 

    def UpdateLabels(self,representatives, X,representatives_sum, representatives_count,membship,labels_matrix):
        cost = 0
        move = 0
        for ipoint, curpoint in enumerate(X):
            representative_id,tmp = self.DistanceRepresentativestoAPoints_LSH(ipoint, curpoint,labels_matrix,representatives)
            #representative_id,tmp = self.DistanceRepresentativestoAPoints(representatives, curpoint)
            cost += tmp
            if membship[representative_id, ipoint]: continue
            old_clust = labels_matrix[ipoint]
            self.MovePoint(ipoint, old_clust,representative_id, representatives_count, representatives_sum,membship,curpoint,labels_matrix  )
            move +=1
        #Check empty clusters
        move  += self.CheckEmptyClusters(representatives, X,representatives_sum, representatives_count,membship,labels_matrix)
        return cost ,move, 0

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
    def DoCluster(self):
        self.name = "LSHkRepresentatives"
        #print("LSHkRepresentatives start clustering")
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
            self.random_state = check_random_state(None)
            membship = np.zeros((k, n), dtype=np.uint8)
            labels_matrix = np.empty(n, dtype=np.uint16)
            
            for i in range(n): labels_matrix[i] = 65535
            representatives_count = [[[0 for i in range(D[j])] for j in range(d)]for ki in range(k)]
            representatives_sum = [0 for ki in range(k)]
            representatives = [[[random.uniform(0,1) for i in range(D[j])] for j in range(d)] for ki in range(k)]
            #Init first cluster
            self.InitClusters(representatives,representatives_sum,representatives_count)
            last_cost = float('inf')

            self.UpdateLabelsInit(representatives, X,representatives_sum, representatives_count,membship,labels_matrix)
            self.UpdateRepresentatives(representatives,representatives_sum,representatives_count ) ;
            for i in range(self.n_iter):
                self.iter =  i
                cost , move, count_empty = self.UpdateLabels(representatives, X,representatives_sum, representatives_count,membship,labels_matrix)
                self.UpdateRepresentatives(representatives,representatives_sum,representatives_count ) ;
                if last_cost == cost and move==0:
                    last_cost = self.UpdateLabelsLast(representatives, X,representatives_sum, representatives_count,membship,labels_matrix)
                    #print("last_cost=", last_cost, "last_cost2=",last_cost2)
                    break 
                last_cost = cost
                #print ("Iter: ", i , " Cost:", cost, "Move:", move)
            labels = self.GetLabels(membship)
            all_costs.append(cost)
            all_labels.append(labels)
            
        best = np.argmin(all_costs)
        labels = all_labels[best]
        self.time_score = (timeit.default_timer() - start_time)/ self.n_init
        self.labels = labels
        print("LSH time:", self.time_lsh ,"Score: ", all_costs[best] , " Time:", self.time_score)
        return self.labels
        # Update representives

def Test_Simple():
    DB = tulti.LoadSynthesisData(n=8096,d=50,k=50,sigma_rate=0.1); 
    MeasureManager.CURRENT_DATASET = DB['name']
    MeasureManager.CURRENT_MEASURE = 'DILCA'

    print("\n\n############## LSHkRepresentatives ###################")

    lshkrepresentatives = LSHkRepresentatives(DB['DB'],DB['labels_'] )
    lshkrepresentatives.SetupMeasure(MeasureManager.CURRENT_MEASURE)
    lshkrepresentatives.SetupLSH(hbits=25,measure=MeasureManager.CURRENT_MEASURE)
    lshkrepresentatives.DoCluster()
    lshkrepresentatives.CalcScore()

    print("\n\n############## KMODES ###################")
    kmodes = kModes(DB['DB'],DB['labels_'] )
    kmodes.SetupMeasure(MeasureManager.CURRENT_MEASURE)
    kmodes.DoCluster()
    kmodes.CalcScore()

    print("\n\n############## kRepresentatives ###################")
    kmodes = kRepresentatives(DB['DB'],DB['labels_'] )
    kmodes.SetupMeasure(MeasureManager.CURRENT_MEASURE)
    kmodes.DoCluster()
    kmodes.CalcScore()
def Test(): 
    DB = tulti.LoadSynthesisData(n=TDef.n,d=TDef.d,k=TDef.k,sigma_rate=0.1); 
    MeasureManager.CURRENT_DATASET = DB['name']
    MeasureManager.CURRENT_MEASURE = 'DILCA'
    print("\n\n############## kModes ###################")
    alo = LSHkRepresentatives(DB['DB'],DB['labels_'] )
    alo.SetupMeasure(MeasureManager.CURRENT_MEASURE)
    #alo.SetupLSH(measure=MeasureManager.CURRENT_MEASURE)
    alo.DoCluster()
    alo.CalcScore()

def TestDatasets(): 
    for dbname in MeasureManager.DATASET_LIST:
        DB = tulti.LoadRealData(dbname)
        MeasureManager.CURRENT_DATASET = dbname
        MeasureManager.CURRENT_MEASURE = 'DILCA'
        print("\n\n############## LSHkRepresentatives ###################")
        alo = LSHkRepresentatives(DB['DB'],DB['labels_'],dbname=MeasureManager.CURRENT_DATASET )
        alo.SetupMeasure(MeasureManager.CURRENT_MEASURE)
        alo.SetupLSH(measure=MeasureManager.CURRENT_MEASURE)
        alo.DoCluster()
        alo.CalcScore()

if __name__ == "__main__":
    TDef.InitParameters(sys.argv)
    if len(sys.argv)==1:
        main()
    elif TDef.test_type == 'datasets':
        TestDatasets()
    else:
        Test()