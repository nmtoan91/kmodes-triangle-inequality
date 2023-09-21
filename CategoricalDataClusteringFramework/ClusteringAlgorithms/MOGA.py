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
import math
import random
import matplotlib.pyplot as plt
import copy
class MOGA(ClusteringAlgorithm):
    #Find max as default
    def Overlap(self,x,y):
        n = len(x)
        sum =0
        for i in range(n):
            if x[i] != y[i]: sum +=1
        return sum
    def function1(self,kis):
        dist_matrix =  scipy.spatial.distance.cdist(self.X, kis, self.Overlap)
        minz = np.min(dist_matrix,1)
        return 1/sum(minz)
    def function2(self,kis):
        #return 0
        value =0
        for i in range(self.k-1):
            for j in range(i+1,self.k):
                value += self.Overlap(kis[i],kis[j])
        return value

    def index_of(self,a,list):
        for i in range(0,len(list)):
            if list[i] == a:
                return i
        return -1
    def sort_by_values(self,list1, values):
        sorted_list = []
        while(len(sorted_list)!=len(list1)):
            if self.index_of(min(values),values) in list1:
                sorted_list.append(self.index_of(min(values),values))
            values[self.index_of(min(values),values)] = math.inf
        return sorted_list
    def fast_non_dominated_sort(self,values1, values2):
        S=[[] for i in range(0,len(values1))]
        front = [[]]
        n=[0 for i in range(0,len(values1))]
        rank = [0 for i in range(0, len(values1))]

        for p in range(0,len(values1)):
            S[p]=[]
            n[p]=0
            for q in range(0, len(values1)):
                if (values1[p] > values1[q] and values2[p] > values2[q]) or (values1[p] >= values1[q] and values2[p] > values2[q]) or (values1[p] > values1[q] and values2[p] >= values2[q]):
                    if q not in S[p]:
                        S[p].append(q)
                elif (values1[q] > values1[p] and values2[q] > values2[p]) or (values1[q] >= values1[p] and values2[q] > values2[p]) or (values1[q] > values1[p] and values2[q] >= values2[p]):
                    n[p] = n[p] + 1
            if n[p]==0:
                rank[p] = 0
                if p not in front[0]:
                    front[0].append(p)

        i = 0
        while(front[i] != []):
            Q=[]
            for p in front[i]:
                for q in S[p]:
                    n[q] =n[q] - 1
                    if( n[q]==0):
                        rank[q]=i+1
                        if q not in Q:
                            Q.append(q)
            i = i+1
            front.append(Q)

        del front[len(front)-1]
        return front

    #Function to calculate crowding distance
    def crowding_distance(self,values1, values2, front):
        distance = [0 for i in range(0,len(front))]
        sorted1 = self.sort_by_values(front, values1[:])
        sorted2 = self.sort_by_values(front, values2[:])
        distance[0] = 4444444444444444
        distance[len(front) - 1] = 4444444444444444
        for k in range(1,len(front)-1):
            distance[k] = distance[k]+ (values1[sorted1[k+1]] - values2[sorted1[k-1]])/(max(values1)-min(values1))
        for k in range(1,len(front)-1):
            distance[k] = distance[k]+ (values1[sorted2[k+1]] - values2[sorted2[k-1]])/(max(values2)-min(values2))
        return distance

    #Function to carry out the crossover
    def crossover(self,a,b):
        vec = copy.deepcopy(a)
        for ki in range(self.k):
            for di in range(self.d):
                if random.random()>0.5:
                    vec[ki][di] = self.mutation(a[ki][di],di)
                else:
                    vec[ki][di] = self.mutation(b[ki][di] ,di)
        return vec
    #Function to carry out the mutation operator
    def mutation(self,solution,di):
        if random.random() <0.5:
            solution = random.randint(0, self.D[di]-1)
        return solution


    def DoCluster(self):
        self.name = "MOGA"
        start_time = timeit.default_timer()
        self.D  = [len(np.unique(self.X[:,i])) for i in range(self.d) ]
        pop_size = 40
        max_gen = 100
        solution=[[[random.randint(0, self.D[j]-1)  for j in range(self.d)] for ki in range(self.k) ] for i in range(0,pop_size)]
        gen_no=0
        while(gen_no<max_gen):
            function1_values = [self.function1(solution[i])for i in range(0,pop_size)]
            function2_values = [self.function2(solution[i])for i in range(0,pop_size)]
            non_dominated_sorted_solution = self.fast_non_dominated_sort(function1_values[:],function2_values[:])
            print("The best front for Generation number ",gen_no, " is",function1_values[0],function2_values[0] )
            crowding_distance_values=[]
            for i in range(0,len(non_dominated_sorted_solution)):
                crowding_distance_values.append(self.crowding_distance(function1_values[:],function2_values[:],non_dominated_sorted_solution[i][:]))
            solution2 = solution[:]
            #Generating offsprings
            while(len(solution2)!=2*pop_size):
                a1 = random.randint(0,pop_size-1)
                b1 = random.randint(0,pop_size-1)
                solution2.append(self.crossover(solution[a1],solution[b1]))
            function1_values2 = [self.function1(solution2[i])for i in range(0,2*pop_size)]
            function2_values2 = [self.function2(solution2[i])for i in range(0,2*pop_size)]
            non_dominated_sorted_solution2 = self.fast_non_dominated_sort(function1_values2[:],function2_values2[:])
            crowding_distance_values2=[]
            for i in range(0,len(non_dominated_sorted_solution2)):
                crowding_distance_values2.append(self.crowding_distance(function1_values2[:],function2_values2[:],non_dominated_sorted_solution2[i][:]))
            new_solution= []
            for i in range(0,len(non_dominated_sorted_solution2)):
                non_dominated_sorted_solution2_1 = [self.index_of(non_dominated_sorted_solution2[i][j],non_dominated_sorted_solution2[i] ) for j in range(0,len(non_dominated_sorted_solution2[i]))]
                front22 = self.sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])
                front = [non_dominated_sorted_solution2[i][front22[j]] for j in range(0,len(non_dominated_sorted_solution2[i]))]
                front.reverse()
                for value in front:
                    new_solution.append(value)
                    if(len(new_solution)==pop_size):
                        break
                if (len(new_solution) == pop_size):
                    break
            solution = [solution2[i] for i in new_solution]
            gen_no = gen_no + 1
        bestmode = solution[0]
        dist_matrix =  scipy.spatial.distance.cdist(self.X, bestmode, self.Overlap)
        self.labels= np.argmin(dist_matrix,1)
        self.time_score = (timeit.default_timer() - start_time)/ self.n_init
        self.iter = max_gen
        print(function1_values)
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
    print("\n\n############## MOGA ###################")
    alo = MOGA(DB['DB'],DB['labels_'] ,dbname=MeasureManager.CURRENT_DATASET ,k=TDef.k)
    alo.SetupMeasure(MeasureManager.CURRENT_MEASURE)
    #alo.SetupLSH(measure=MeasureManager.CURRENT_MEASURE)
    alo.DoCluster()
    alo.CalcScore()