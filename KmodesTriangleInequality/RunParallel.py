import sys
sys.path.append("..")
sys.path.append(".")
import datetime
import pandas as pd
from CategoricalDataClusteringFramework.Measures.Overlap import Overlap
from CategoricalDataClusteringFramework.ClusteringAlgorithms.kModes import kModes
from kModesBaseline import kModesBaseline
from kModesTriangleInequality import kModesTriangleInequality
from kModesTriangleInequality_IEEEAccess import kModesTriangleInequality_IEEEAccess
from multiprocessing import Pool
from multiprocessing import Process
from multiprocessing import cpu_count
import random
from toansttlib import TCSVResult
import argparse
import copy
from CategoricalDataClusteringFramework.Dataset.GenerateDataset import *


def f(args):
    seed = args.seed#int(random.uniform(0, 200000))
    dataPath = args.datapath
    dataFile = args.filename
    data = pd.read_csv(dataPath+dataFile, header=None)
    X = data.to_numpy()
    y = X[:,X.shape[1]-1]
    X = X[:,0:X.shape[1]-1]

    if args.method == 'kmodes_ti':
        alg4 = kModesTriangleInequality_IEEEAccess(X,y,dbname = dataFile)
    else :alg4 = kModesBaseline(X,y,dbname = dataFile)

    alg4.DoCluster(seed, args.init_clusters)
    alg4.CalcScore()
    return alg4

def RunParallel(n,d,k,range_,sigma,np,method,datapath='./DataSample/', init_clustersS = None):
    
    now = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some integers.')
    args = parser.parse_args()
    args.n = n
    args.d = d
    args.k = k
    args.np = np
    args.datapath = datapath
    args.filename = GetFileNameOnly(n,d,k,range_,sigma)
    args.method = method

    args.seed = 41
    

    parameters = [copy.deepcopy(args) for i in range(args.np) ]
    for i in range(len(parameters)):
        parameters[i].seed  = int(random.uniform(0, 200000))
        if init_clustersS != None:
            parameters[i].init_clusters = init_clustersS[i]
        else : parameters[i].init_clusters = None

    with Pool( min(args.np,cpu_count())) as p:
        R = p.map(f, parameters)
    return R

    
    
if __name__ == '__main__':
    n = 5000
    d = 256
    k = 32
    range_ = 8
    sigma =0.1
    ncores = 64
    dataPath = GetDataPath()
    
    GenerateDataset(n,d,k,range_,sigma)

    print('kmodes baseline')
    R1 = RunParallel(n,d,k,range_,sigma,ncores,'kmodes',dataPath)
    print('\n\n\n\n\n\n\n\nkmodes_ti')
    R2 = RunParallel(n,d,k,range_,sigma,ncores,'kmodes_ti',dataPath)
    


    #print(R)
    print(cpu_count())

