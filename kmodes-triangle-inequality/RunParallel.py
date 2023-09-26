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

    alg4.DoCluster(seed)
    alg4.CalcScore()
    return alg4

def RunParallel(n,d,k,np,method,datapath='./DataSample/'):
    
    now = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some integers.')
    args = parser.parse_args()
    args.n = n
    args.d = d
    args.k = k
    args.np = np
    args.datapath = datapath
    args.filename = 'SYN_'+str(n)+'_'+str(d)+'_'+str(k)+'_8_10.csv'
    args.method = method

    args.seed = 41
    

    parameters = [copy.deepcopy(args) for i in range(args.np) ]
    for i in range(len(parameters)):
        parameters[i].seed  = int(random.uniform(0, 200000))

    with Pool( min(args.np,cpu_count())) as p:
        R = p.map(f, parameters)
    return R

    
    
if __name__ == '__main__':
    R = RunParallel(512,10,20,16,'kmodes_ti','./DataSample/')
    print(R)
    print(cpu_count())

