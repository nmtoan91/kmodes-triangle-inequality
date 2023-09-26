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
import random
from toansttlib import TCSVResult
import argparse


def f(args):
    seed = int(random.uniform(0, 200000))
    dataPath = args.datapath
    dataFile = args.filename
    data = pd.read_csv(dataPath+dataFile, header=None)
    X = data.to_numpy()
    y = X[:,X.shape[1]-1]
    X = X[:,0:X.shape[1]-1]
    alg4 = kModesTriangleInequality_IEEEAccess(X,y,dbname = dataFile)
    alg4.DoCluster(seed)
    alg4.CalcScore()
    return alg4

def main():
    now = datetime.datetime.now()

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--filename',default='SYN_512_10_20_8_10.csv')
    parser.add_argument('--datapath',default='./DataSample/')
    parser.add_argument('--method',default='kmodes')
    parser.add_argument('--n',default=2)
    args = parser.parse_args()

    testname = 'r'+str(now.year-2023)+str(now.month)+str(now.day)+str(now.hour)+str(now.second)+ '_'  + '_' + args.method +args.filename

    table = TCSVResult(testname)
    with Pool( min(args.n,32)) as p:
        R = p.map(f, [args for i in range(args.n) ])
        print(R)
        table.AddVariableToPrint('dataset',args.filename )
        table.AddVariableToPrint('n',R[0].n )
        table.AddVariableToPrint('d',R[0].n )
        table.AddVariableToPrint('k',R[0].n )

    table.WriteResultToCSV()

    
    print('toansts')
if __name__ == '__main__':
    main()

