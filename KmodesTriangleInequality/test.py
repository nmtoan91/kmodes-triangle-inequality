import sys
sys.path.append("..")
sys.path.append(".")
import pandas as pd
from CategoricalDataClusteringFramework.Measures.Overlap import Overlap
from CategoricalDataClusteringFramework.ClusteringAlgorithms.kModes import kModes
from kModesBaseline import kModesBaseline
from kModesTriangleInequality import kModesTriangleInequality
from kModesTriangleInequality_IEEEAccess import kModesTriangleInequality_IEEEAccess
from CategoricalDataClusteringFramework.Dataset.GenerateDataset import *


n = 512
d = 10
k = 20
range_ = 8
sigma =0.1
GenerateDataset(n,d,k,range_,sigma)

dataPath = GetDataPath()
dataFile = GetFileNameOnly(n,d,k,range_,sigma)


data = pd.read_csv(dataPath+dataFile, header=None)
X = data.to_numpy()
y = X[:,X.shape[1]-1]
X = X[:,0:X.shape[1]-1]
print( X.shape, y.shape)


# print('\nkModes lib')
# alg = kModes(X,y,dbname = dataFile)
# alg.DoCluster()
# alg.CalcScore()


print('\nkModesBaseline')
alg2 = kModesBaseline(X,y,dbname = dataFile)
alg2.DoCluster()
alg2.CalcScore()

# print('\nkModesTriangleInequality')
# alg3 = kModesTriangleInequality(X,y,dbname = dataFile)
# alg3.DoCluster()
# alg3.CalcScore()


print('\nkModesTriangleInequality_IEEEAccess')
alg4 = kModesTriangleInequality_IEEEAccess(X,y,dbname = dataFile)
alg4.DoCluster()
alg4.CalcScore()




