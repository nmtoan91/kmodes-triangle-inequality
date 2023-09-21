import sys
sys.path.append("..")
sys.path.append(".")
import pandas as pd
from CategoricalDataClusteringFramework.Measures.Overlap import Overlap
from CategoricalDataClusteringFramework.ClusteringAlgorithms.kModes import kModes

dataPath = 'D:/DATA/CATEGORICAL/SYN/'
dataFile = 'SYN_100000_16_256_8_10.csv'


data = pd.read_csv(dataPath+dataFile, header=None)
X = data.to_numpy()
y = X[:,X.shape[1]-1]
X = X[:,0:X.shape[1]-1]
print( X.shape, y.shape)

measure = Overlap(dataFile)
measure.setUp(X, y)





