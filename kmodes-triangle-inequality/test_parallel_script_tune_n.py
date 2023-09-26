import sys
sys.path.append("..")
sys.path.append(".")
from CategoricalDataClusteringFramework.Dataset.GenerateDataset import *
from RunParallel import RunParallel



if __name__ == '__main__':
    n = 512
    d = 10
    k = 20
    npr = 4

    ns = [2**i for i in  range(9,20)]
    datapath = GetDataPath()
    GenerateDataset(n,d,k)

    for ni in [512]:
        GenerateDataset(ni,d,k)
        DB = RunParallel(ni,d,k,npr, 'kmodes_ti', datapath)
        DN = RunParallel(ni,d,k,npr, 'kmodes', datapath )






    # for ni in ns:
    #     GenerateDataset(ni,16,16)
    #     RunParallel(n,d,k,32, 'kmodes_ti', datapath )
