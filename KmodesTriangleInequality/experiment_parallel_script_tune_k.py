import sys
sys.path.append("..")
sys.path.append(".")
from CategoricalDataClusteringFramework.Dataset.GenerateDataset import *
from CategoricalDataClusteringFramework.Measures.Overlap import Overlap
from kModesBaseline import kModesBaseline
from RunParallel import RunParallel
from toansttlib import TCSVResult
import datetime
import pandas as pd

if __name__ == '__main__':
    
    sigma = 0.1
    ni=n = 4096
    di=d = 64
    ki=k = 64
    range_ = 16
    npr = 4

    kk = [2**i for i in  range(3,12)]
    datapath = GetDataPath()
    

    now = datetime.datetime.now()
    testname = 'r_tune_k_'+str(now.year-2023)+str(now.month)+str(now.day)+str(now.hour)+str(now.second)+ '_'  + GetFileNameOnly(n,d,k)

    for ki in kk:#[512]:
        #Generate datasets and measures
        GenerateDataset(ni,di,ki,range_,sigma)
        dataPath = GetDataPath()
        dataFile = GetFileNameOnly(ni,di,ki,range_,sigma)
        data = pd.read_csv(dataPath+dataFile, header=None)
        X = data.to_numpy()
        y = X[:,X.shape[1]-1]
        X = X[:,0:X.shape[1]-1]
        measure = Overlap(dataFile)
        measure.setUp(X,y)

        #Do parallel clustering

        DB = RunParallel(ni,di,ki,range_,sigma,npr, 'kmodes', datapath)
        init_clustersS = [i.init_clusters for i in DB]
        DN = RunParallel(ni,di,ki,range_, sigma,npr, 'kmodes_ti', datapath,init_clustersS )
        

        table = TCSVResult(testname)
        table.AddVariableToPrint('dataset',GetFileNameOnly(ni,di,ki,range_,sigma))
        table.AddVariableToPrint('n',ni)
        table.AddVariableToPrint('d',di)
        table.AddVariableToPrint('k',ki)
        table.AddVariableToPrint('npr',npr)

        AMI1 = [i.AMI_score for i in DB];table.AddVariableToPrint('AMI1', np.mean(AMI1)); table.AddVariableToPrint('AMI1_', np.std(AMI1))
        ARI1 = [i.ARI_score for i in DB] ; table.AddVariableToPrint('ARI1', np.mean(ARI1)); table.AddVariableToPrint('ARI1_', np.std(ARI1))
        Ac1 = [i.Ac_score for i in DB]; table.AddVariableToPrint('Ac1', np.mean(Ac1)); table.AddVariableToPrint('Ac1_', np.std(Ac1))
        HOMO1 = [i.HOMO_score for i in DB]; table.AddVariableToPrint('HOMO1', np.mean(HOMO1)); table.AddVariableToPrint('HOMO1_', np.std(HOMO1))
        Pr1 = [i.Pr_score for i in DB]; table.AddVariableToPrint('Pr1', np.mean(Pr1)); table.AddVariableToPrint('Pr1_', np.std(Pr1))
        Rc1 = [i.Rc_score for i in DB]; table.AddVariableToPrint('Rc1', np.mean(Rc1)); table.AddVariableToPrint('Rc1_', np.std(Rc1))
        Sil1 = [i.SILHOUETTE_score for i in DB]; table.AddVariableToPrint('Sil1', np.mean(Sil1)); table.AddVariableToPrint('Sil1_', np.std(Sil1))
        Purity1 = [i.purity_score for i in DB]; table.AddVariableToPrint('Purity1', np.mean(Purity1)); table.AddVariableToPrint('Purity1_', np.std(Purity1))
        Time1 = [i.time_score for i in DB]; table.AddVariableToPrint('Time1', np.mean(Time1)); table.AddVariableToPrint('Time1_', np.std(Time1))


        AMI2 = [i.AMI_score for i in DN];table.AddVariableToPrint('AMI2', np.mean(AMI2)); table.AddVariableToPrint('AMI2_', np.std(AMI2))
        ARI2 = [i.ARI_score for i in DN] ; table.AddVariableToPrint('ARI2', np.mean(ARI2)); table.AddVariableToPrint('ARI2_', np.std(ARI2))
        Ac2 = [i.Ac_score for i in DN]; table.AddVariableToPrint('Ac2', np.mean(Ac2)); table.AddVariableToPrint('Ac2_', np.std(Ac2))
        HOMO2 = [i.HOMO_score for i in DN]; table.AddVariableToPrint('HOMO2', np.mean(HOMO2)); table.AddVariableToPrint('HOMO2_', np.std(HOMO2))
        Pr2 = [i.Pr_score for i in DN]; table.AddVariableToPrint('Pr2', np.mean(Pr2)); table.AddVariableToPrint('Pr2_', np.std(Pr2))
        Rc2 = [i.Rc_score for i in DN]; table.AddVariableToPrint('Rc2', np.mean(Rc2)); table.AddVariableToPrint('Rc2_', np.std(Rc2))
        Sil2 = [i.SILHOUETTE_score for i in DN]; table.AddVariableToPrint('Sil2', np.mean(Sil2)); table.AddVariableToPrint('Sil2_', np.std(Sil2))
        Purity2 = [i.purity_score for i in DN]; table.AddVariableToPrint('Purity2', np.mean(Purity2)); table.AddVariableToPrint('Purity2_', np.std(Purity2))
        Time2 = [i.time_score for i in DN]; table.AddVariableToPrint('Time2', np.mean(Time2)); table.AddVariableToPrint('Time2_', np.std(Time2))


        table.WriteResultToCSV()

