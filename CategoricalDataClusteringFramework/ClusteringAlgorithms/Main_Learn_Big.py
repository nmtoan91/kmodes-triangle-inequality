from kRepresentatives import kRepresentatives
import os
import os.path
import sys
from sys import platform
import numpy as np
sys.path.append(os.path.join(os.getcwd(), "Measures"))
sys.path.append(os.path.join(os.getcwd(), "LSH"))
sys.path.append(os.path.join(os.getcwd(), "../"))
sys.path.append(os.path.join(os.getcwd(), "../Dataset"))
sys.path.append(os.path.join(os.getcwd(), "../Measures"))
sys.path.append(os.path.join(os.getcwd(), "../LSH"))
sys.path.append(os.path.join(os.getcwd(), "./ClusteringAlgorithms"))
import TDef
import TUlti as tulti
from Measures import *

if __name__ == "__main__":
    TDef.InitParameters(sys.argv)
    kRepresentatives
    MeasureManager.CURRENT_DATASET = 'zoo_c.csv' 
    MeasureManager.CURRENT_MEASURE = 'Overlap'
    if TDef.data!='': MeasureManager.CURRENT_DATASET = TDef.data
    if TDef.measure!='': MeasureManager.CURRENT_MEASURE = TDef.measure
    if TDef.test_type == 'syn':
        DB = tulti.LoadSynthesisData(TDef.n,  TDef.d, TDef.k)
        MeasureManager.CURRENT_DATASET= DB['name']
    else:
        DB = tulti.LoadRealData(MeasureManager.CURRENT_DATASET)
    algo = kRepresentatives(DB['DB'],DB['labels_'],k=TDef.k ,dbname=MeasureManager.CURRENT_DATASET)
    algo.SetupMeasure(MeasureManager.CURRENT_MEASURE)
    algo.DoCluster()

    labels = algo.labels
    print(labels)
    db_new = np.column_stack((DB['DB'],labels))
    filename_new = MeasureManager.CURRENT_DATASET.replace(".csv","") + "_"+ str(TDef.k) + "_groundtruth_num.csv"
    np.savetxt(filename_new, db_new,fmt='%i', delimiter=",")