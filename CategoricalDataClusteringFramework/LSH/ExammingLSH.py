import os
import os.path
import sys
from sys import platform
sys.path.append(os.path.join(os.getcwd(), "Measures"))
import numpy as np
import pandas as pd
import TUlti as tulti
from collections import defaultdict
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array
import timeit
from Measures import *
import random
from SimpleHashing import SimpleHashing
import TUlti as tulti
from collections import defaultdict
import statistics 
from LSH import LSH
from TUlti import MyTable



def TestTurningMeasure():
    mytable = MyTable()
    mytable.SetupSheetAndColum(["Score",'Time'],MeasureManager.MEASURE_LIST)
    for line in MeasureManager.DATASET_LIST:
        MeasureManager.CURRENT_DATASET = line
        DB = tulti.ReadAndNormalizeBD(MeasureManager.CURRENT_DATASET)
        for class_name in MeasureManager.MEASURE_LIST:
            MeasureManager.CURRENT_MEASURE = class_name
            hashing = LSH(DB['DB'],DB['labels_'],measure=MeasureManager.CURRENT_MEASURE)
            hashing.test()
            start = timeit.default_timer()
            hashing.DoHash()
            score = hashing.TestHashTable()
            print('Score: ', score)
            mytable.AddValuestoMultipleSheets(class_name,[score,timeit.default_timer() - start ])
        mytable.SaveToExcel("ExammingLSH_mesure",MeasureManager.DATASET_LIST)

def TestTurninghbits():
    mytable = MyTable()
    MeasureManager.CURRENT_MEASURE = "New_Lin"
    #mytable.SetupSheetAndColum(["Score",'Time'],MeasureManager.MEASURE_LIST)
    for line in MeasureManager.DATASET_LIST:
        MeasureManager.CURRENT_DATASET = line
        DB = tulti.ReadAndNormalizeBD(MeasureManager.CURRENT_DATASET)
        for i in range(1,DB['d']):
            hashing = LSH(DB['DB'],DB['labels_'],measure=MeasureManager.CURRENT_MEASURE,hbits=i)
            hashing.test()
            start = timeit.default_timer()
            hashing.DoHash()
            score = hashing.TestHashTable()
            print('Score: ', score)
            mytable.AddValue("Score",MeasureManager.CURRENT_DATASET,score )
            mytable.AddValue("Hashing_time",MeasureManager.CURRENT_DATASET,timeit.default_timer() - start )
        mytable.SaveToExcel("ExammingLSH_hbits",[i for i in range(1,1000)])


def main():
    TestTurninghbits()
if __name__ == "__main__":
    main()
