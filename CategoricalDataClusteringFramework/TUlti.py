#toanstt's lib for preprocessing data
import os
import os.path
import sys
sys.path.append("..")
sys.path.append(".")
from sys import platform
import numpy as np
import time
import pandas as pd
import itertools;
import timeit
import math
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import  contingency_matrix
from sklearn.metrics.cluster import  v_measure_score
from datetime import date
from CategoricalDataClusteringFramework.Dataset.GenerateDataset import GenerateDataset

from CategoricalDataClusteringFramework.Measures.MeasureManager import MeasureManager

import copy
from numpy import asarray
from numpy import savetxt

def nCr(n,r):
    f = math.factorial
    return f(n) // f(r) // f(n-r)
def GetDatasetFolder():
    db_path = "F:\\DATASET\\ANN_CATEGORICAL\\"
    if platform == "linux" or platform == "linux2":
        db_path = '/home/s1620409/DATASET/ANN_CATEGORICAL/'
    return db_path
def GetDatasetFolderSyn():
    db_path = "F:\\DATASET\\ANN_CATEGORICAL\\SYN\\"
    if platform == "linux" or platform == "linux2":
        db_path = '/home/s1620409/DATASET/ANN_CATEGORICAL/SYN/'
    return db_path
def NormalizeDB(db,is_num=False):
    if is_num== False:
        max_=[];
        DB=np.array([]);
        dicts_by_attributes ={};
        for i in range(db.shape[1]):
            sdb =  db[:,i];
            _uniques={};
            tmp = db[:,i];
            _uniques = np.unique(tmp);
            a = [np.where(sdb[i]==_uniques)[0][0]  for i in range(sdb.shape[0]) ]
            max_.append(max(a))
            if len(DB)==0:
                DB=np.array([a]).T
            else:
                DB = np.hstack((DB,np.array([a]).T))
    else:
        DB = db
        d = DB.shape[1]-1
        max_ = [max(DB[:,i]) for i in range(d)]

    DB_ = DB[:,0:DB.shape[1]-1];
    labels_ = DB[:,DB.shape[1]-1];
    return {'DB':DB_,'labels_':labels_,'max':max_, 'n':DB_.shape[0], 'd':DB_.shape[1]}

def LoadSynthesisData(n,d,k,range_=8, sigma_rate=0.1):
    if k <=0: k = 8
    db_path = GetDatasetFolderSyn()
    name = "SYN_"+str(n) + "_" + str(d) + "_" + str(k)+ "_" + str(range_) +"_"+ str(int(sigma_rate*100))+".csv"
    filename = db_path + name 
    if os.path.isfile(filename)==False:
        GenerateDataset(n,d,k,range_, sigma_rate)
    db = pd.read_csv(filename,header=None)._values;
    return {'DB':db[:,0:d],'labels_':db[:,d], 'n':n, 'd':d, 'name': name}
def LoadRealData(dbname, is_num=True):
    db_path = GetDatasetFolder()
    filename = db_path + dbname 
    filename_num = filename.replace(".csv","") + "_num.csv"

    if os.path.isfile(filename)==False:
        print("Data: " + filename , "does not exist")
    if not is_num:
        db = pd.read_csv(filename,header=None).replace(np.nan, 'nan', regex=True)._values;
    else :
        db = pd.read_csv(filename_num,header=None).replace(np.nan, 'nan', regex=True)._values;

    DB = NormalizeDB(db,is_num)
    return DB
def CheckCLusteringPurityByPermutations(labels_,km_labels_):
    if min(labels_) != min(km_labels_) or  max(labels_) != max(km_labels_) or min(labels_)!=0:
        print("ERROR: CLUSTER LABELS DONOT MATCH")
        return -1
    score_max=0;
    a = itertools.permutations(min(labels_) + range(1+max(labels_)-min(labels_)), int(1+max(labels_)-min(labels_)))
    for indices in a:
        km_labels_tmp=[indices[i] for i in km_labels_ ];
        score = sum(km_labels_tmp== labels_)/len(labels_)
        score_max = max(score_max,score)
        #print("check score:", score )
    return score_max
def CheckCLusteringPurityByHeuristic(labels_,km_labels_):
    #start = timeit.default_timer()
    unique_ = np.unique(labels_)
    n_clusters = len(unique_)
    matching_matrix = [-1 for i in range (n_clusters)]
    n = len(labels_)
    n_range = range(n)
    #Computer matching matrix
    count_item =0
    for i in range(n_clusters):
        max_count =-1
        max_index = 0
        for j in range(n_clusters):
            if j in matching_matrix:
                continue
            count = sum([labels_[k] == i and km_labels_[k] == j for k in n_range ])
            if count > max_count:
                max_count = count
                max_index = j
        matching_matrix[i] = max_index
        count_item =count_item+ max_count
    #Compute score
    score2 = count_item/len(labels_)
    return score2
def AcPrRc(A,B):
    #B = np.array([0,0,0,0,0,1,2,2,   0,1,1,1,1,   1,2,2,2 ])
    #B = np.array([1,1,1,1,1,2,0,0,   1,2,2,2,2,   2,0,0,0 ])

    k = len(np.unique(A))
    n= len(A)
    MAP =[0,1,2]

    clustersA = [np.where(A==i)[0] for i in range(k)]
    clustersB = [np.where(B==i)[0] for i in range(k)]

    TP_FP=0
    for i in range(k):
        if len(clustersB[i]) > 1: 
            TP_FP+=  nCr(len(clustersB[i]),2)
    TP=0
    for i in range(k):
        for j in range(k):
            num = sum(A[clustersB[i]]==j)
            if num > 1:
                TP+= nCr(num,2)

    FP = TP_FP-TP


    TN_FN = int(n*(n-1)/2) - TP_FP
    FN=0

    for i in range(k):
        for j in range(k):
            num_ = sum(A[clustersB[i]]==j)
            sum_=0
            for i2 in range(i+1,k):
                sum_ += sum(A[clustersB[i2]]==j)
            FN+=num_*sum_
    TN = TN_FN - FN
    #print('TP=',TP,'TP_FP=',TP_FP ,'TN_FN=',TN_FN,'FN=',FN)
    PR = TP/(TP+FP)
    RC = TP/(TP+FN)
    AC = (TP+TN)/(TP+TN+FP+FN)
    #print('PR=',PR,'RC=',RC,'AC=',AC )
    return AC,PR,RC
def CheckClusteringNMI(l1,l2):
    n = len(l1)
    u1 = np.unique(l1)
    u2 = np.unique(l2)
    n1 = len(u1)
    n2 = len(u2)
    I = 0
    numerator = 0
    for i1 in range(n1):
        for i2 in range(n2):
            set1 = np.where(l1==i1)[0]
            set2 = np.where(l2==i2)[0]
            len1 = len(set1)
            len2 = len(set2)
            interset_len = len(np.intersect1d(set1,set2))
            if len1 != 0 and len2 !=0 and interset_len!=0:
                numerator = numerator+interset_len*math.log(n*interset_len/(len1*len2))
    denumerator=0
    for i2 in range(n2):
        set2 = np.where(l2==i2)[0]
        len2 = len(set2)
        sum=0
        for i1 in range(n1):
            set1 = np.where(l1==i1)[0]
            len1 = len(set1)
            sum = sum + len1*math.log(len1/n)
        denumerator = denumerator+ len2*math.log(len2/n)*sum
    if denumerator==0:
        return -1
    #score = normalized_mutual_info_score(l1,l2)
    #score_me = numerator/math.sqrt(denumerator)
    return numerator/math.sqrt(denumerator)
def CheckClusteringARI(l1,l2):
    return adjusted_rand_score(l1,l2)
def ReadAndNormalizeBD(db_name):
    #MeasureManager.CURRENT_DATASET = db_name
    db_path = "F:\\DATASET\\ANN_CATEGORICAL\\" + db_name
    if platform == "linux" or platform == "linux2":
        db_path = '/home/s1620409/DATASET/ANN_CATEGORICAL/' + db_name
    db = pd.read_csv(db_path,header=None).replace(np.nan, 'nan', regex=True)._values;
    DB = NormalizeDB(db)
    return DB

class MyTable:
    def __init__(self):
        self.df_lists = {}
    def __pad_dict_list(self,dict_list, padel):
        lmax = 0
        for lname in dict_list.keys():
            lmax = max(lmax, len(dict_list[lname]))
        for lname in dict_list.keys():
            ll = len(dict_list[lname])
            if  ll < lmax:
                dict_list[lname] += [padel] * (lmax - ll)
        return dict_list
    def AddValue(self,sheetname, colname, value ):
        if sheetname in self.df_lists:
            #self.AddValuetoColum(self.df_lists[sheetname],colname,value);
            if colname in self.df_lists[sheetname]:
                self.df_lists[sheetname][colname].append(value);
            else :
                self.df_lists[sheetname][colname] = [value]
        else :
            self.df_lists[sheetname] ={}
            if colname in self.df_lists[sheetname]:
                self.df_lists[sheetname][colname].append(value);
            else :
                self.df_lists[sheetname][colname] = [value]
    def SaveToExcel(self, filename,rownames = None):
        df_lists2 = copy.deepcopy(self.df_lists )
        sheet_names = [dict for dict in df_lists2 ] 
        col_names = [list(df_lists2[key].keys()) for key in sheet_names ] 

        
        #Padding empty values
        for name in sheet_names:
            list1 = df_lists2[name]
            self.__pad_dict_list(list1,'')
        #end padding
        df = [pd.DataFrame(df_lists2[i], columns =  col_names[sheet_names.index(i)]) for i in sheet_names]
        if rownames != None:
            name_rules = {i: rownames[i] for i in range(len(rownames)) }
            for i in range(len(df_lists2)):
                    df[i] = df[i].rename(index=name_rules)
        with pd.ExcelWriter(filename + '_'+str(date.today())+ '.xlsx') as writer: 
            for i in range(len(df_lists2)):
                df[i].to_excel(writer,sheet_name=sheet_names[i])

    def SaveToExcelFolder(self,folder, filename,rownames = None):
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = folder + "/" + filename
        self.SaveToExcel(filename,rownames)

    def SetupSheetAndColum(self, sheetnames, colnames):
        self.df_lists = {}
        for i in sheetnames:
            self.df_lists[i] = {}
            for j in colnames:
                self.df_lists[i][j] = []
    def AddValuestoMultipleSheets(self,colname, values ):
        sheet_names = [dict for dict in self.df_lists ]
        if(len(sheet_names) != len(values)):
            print("ERROR: Number of values does not match number of sheets")
        for i in range(len(values)):
            self.AddValue(sheet_names[i],colname,values[i])
    def AddValuestoMultipleSheetsByColIndex(self,colindex, values ):
        sheets = self.df_lists.values()
        colname = list(list(sheets)[0].keys())[colindex]
        self.AddValuestoMultipleSheets(colname, values)

def main():

    DB = LoadSynthesisData(100,10,10);
def ShowDatabaseInfor():
    m = MyTable()
    for dbname in MeasureManager.DATASET_LIST:
        m.AddValue('datasets', 'FName', dbname )
        #dbname= dbname.replace("_c","")
        m.AddValue('datasets', 'Name', dbname.replace(".csv","").capitalize() )
        DB = LoadRealData(dbname,False) 
        savetxt(GetDatasetFolder()+dbname.replace(".csv","")+"_num.csv", np.hstack((DB['DB'],np.array([DB['labels_']]).T)), fmt='%i', delimiter=',')

        m.AddValue('datasets', '#Items', DB['n'] )
        m.AddValue('datasets', '#Attributes', DB['d'] )
        m.AddValue('datasets', '#Classes', len(np.unique(DB['labels_'])))

        m.AddValue('info', 'Name', dbname.replace(".csv","").capitalize())
        for i in range(100):
            val ='_'
            if(i< DB['d']):
                val = DB['max'][i]+1
            m.AddValue('info', 'A'+str(i), val )

    m.SaveToExcelFolder('RESULT','DatasetInfo')
if __name__ == "__main__":
    ShowDatabaseInfor()