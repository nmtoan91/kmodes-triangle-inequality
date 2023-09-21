import os
import os.path
import sys
sys.path.append("..")
sys.path.append(".")
from sys import platform
import numpy as np
from sklearn.utils.validation import check_array
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import silhouette_score
from CategoricalDataClusteringFramework import TUlti as tulti
import timeit
import CategoricalDataClusteringFramework.TDef
import csv
from csv import writer
import random

class ClusteringAlgorithm:
    ALGORITHM_LIST = ['kModes','kRepresentatives']
    def __init__(self, X, y,n_init=-1,k=-1,n_iter=-1,dbname='dpname'):
        self.measurename = 'None'
        self.dicts = [];self.dicts2 = []
        self.iter=-1
        self.dbname = dbname
        self.time_lsh=-1
        self.X = X
        self.y = y
        self.n = len(self.X)
        self.d = len(self.X[0])
        self.k = k if k > 0 else len(np.unique(y))
        self.n_init = n_init
        self.n_iter = n_iter
        if n_init == -1: self.n_init = TDef.n_init 
        if n_iter ==-1 : self.n_iter = TDef.n_iter 
        self.scorebest = -2
    def SetupMeasure(self, classname):
        self.measurename = classname
        module = __import__(classname, globals(), locals(), ['object'])
        class_ = getattr(module, classname)
        self.measure = class_()
        self.measure.setUp(self.X, self.y)
    def Overlap(self,x,y):
        n = len(x)
        sum =0
        for i in range(n):
            if x[i] != y[i]: sum +=1
        return sum
    def DoCluster(self):
        print("Do something")
        return -1
    def _labels_cost(self,X, centroids, dissim, membship=None):
        X = check_array(X)
        n_points = X.shape[0]
        cost = 0.
        labels = np.empty(n_points, dtype=np.uint16)
        for ipoint, curpoint in enumerate(X):
            diss = self.ComputeDistances(centroids, curpoint)
            clust = np.argmin(diss)
            labels[ipoint] = clust
            cost += diss[clust]
        return labels, cost
    def ComputeDistances(self, X, mode):
        return [ self.measure.calculate(i, mode ) for i in X ]
    def CalcScore(self, verbose=True):
        starttime = timeit.default_timer()
        s="";
        if self.n*self.k <= 8000000: 
            self.purity_score = tulti.CheckCLusteringPurityByHeuristic(self.y, self.labels)
        else: self.purity_score =-2
        s+= str(timeit.default_timer()-starttime)+"|";  starttime = timeit.default_timer()
        self.NMI_score = normalized_mutual_info_score(self.y,self.labels) #tulti.CheckClusteringNMI(self.y, self.labels)
        s+= str(timeit.default_timer()-starttime)+"|";  starttime = timeit.default_timer()
        self.ARI_score = adjusted_rand_score(self.labels,self.y) # tulti.CheckClusteringARI(self.y, self.labels)
        s+= str(timeit.default_timer()-starttime)+"|";  starttime = timeit.default_timer()
        self.AMI_score = adjusted_mutual_info_score(self.labels,self.y)
        s+= str(timeit.default_timer()-starttime)+"|";  starttime = timeit.default_timer()
        self.HOMO_score = homogeneity_score(self.labels,self.y)
        s+= str(timeit.default_timer()-starttime)+"|";  starttime = timeit.default_timer()
        if self.n*self.k <= 800000:
            try: 
                self.SILHOUETTE_score = silhouette_score(self.X, self.labels, metric= self.Overlap)
            except:
                self.SILHOUETTE_score=-1
        else: self.SILHOUETTE_score=-2

        s+= str(timeit.default_timer()-starttime)+"|";  starttime = timeit.default_timer()
        if self.n*self.k <= 8000000: 
            self.Ac_score, self.Pr_score,self.Rc_score =  tulti.AcPrRc(self.y, self.labels)
        else: self.Ac_score =  self.Pr_score = self.Rc_score = -2  

        s+= str(timeit.default_timer()-starttime)+"|";  starttime = timeit.default_timer()
        self.AddVariableToPrint("Scoringtime",s )
        self.WriteResultToCSV()
        if verbose: print("Purity:", "%.2f" % self.purity_score,"NMI:", "%.2f" %self.NMI_score,"ARI:", "%.2f" %self.ARI_score,"Sil: ", "%.2f" %self.SILHOUETTE_score,"Acc:", "%.2f" %self.Ac_score,
                          "Recall:", "%.2f" %self.Rc_score,"Precision:", "%.2f" %self.Pr_score)
        return (self.purity_score,self.NMI_score,self.ARI_score,self.AMI_score,self.HOMO_score,self.SILHOUETTE_score,self.time_score, self.time_lsh,
                self.Ac_score, self.Pr_score, self.Rc_score)

    def append_list_as_row(self,file_name, list_of_elem):
        with open(file_name, 'a+', newline='') as write_obj:
            csv_writer = writer(write_obj)
            csv_writer.writerow(list_of_elem)
    def AddVariableToPrint(self,name,val):
        self.dicts2.append((name,val ))

    def WriteResultToCSV(self,file=''):
        if not os.path.exists(TDef.folder):
            os.makedirs(TDef.folder)
        if file=='':
            file = TDef.folder+ '/' + self.name + TDef.fname + ".csv" 
        
        self.dbname = self.dbname.replace("_c","").replace(".csv","").capitalize()
        self.dicts.append(('dbname',self.dbname ))
        self.dicts.append(('n',self.n ))
        self.dicts.append(('d',self.d ))
        self.dicts.append(('k',self.k ))
        self.dicts.append(('range','-1' ))
        self.dicts.append(('sigma_ratio',-1 ))
        self.dicts.append(('Measure',  self.measurename))
        self.dicts.append(('n_init',self.n_init ))
        self.dicts.append(('n_iter',self.n_iter ))
        self.dicts.append(('iter',self.iter ))
        self.dicts.append(('Purity',self.purity_score ))
        self.dicts.append(('NMI',self.NMI_score ))
        self.dicts.append(('ARI',self.ARI_score ))
        self.dicts.append(('AMI',self.AMI_score ))
        self.dicts.append(('Homogeneity',self.HOMO_score ))
        self.dicts.append(('Silhouette',self.SILHOUETTE_score ))
        self.dicts.append(('Accuracy',self.Ac_score ))
        self.dicts.append(('Precision',self.Pr_score ))
        self.dicts.append(('Recall',self.Rc_score ))
        self.dicts.append(('Time',self.time_score ))
        self.dicts.append(('LSH_time',self.time_lsh ))
        self.dicts.append(('Score',self.scorebest ))
        
        dicts = self.dicts+self.dicts2;
        try:
            if os.path.isfile(file)==False:
                colnames = [i[0] for i in dicts]
                self.append_list_as_row(file,colnames)
            vals = [i[1] for i in dicts]
            self.append_list_as_row(file,vals)
        except Exception  as ex:
            #self.exe(file,ex)
            print('Cannot write to file ', file ,'', ex);
            self.WriteResultToCSV(file + str(random.randint(0,1000000)) + '.csv')

    def AddValuesToMyTable(self, mytable, is_first=False,dbname=''):
        if is_first:
            mytable.AddValue("Purity", 'Dataset' ,dbname )
            mytable.AddValue("NMI", 'Dataset',dbname )
            mytable.AddValue("ARI", 'Dataset',dbname )
            mytable.AddValue("AMI", 'Dataset',dbname )
            mytable.AddValue("Homogeneity", 'Dataset',dbname )
            mytable.AddValue("Silhouette", 'Dataset',dbname )
            mytable.AddValue("iter", 'Dataset', dbname)
            mytable.AddValue("Precision", 'Dataset',dbname )
            mytable.AddValue("Recall", 'Dataset',dbname )
            mytable.AddValue("Accuracy", 'Dataset',dbname )
            mytable.AddValue("LSH_time", 'Dataset',dbname )
            mytable.AddValue("Time", 'Dataset',dbname )
            mytable.AddValue("Time", 'Dataset',dbname )
            

        mytable.AddValue("Purity", self.measurename ,self.purity_score )
        mytable.AddValue("NMI", self.measurename,self.NMI_score )
        mytable.AddValue("ARI", self.measurename,self.ARI_score )
        mytable.AddValue("AMI", self.measurename,self.AMI_score )
        mytable.AddValue("Homogeneity", self.measurename,self.HOMO_score )
        mytable.AddValue("Silhouette", self.measurename,self.SILHOUETTE_score )
        mytable.AddValue("iter", self.measurename, self.iter )
        mytable.AddValue("Precision", self.measurename,self.Pr_score )
        mytable.AddValue("Recall", self.measurename,self.Rc_score )
        mytable.AddValue("Accuracy", self.measurename,self.Ac_score )
        mytable.AddValue("LSH_time", self.measurename,self.time_lsh )
        mytable.AddValue("Time", self.measurename,self.time_score )
