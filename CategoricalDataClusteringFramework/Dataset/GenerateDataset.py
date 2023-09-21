import os
import os.path
import sys
from sys import platform
sys.path.append(os.path.join(os.getcwd(), "Measures"))
import random
import copy
import csv
import numpy as np
def GenerateDataset(n,d,k,range_=8, sigma_rate=0.1):
    print("Generateing dataset n=",n,"d=",d,"k=",k,"range_=", range_,"sigmarate=", sigma_rate) 
    db_path = "D:\\DATASET\\ANN_CATEGORICAL\\SYN\\"
    if platform == "linux" or platform == "linux2":
        db_path = '/home/s1620409/DATASET/ANN_CATEGORICAL/SYN/'
    while True:
        random.seed(None)
        clusters = [[random.randint(0, range_-1) for di in range(d) ] for i in range(k) ]
        data = [copy.deepcopy(clusters[i%k]) for i in range(n)]
        labels = [i%k for i in range(n)]
        sigma = sigma_rate*range_
        for x in data:
            for i in range(d):
                x[i] = (x[i]+int(random.gauss(0,sigma))+range_)%range_
        a = np.array(data)
        for i in range(d):
            unique= np.unique(a[:,i])
            unique_index=[]
            for j in range(range_):
                index = np.where(unique==j)[0]
                if len(index) >0:
                    unique_index.append(index[0])
                else: unique_index.append(-1)
            for j in range(n):
                data[j][i] =  unique_index[data[j][i]]
                asd=1
        #Save
        DATA = [ data[i] + [labels[i]] for i in range(n)] 
        filename = db_path + "SYN_"+str(n) + "_" + str(d) + "_" + str(k)+ "_" + str(range_) +"_"+ str(int(sigma_rate*100)) +".csv"
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(DATA)
        #print("Try OK")
        break
def main():
    GenerateDataset(100,10,10);
    #n = 65536
    #for i in range(1,20):    
     #   k = 2**i
      #  GenerateDataset(n,8,k,8);

if __name__ == "__main__":
    main()