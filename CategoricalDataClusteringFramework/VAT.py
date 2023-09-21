import TUlti as tulti
from Measures import *
from sklearn import datasets
from pyclustertend import vat
from pyclustertend import compute_ordered_dissimilarity_matrix
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import numpy as np
import scipy
MeasureManager.CURRENT_DATASET = 'soybean_small.csv'
def Overlap(x,y):
        n = len(x)
        sum =0
        for i in range(n):
            if x[i] != y[i]: sum +=1
        return sum

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 6}
#plt.rcParams.update({"text.usetex": True,"font.family": "serif","font.sans-serif": ["Computer Modern"]})
plt.rc('font', **font)
list = MeasureManager.DATASET_LIST
list = ['soybean_small.csv','audiology_c.csv','zoo_c.csv','tae_c.csv','hayes-roth_c.csv','dermatology_c.csv','soybean.csv','connect.csv','chess.csv','breast.csv','car.csv','mushroom.csv','splice.csv','vote.csv','lymph.csv','lung.csv']
nrow = 4
ncol = 4
fig, axs = plt.subplots(nrow, ncol, constrained_layout=True)

ii=0
#list = ['soybean_small.csv','soybean_small.csv','soybean_small.csv','soybean_small.csv']
#Fuzzy
#list = ['soybean_small.csv','balance-scale.csv','zoo_c.csv','tae_c.csv','post-operative.csv','hayes-roth_c.csv','dermatology_c.csv','soybean.csv','connect.csv','chess.csv','breast.csv','car.csv','mushroom.csv','vote.csv','lymph.csv','lung.csv']
#list = ['soybean_small.csv','soybean_small.csv','soybean_small.csv','soybean_small.csv','soybean_small.csv','soybean_small.csv','soybean_small.csv','soybean_small.csv','soybean_small.csv','soybean_small.csv','soybean_small.csv','soybean_small.csv','soybean_small.csv','soybean_small.csv','soybean_small.csv','soybean_small.csv']
for d in list:
    print(d)
    row = ii//ncol
    col = ii%ncol
    ii +=1
    ax=axs[row,col]
    DB = tulti.LoadRealData(d)
    X=DB['DB']
    #ordered_dissimilarity_matrix = np.zeros((len(X), len(X)))
    #for i in range(len(X)-1):
    #    for j in range(i+1,len(X)):
    #        ordered_dissimilarity_matrix[j][i]=ordered_dissimilarity_matrix[i][j] = Overlap(X[i],X[j])

    #ordered_dissimilarity_matrix = compute_ordered_dissimilarity_matrix(DB['DB'])
    #
    ordered_dissimilarity_matrix = scipy.spatial.distance.cdist(X,X,Overlap)
    ax.imshow(ordered_dissimilarity_matrix, cmap='gray', vmin=0, vmax=np.max(ordered_dissimilarity_matrix))
    #ax.set_title(d.replace('.csv','').replace('_c',''), fontsize=6)
    #ax.set_xlabel('xlabel', fontsize=6)
    for aa in ax.xaxis.get_ticklabels():
        aa.set_y(+.08)
    ax.text(.5,1.04,d.replace('.csv','').replace('_c',''),horizontalalignment='center',transform=ax.transAxes, fontsize=6)

    #for aa in ax.yaxis.get_ticklabels():
     #   aa.set_x(+.08)
    if row == nrow -1 and col == ncol-1: break
plt.show() 

   
