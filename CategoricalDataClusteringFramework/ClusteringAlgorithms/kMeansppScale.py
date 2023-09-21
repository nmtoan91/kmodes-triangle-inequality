import os
import os.path
import sys
from sys import platform
sys.path.append(os.path.join(os.getcwd(), "Measures"))
sys.path.append(os.path.join(os.getcwd(), "LSH"))
sys.path.append(os.path.join(os.getcwd(), "../"))
sys.path.append(os.path.join(os.getcwd(), "../Dataset"))
sys.path.append(os.path.join(os.getcwd(), "../Measures"))
sys.path.append(os.path.join(os.getcwd(), "../LSH"))
import numpy as np
import pandas as pd
#from kmodes_lib import KModes
import TUlti as tulti
from collections import defaultdict
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array
import timeit
from kmodes.util import get_max_value_key, encode_features, get_unique_rows, \
    decode_centroids, pandas_to_numpy
from Measures import *
from ClusteringAlgorithm import ClusteringAlgorithm
from kmodes_lib import KModes
import TDef
import random
#from sklearn.cluster import KMeans
#from kmodes.util.dissim import matching_dissim



#FROM ANH TAI
def sample_new(data, distribution, l):
    """ Sample new centers

    Parameters:
       data         n*d
       distribution n*1
       l            the number of new centers to sample
    Returns:        new centers
    """
    return data[np.random.choice(range(len(distribution)), l, p=distribution), :]
def cost(dist):
    """ Calculate the cost of data with respect to the current centroids
    Parameters:
       dist     distance matrix between data and current centroids

    Returns:    the normalized constant in the distribution
    """
    return np.sum(np.min(dist, axis=1))
def distribution(dist, cost):
    """ Calculate the distribution to sample new centers
    Parameters:
       dist       distance matrix between data and current centroids
       cost       the cost of data with respect to the current centroids
    Returns:      distribution
    """
    return np.min(dist, axis=1) / cost
def distance(data, centroids):
    """ Calculate the distance from each data point to each center
    Parameters:
       data   n*d
       center k*d

    Returns:
       distence n*k
    """
    ## calculate distence between each point to the centroids
    dist = np.sum((data[:, np.newaxis, :] - centroids) ** 2, axis=2)
    return dist
def get_weight(dist, centroids):
    min_dist = np.zeros(dist.shape)
    min_dist[range(dist.shape[0]), np.argmin(dist, axis=1)] = 1
    count = np.array([np.count_nonzero(min_dist[:, i]) for i in range(centroids.shape[0])])
    return count / np.sum(count)

def KMeans(data, k, centroids, max_iter=10000):
    """ Apply the KMeans clustering algorithm

    Parameters:
      data                        ndarrays data
      k                           number of cluster
      centroids                   initial centroids

    Returns:
      "Iteration before Coverge"  time used to converge
      "Centroids"                 the final centroids finded by KMeans
      "Labels"                    the cluster of each data
    """

    n = data.shape[0]
    iterations = 0
    iter=0
    while iterations < max_iter:
        iter = iterations
        dist = distance(data, centroids)

        ## give cluster label to each point
        cluster_label = np.argmin(dist, axis=1)

        ## calculate new centroids
        newCentroids = np.zeros(centroids.shape)
        for j in range(0, k):
            if sum(cluster_label == j) == 0:
                newCentroids[j] = centroids[j]
            else:
                newCentroids[j] = np.mean(data[cluster_label == j, :], axis=0)

        ## Check if it is converged
        if np.array_equal(centroids, newCentroids):
            print("Converge! after:", iterations, "iterations")
            break

        centroids = newCentroids
        iterations += 1

    return ({"Iteration before Coverge": iterations,
             "Centroids": centroids,
             "Labels": cluster_label,"iter":iter})
def ScalableKMeansPlusPlus(data, k, l, iter=5):
    """ Apply the KMeans|| clustering algorithm

    Parameters:
      data     ndarrays data
      k        number of cluster
      l        number of point sampled in each iteration

    Returns:   the final centroids finded by KMeans||

    """

    centroids = data[np.random.choice(range(data.shape[0]), 1), :]

    for i in range(iter):
        # Get the distance between data and centroids
        dist = distance(data, centroids)

        # Calculate the cost of data with respect to the centroids
        norm_const = cost(dist)

        # Calculate the distribution for sampling l new centers
        p = distribution(dist, norm_const)

        # Sample the l new centers and append them to the original ones
        centroids = np.r_[centroids, sample_new(data, p, l)]

    ## reduce k*l to k using KMeans++
    dist = distance(data, centroids)
    weights = get_weight(dist, centroids)

    return centroids[np.random.choice(len(weights), k, replace=False, p=weights), :]

def run(argv):
    ## Simulate data
    k = 20
    n = 10000
    d = 15

    ## simulate k centers from 15-dimensional spherical Gaussian distribution
    mean = np.hstack(np.zeros((d, 1)))
    cov = np.diag(np.array([1, 10, 100] * 5))
    centers = np.random.multivariate_normal(mean, cov, k)

    ## Simulate n data
    for i in range(k):
        mean = centers[i]
        if i == 0:
            data = np.random.multivariate_normal(mean, np.diag(np.ones(d)), int(n / k + n % k))
            trueLabels = np.repeat(i, int(n / k + n % k))
        else:
            data = np.append(data, np.random.multivariate_normal(mean, np.diag(np.ones(d)), int(n / k)), axis=0)
            trueLabels = np.append(trueLabels, np.repeat(i, int(n / k)))

    data_v = pd.DataFrame(data[:, 0:3])
    data_v['trueLabels'] = trueLabels
    # import seaborn as sns
    # sns.set_context("notebook", font_scale=1.3)
    # sns.pairplot(data_v, hue="trueLabels")
    import matplotlib.pyplot as plt
    # plt.show()
    l = 10
    centroids_initial = ScalableKMeansPlusPlus(data, 20, l)
    output_spp = KMeans(data, k, centroids_initial)
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, k)]

    centroids1 = output_spp["Centroids"]
    labels1 = output_spp["Labels"]

    for i, color in enumerate(colors, start=1):
        plt.scatter(data[labels1 == i, :][:, 0], data[labels1 == i, :][:, 1], color=color)

    for j in range(k):
        plt.scatter(centroids1[j, 0], centroids1[j, 1], color='w', marker='x')
    plt.show()
#END FROM ANH TAI


class kMeansppScale(ClusteringAlgorithm):
    def test(self):
        print("a234 " + str(self.k))
    def DoCluster(self):
        self.labels = np.array([random.randint(0,self.k-1) for i in range(self.n)])
        self.name = 'kMeansppScale'
        start_time = timeit.default_timer()
        l=10
        try:
            centroids_initial = ScalableKMeansPlusPlus(self.X, self.k, l)
            output_spp = KMeans(self.X, self.k, centroids_initial)
            self.labels = output_spp["Labels"]
            self.iter = output_spp["iter"]
        except:
            self.iter = -100
            self.NMI=-2
            print("ERROR")
        
        self.time_score = (timeit.default_timer() - start_time)/ self.n_init
        return self.labels

def Test(): 
    MeasureManager.CURRENT_DATASET = 'soybean_small.csv'
    MeasureManager.CURRENT_MEASURE = 'Overlap'
    if TDef.data!='': MeasureManager.CURRENT_DATASET = TDef.data
    if TDef.measure!='': MeasureManager.CURRENT_MEASURE = TDef.measure
    if TDef.test_type == 'syn':
        DB = tulti.LoadSynthesisData(TDef.n,  TDef.d, TDef.k)
        MeasureManager.CURRENT_DATASET= DB['name']
    else:
        DB = tulti.LoadRealData(MeasureManager.CURRENT_DATASET)
    print("\n\n############## kMeansppScale ###################")
    alo = kMeansppScale(DB['DB'],DB['labels_'] ,dbname=MeasureManager.CURRENT_DATASET ,k=TDef.k)
    alo.SetupMeasure(MeasureManager.CURRENT_MEASURE)
    alo.DoCluster()
    alo.CalcScore()

def TestDatasets(): 
    for dbname in MeasureManager.DATASET_LIST:
        DB = tulti.LoadRealData(dbname)
        MeasureManager.CURRENT_DATASET = dbname
        MeasureManager.CURRENT_MEASURE = 'Overlap'
        print("\n\n############## kMeansppScale ###################")
        alo = kMeansppScale(DB['DB'],DB['labels_'],dbname=MeasureManager.CURRENT_DATASET )
        alo.SetupMeasure(MeasureManager.CURRENT_MEASURE)
        #alo.SetupLSH(measure=MeasureManager.CURRENT_MEASURE)
        alo.DoCluster()
        alo.CalcScore()

if __name__ == "__main__":
    TDef.InitParameters(sys.argv)
    if TDef.test_type == 'datasets':
        TestDatasets()
    else:
        Test()