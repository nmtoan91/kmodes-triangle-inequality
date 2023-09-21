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
sys.path.append(os.path.join(os.getcwd(), "./ClusteringAlgorithms"))

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
from LSH import LSH as LSH

#from kmodes.util.dissim import matching_dissim
class MHkModes(ClusteringAlgorithm):
    def SetupLSH(self, hbits=-1,k=-1,measure='DILCA' ):
        start = timeit.default_timer()
        self.lsh = LSH.LSH(self.X,self.y,measure=measure,hbits=hbits)
        self.lsh.DoHash()
        self.lsh.CorrectSingletonBucket()
        #print("SETING UP: ",self.n,self.d,self.k,hbits,'->',self.lsh.hbits, "avg=" + str(self.n/ (2**self.lsh.hbits))  )
        self.time_lsh = timeit.default_timer() - start
        self.AddVariableToPrint("Time_lsh",self.time_lsh)
        return self.time_lsh
    def move_point_cat(self,point, ipoint, to_clust, from_clust, cl_attr_freq,
                   membship, centroids):
        """Move point between clusters, categorical attributes."""
        membship[to_clust, ipoint] = 1
        membship[from_clust, ipoint] = 0
        # Update frequencies of attributes in cluster.
        for iattr, curattr in enumerate(point):
            to_attr_counts = cl_attr_freq[to_clust][iattr]
            from_attr_counts = cl_attr_freq[from_clust][iattr]

            # Increment the attribute count for the new "to" cluster
            to_attr_counts[curattr] += 1

            current_attribute_value_freq = to_attr_counts[curattr]
            current_centroid_value = centroids[to_clust][iattr]
            current_centroid_freq = to_attr_counts[current_centroid_value]
            if current_centroid_freq < current_attribute_value_freq:
                # We have incremented this value to the new mode. Update the centroid.
                centroids[to_clust][iattr] = curattr

            # Decrement the attribute count for the old "from" cluster
            from_attr_counts[curattr] -= 1

            old_centroid_value = centroids[from_clust][iattr]
            if old_centroid_value == curattr:
                # We have just removed a count from the old centroid value. We need to
                # recalculate the centroid as it may no longer be the maximum
                centroids[from_clust][iattr] = get_max_value_key(from_attr_counts)

        return cl_attr_freq, membship, centroids

    def DistanceCentroidstoAPoints_LSH(self,item_id, point,centroids):
        myset = self.preferList[self.lsh.hash_values[item_id]]
        dist_min = 1000000000
        dist_index =-1
        for i in myset:
            dist = self.measure.calculate(centroids[i], point)
            if dist_min > dist:
                dist_min = dist 
                dist_index = i
        return dist_index, dist_min 

    def _k_modes_iter(self,X, centroids, cl_attr_freq, membship, dissim, random_state):
        """Single iteration of k-modes clustering algorithm"""
        moves = 0
        for ipoint, curpoint in enumerate(X):
            #clust = np.argmin(matching_dissim(centroids, curpoint, X=X, membship=membship))
            #clust = np.argmin(self.ComputeDistances(centroids, curpoint))
            clust,a = self.DistanceCentroidstoAPoints_LSH(ipoint,curpoint, centroids )

            if membship[clust, ipoint]:
                # Point is already in its right place.
                continue

            # Move point, and update old/new cluster frequencies and centroids.
            moves += 1
            old_clust = np.argwhere(membship[:, ipoint])[0][0]

            cl_attr_freq, membship, centroids = self.move_point_cat(
                curpoint, ipoint, clust, old_clust, cl_attr_freq, membship, centroids
            )

            # In case of an empty cluster, reinitialize with a random point
            # from the largest cluster.
            if not membship[old_clust, :].any():
                from_clust = membship.sum(axis=1).argmax()
                choices = [ii for ii, ch in enumerate(membship[from_clust, :]) if ch]
                rindx = random_state.choice(choices)

                cl_attr_freq, membship, centroids = self.move_point_cat(
                    X[rindx], rindx, old_clust, from_clust, cl_attr_freq, membship, centroids
                )

        return centroids, moves
    def test(self):
        print("a234 " + str(self.k))
    def DoCluster(self):
        self.name = "MHkModes"
        X = self.X
        n_clusters = self.k
        #print("Do kModes")
        n_points = self.X.shape[0];
        results = []
        random_state=None
        start_time = timeit.default_timer()
        random_state = check_random_state(random_state)
        for init_no in range(self.n_init):
            if TDef.verbose >=1: print ('MHkModes Init ' + str(init_no))
            n_attrs = X.shape[1]
            centroids = np.empty((n_clusters, n_attrs), dtype='object')
            # determine frequencies of attributes
            for iattr in range(n_attrs):
                freq = defaultdict(int)
                for curattr in X[:, iattr]:
                    freq[curattr] += 1
                choices = [chc for chc, wght in freq.items() for _ in range(wght)]
                choices = sorted(choices)
                centroids[:, iattr] = random_state.choice(choices, n_clusters)
            for ik in range(n_clusters):
                ndx = np.argsort(self.ComputeDistances(X, centroids[ik]))
                # We want the centroid to be unique, if possible.
                while np.all(X[ndx[0]] == centroids, axis=1).any() and ndx.shape[0] > 1:
                    ndx = np.delete(ndx, 0)
                centroids[ik] = X[ndx[0]]
            membship = np.zeros((n_clusters, n_points), dtype=np.uint8)
            cl_attr_freq = [[defaultdict(int) for _ in range(n_attrs)]
                            for _ in range(n_clusters)]
            for ipoint, curpoint in enumerate(X):
                # Initial assignment to clusters
                clust = np.argmin(self.ComputeDistances(centroids, curpoint))
                #clust = np.argmin(matching_dissim(centroids, curpoint, X=X, membship=membship))

                membship[clust, ipoint] = 1
                # Count attribute values per cluster.
                for iattr, curattr in enumerate(curpoint):
                    cl_attr_freq[clust][iattr][curattr] += 1
            # Perform an initial centroid update.
            for ik in range(n_clusters):
                for iattr in range(n_attrs):
                    if sum(membship[ik]) == 0:
                        # Empty centroid, choose randomly
                        centroids[ik, iattr] = random_state.choice(X[:, iattr])
                    else:
                        centroids[ik, iattr] = get_max_value_key(cl_attr_freq[ik][iattr])
            #print("Init time: ", timeit.default_timer() - start)
            #print("Starting iterations...")

            #MHkModes Init
            self.preferList= defaultdict(set)
            for ipoint, curpoint in enumerate(X):
                representative_id = self.ComputeDistances(centroids, curpoint)
                representative_id = np.argmin(representative_id)
                #membship[representative_id, ipoint] = 1
                self.preferList[self.lsh.hash_values[ipoint]].add(representative_id)
            #End MHkModes Init
            verbose = False
            itr = 0
            labels = None
            converged = False
            _, cost = self._labels_cost(X, centroids, None, membship)
            epoch_costs = [cost]
            while itr <= self.n_iter and not converged:
                start_time_iter = timeit.default_timer()
                itr += 1
                self.iter = itr
                centroids, moves = self._k_modes_iter(
                    X,
                    centroids,
                    cl_attr_freq,
                    membship,
                    None,
                    random_state
                )
                # All points seen in this iteration
                labels, ncost = self._labels_cost(X, centroids, None, membship)
                converged = (moves == 0) or (ncost >= cost)
                epoch_costs.append(ncost)
                cost = ncost
                if TDef.verbose >=2: print ('Iter ' + str(itr),"Cost: ", cost,";Move: ", moves, "%.2f"%(timeit.default_timer()-start_time_iter))
                if verbose:
                    print("Run {}, iteration: {}/{}, moves: {}, cost: {}"
                            .format(init_no + 1, itr, max_iter, moves, cost))
            re = centroids, labels, cost, itr, epoch_costs
            results.append(re)

        all_centroids, all_labels, all_costs, all_n_iters, all_epoch_costs = zip(*results)
        best = np.argmin(all_costs)
        if self.n_init > 1 and verbose:
            print("Best run was number {}".format(best + 1))
        centroids = all_centroids[best]
        self.labels = all_labels[best]
        self.time_score = (timeit.default_timer() - start_time)/ self.n_init
        print("Score: ", all_costs[best] , " Time:", self.time_score)
        return self.labels
        #return self.CalcScore(False)


def main():
    MeasureManager.CURRENT_DATASET = 'soybean_small.csv'
    MeasureManager.CURRENT_MEASURE = 'DILCA'
    DB = tulti.LoadRealData(MeasureManager.CURRENT_DATASET)
    kmodes = MHkModes(DB['DB'],DB['labels_'] ,dbname=MeasureManager.CURRENT_DATASET)
    kmodes.SetupMeasure(MeasureManager.CURRENT_MEASURE)
    kmodes.SetupLSH(measure=MeasureManager.CURRENT_MEASURE)
    labels = kmodes.DoCluster()
    kmodes.CalcScore()


    #km = KModes(n_clusters=3, init='Huang', n_init=1, verbose=1)
    #clusters = km.fit_predict(DB['DB'])
    #km_labels = km.labels_
     

def Test():  
    MeasureManager.CURRENT_DATASET = 'balance-scale.csv'
    MeasureManager.CURRENT_MEASURE = 'Overlap'
    if TDef.data!='': MeasureManager.CURRENT_DATASET = TDef.data
    if TDef.measure!='': MeasureManager.CURRENT_MEASURE = TDef.measure
    if TDef.test_type == 'syn':
        DB = tulti.LoadSynthesisData(TDef.n,  TDef.d, TDef.k)
        MeasureManager.CURRENT_DATASET= DB['name']
    else:
        DB = tulti.LoadRealData(MeasureManager.CURRENT_DATASET)
    print("\n\n############## MHkModes ###################")
    alo = MHkModes(DB['DB'],DB['labels_'],dbname=MeasureManager.CURRENT_DATASET ,k=TDef.k )
    alo.SetupMeasure(MeasureManager.CURRENT_MEASURE)
    alo.SetupLSH(measure=MeasureManager.CURRENT_MEASURE)
    alo.DoCluster()
    alo.CalcScore()

def TestDatasets(): 
    for dbname in MeasureManager.DATASET_LIST:
        DB = tulti.LoadRealData(dbname)
        MeasureManager.CURRENT_DATASET = dbname
        MeasureManager.CURRENT_MEASURE = 'DILCA'
        print("\n\n############## MHkModes ###################")
        alo = MHkModes(DB['DB'],DB['labels_'],dbname=MeasureManager.CURRENT_DATASET )
        alo.SetupMeasure(MeasureManager.CURRENT_MEASURE)
        alo.SetupLSH(measure=MeasureManager.CURRENT_MEASURE)
        alo.DoCluster()
        alo.CalcScore()

if __name__ == "__main__":
    TDef.InitParameters(sys.argv)
    if TDef.test_type == 'datasets':
        TestDatasets()
    else:
        Test()