# -*- coding: utf-8 -*-
'''Noted by Tai Dinh
This file is used for the k-representatives algorithm
'''
from __future__ import division
import os
import os.path
import sys
from sys import platform
sys.path.append(os.path.join(os.getcwd(), "Measures"))
sys.path.append(os.path.join(os.getcwd(), "LSH"))
sys.path.append(os.path.join(os.getcwd(), "../"))
sys.path.append(os.path.join(os.getcwd(), "../../"))
sys.path.append(os.path.join(os.getcwd(), "../Dataset"))
sys.path.append(os.path.join(os.getcwd(), "../Measures"))
sys.path.append(os.path.join(os.getcwd(), "../LSH"))
from collections import defaultdict
import numpy as np
from scipy import *
import TDef
from Measures import *
import TUlti as tulti
from ClusteringAlgorithm import ClusteringAlgorithm
import timeit
import numpy as np
def get_max_value_key(dic):
    '''Fast method to get key for maximum value in dict'''
    v = list(dic.values())
    k = list(dic.keys())
    return k[v.index(max(v))]

'''
Hàm này tính dissimilarity giữa 2 thuộc tính x và y tại vị trí iattr (d)
sử dụng bảng tính xác suất global_attr_freq. Trong đó global_axttr_freq[i][x]
sẽ là xác suất của thuộc tính tại vị trí i mang giá trị là x trên toàn bộ tập
mẫu samples.

Công thức tính sử dụng là
dis(x, y) = 1 - 2 * log(P{x, y}) / (log(P{x}) + log(P{y}))
'''


def attr_dissim(x, y):
    '''
    Dissimilarity between 2 categorical attributes x and y at the attribute iattr, i.e
    dis(x, y) = 1 - 2 * log(P{x, y}) / (log(P{x}) + log(P{y}))
    '''
    if x == y:
        return 0
    else:
        return 1
'''
Hàm này tính dissimilarity giữa 1 centroid (vector trung tâm) và vector a
'''


def vector_matching_dissim(centroid, a):
    '''Get distance between a centroid and a'''

    '''
    Giá trị ic ở bên dưới là chỉ số các thuộc tính (1..D)
    curc là giá trị của thuộc tính đấy, chính là centroid[ic]
    '''
    distance = 0.
    for ic, curc in enumerate(centroid):
        '''
        keys ở đây là tập các giá trị của thuộc tính tại vị trí ic
        Khoảng cách distance chính là tổng các dissimilarity giữa
        mỗi key (1 giá trị trong tập keys) với thuộc tính tại vị trí ic của a
        '''
        keys = curc.keys()
        for key in keys:
            distance += curc[key] * attr_dissim(key, a[ic])
    return distance


'''
Hàm này tính khoảng cách giữa các vectors trung tâm (còn gọi là centroids)
và 1 vector a, sử dụng bảng tính xác suất global_attr_freq. Trong đó global_axttr_freq[i][x]
sẽ là xác suất của thuộc tính tại vị trí i mang giá trị là x trên toàn bộ tập
mẫu samples.
'''


def vectors_matching_dissim(vectors, a):
    '''Get nearest vector in vectors to a'''
    '''
    Ban đầu gán giá trị nhỏ nhất là vô cùng lớn
    Với mỗi vector trung tâm, tìm khoảng cách (dissimilarity) giữa nó với a
    bằng hàm vector_matching_dissim. Nếu dissim mới tìm được nhỏ hơn giá trị
    hiện có thì cập nhật lại.

    Cuối cùng là trả về giá trị dissim và min_clust (id của cụm gần nhất)
    '''
    min = np.Inf
    min_clust = -1
    for clust in range(len(vectors)):
        distance = vector_matching_dissim(vectors[clust], a)
        if distance < min:
            min = distance
            min_clust = clust
    return min_clust, min


'''
Hàm này thực hiện việc chuyển 1 vector point từ cụm này (from_clust) sang
cụm kia (to_clust), ở đây ipoint là chỉ số của vector trong tập samples.

membership[cluster_index, ipoint] = 1 có nghĩa là vector có chỉ số ipoint thuộc
về cụm cluster_index và ngược lại.

cl_attr_freq[cluster_index][iattr][curattr] là xác suất để thuộc tính mang giá
trị curattr tại vị trí iattr trong cụm cluster_index. Nói là xác suất nhưng
thực ra để giá trị số nguyên k (thay vì k/N) với k là số lần xuất hiện của
thuộc tính tại vị trí iattr mang giá trị curattr, N là số phần tử trong cụm.
Bởi nếu lưu k/N mà k, N đều có khả năng thay đổi thì phải tính toán lại khá
phiền toái, lưu mỗi k thì việc +/- dễ dàng hơn.

* Lưu ý global_attr_freq thì lưu xác suất luôn (giá trị dạng k/N) vì chỉ cần
tính toán 1 lần duy nhất và không có thay đổi
'''


def move_point_between_clusters(point, ipoint, to_clust, from_clust,
    cl_attr_freq, membership):
    '''Move point between clusters, categorical attributes'''

    '''Đánh dấu lại ipoint thuộc về cluster mới, xoá bỏ nó khỏi cluster cũ'''
    membership[to_clust, ipoint] = 1
    membership[from_clust, ipoint] = 0
    # Update frequencies of attributes in clusters
    for iattr, curattr in enumerate(point):
        cl_attr_freq[to_clust][iattr][curattr] += 1
        cl_attr_freq[from_clust][iattr][curattr] -= 1
    return cl_attr_freq, membership

def matching_dissim(a, b):
    '''Simple matching dissimilarity function'''
    result=np.sum(a != b, axis=1)
    return result

'''
Khởi tạo phân bố "ngẫu nhiên" các vector trong X vào các cụm
'''


def _init_clusters(X, centroids, n_clusters, nattrs, npoints, verbose):
    # __INIT_CLUSTER__
    if verbose:
        print("Init: Initalizing clusters")
    membership = np.zeros((n_clusters, npoints), dtype='int64')
    # cl_attr_freq is a list of lists with dictionaries that contain the
    # frequencies of values per cluster and attribute.
    cl_attr_freq = [[defaultdict(int) for _ in range(nattrs)]
                    for _ in range(n_clusters)]
    for ipoint, curpoint in enumerate(X):
        # Initial aassignment to clusterss
        clust = np.argmin(matching_dissim(centroids, curpoint))
        membership[clust, ipoint] = 1
        # Count attribute values per cluster
        for iattr, curattr in enumerate(curpoint):
            cl_attr_freq[clust][iattr][curattr] += 1

    # Move random selected point from largest cluster to empty cluster if exists
    for ik in range(n_clusters):
        if np.sum(membership[ik, :]) == 0:
            from_clust = membership.sum(axis=1).argmax()
            choices = \
                [ii for ii, ch in enumerate(membership[from_clust, :]) if ch]
            rindex = np.random.choice(choices)
            # Move random selected point to empty cluster
            cl_attr_freq, membership = move_point_between_clusters(
                X[rindex], rindex, ik, from_clust, cl_attr_freq, membership)

    return cl_attr_freq, membership


'''
Tính giá trị vector trung tâm (centroid cho cụm) tại 1 vị trí thuộc tính

* ldb là lambda
* cl_attr_freq_attr là cl_attr_freq[clust][iattr], có nghĩa là tập các giá trị
của thuộc tính tại vị trí iattr trong cụm clust
* cluster_members là tổng số phần tử của cụm
* global_attr_count là tổng số giá trị của thuộc tính tại vị trí iattr tính trên
toàn bộ tập mẫu X
'''


def cal_centroid_value(cl_attr_freq_attr, cluster_members):
    '''Calculate centroid value at iattr'''
    assert cluster_members >= 1, "Cluster has no member, why?"

    keys = cl_attr_freq_attr.keys()
    vjd = defaultdict(float)
    for odl in keys:
        vjd[odl] = (1.0 * cl_attr_freq_attr[odl] / cluster_members)
    return vjd


'''
Đây là mỗi bước lặp của thuật toán k-representative

Với mỗi vector curpoint với chỉ số ipoint trong tập X, tìm ra cụm có trung tâm
(centroid) gần nó nhất. Nếu chỉ số của cụm này trùng với chỉ số cụm hiện tại của
vector này (tức là membership[clust, ipoint] == 1) thì thực hiện tính cho vector
tiếp theo trong cụm.

Nếu không, thì thực hiện gán lại chỉ số cụm cho vector này, chú thích sẽ viết
tiếp ở bên dưới cho dễ hiểu.
'''


def _k_presentative_iter(X, centroids, cl_attr_freq, membership):
    '''Single iteration of k-representative clustering algorithm'''
    moves = 0
    for ipoint, curpoint in enumerate(X):
        clust, distance = vectors_matching_dissim(centroids, curpoint)
        if membership[clust, ipoint]:
            # Sample is already in its right place
            continue

        # Move point and update old/new cluster frequencies and centroids
        '''
        moves là tổng số bước chuyển vector giữa các cụm
        old_clust là chỉ số cụm cũ của vector curpoint
        '''
        moves += 1
        old_clust = np.argwhere(membership[:, ipoint])[0][0]

        '''
        Chuyển vector chỉ số ipoint từ cụm old_clust sang cụm clust, đồng thời
        tính lại giá trị xác suất của các thuộc tính trong các cụm tương ứng
        '''
        cl_attr_freq, membership = move_point_between_clusters(
            curpoint, ipoint, clust, old_clust, cl_attr_freq, membership)

        # In case of an empty cluster, reinitialize with a random point
        # from the largest cluster.
        '''
        Nếu như sau khi chuyển vector từ cụng old_clust sang cụm mới, mà cụm
        old_clust không còn vector nào, thì lấy 1 vector bất kì từ cụm nhiều
        phần tử nhất gán sang cho nó. Cái này để tránh sau bước này có cụm không
        có vector nào.
        '''
        if np.sum(membership[old_clust, :]) == 0:
            from_clust = membership.sum(axis = 1).argmax()
            choices = \
                [ii for ii, ch in enumerate(membership[from_clust, :]) if ch]
            rindex = np.random.choice(choices)

            cl_attr_freq, membership = move_point_between_clusters(
                X[rindex], rindex, old_clust, from_clust, cl_attr_freq, membership)

        # Update new and old centroids by choosing mode of attribute.
        for iattr in range(len(curpoint)):
            for curc in (clust, old_clust):
                cluster_members = np.sum(membership[curc, :])
                centroids[curc][iattr] = cal_centroid_value(cl_attr_freq[curc][iattr], cluster_members)

    return centroids, moves


'''
Hàm này tính tổng khoảng cách giữa các vectors trong tập mẫu X (samples)
đến các vectors trung tâm tính ra được sau mỗi bước.

labels sẽ là nhãn của các vector trong X. labels[x] = c có nghĩa là vector
có chỉ số x sẽ thuộc về cụm có chỉ số c

cost là tổng dissimilarity
'''


def _labels_cost(X, centroids):
    '''
    Calculate labels and cost function given a matrix of points and
    a list of centroids for the k-modes algorithm.
    '''

    npoints, nattrs = X.shape
    cost = 0.
    labels = np.empty(npoints, dtype = 'int64')
    for ipoint, curpoint in enumerate(X):
        '''
        Với mỗi vector có chỉ số ipoint (giá trị vector là curpoint) trong X
        tìm ra cluster gần nhất với nó bằng hàm vectors_matching_dissim, sau đó
        tính khoảng cách theo công thức trong slide trang 16.
        '''
        clust, diss = vectors_matching_dissim(centroids, curpoint)
        assert clust != -1, "Why there is no cluster for me?"
        labels[ipoint] = clust
        cost += diss

    return labels, cost


'''
Thuật toán k-center (tên k-representative chỉ là do cách đặt) tính toán phân cụm
cho tập dữ liệu X thành n_clusters cụm.

* init là thuật toán khởi tạo dữ liệu ban đầu (ở đây dùng "ngẫu nhiên")
* n_init là số lần chạy thuật toán này với khởi tạo ban đầu khác nhau, mặc định
là 10 lần
* max_iter là số lần chạy thuật toán tối đa
* verbose == 1 là in ra các bước chạy, == 0 thì không in
'''


def k_representative(X, n_clusters, init, n_init, verbose):
    '''k-representative algorithm'''

    X = np.asanyarray(X)
    npoints, nattrs = X.shape
    assert n_clusters < npoints, "More clusters than data points?"

    all_centroids = []
    all_labels = []
    all_costs = []

    if init == 'random':
        seeds = np.random.choice(range(npoints), n_clusters)
        centroids = X[seeds]
    else:
        raise NotImplementedError

    '''
    Tính bảng xác suất các thuộc tính trên toàn cụm sử dụng hàm
    _cal_global_attr_freq()

    _init_clusters() là hàm khởi tạo các cụm 1 cách "ngẫu nhiên"
    NOTE: Chỗ này cũng có thể tiềm ẩn cài đặt sai.
    '''
    cl_attr_freq, membership = _init_clusters(X, centroids, n_clusters, nattrs, npoints, verbose)

    centroids = [[defaultdict(float) for _ in range(nattrs)]
                 for _ in range(n_clusters)]
    # Perform initial centroid update
    lbd = np.zeros(n_clusters, dtype='float')
    for ik in range(n_clusters):
        cluster_members = np.sum(membership[ik, :])
        for iattr in range(nattrs):
            centroids[ik][iattr] = cal_centroid_value(cl_attr_freq[ik][iattr], cluster_members)

    # __ITERATION__
    if verbose:
        print("Starting iterations...")
    converged = False
    cost = np.Inf
    '''
    Bước lặp chính của thuật toán
    1. Tính các vector trung tâm, lambda
    2. Nếu dissimilarity mới (cost) nhỏ hơn thì cập nhật và tiếp tục thực
    hiện thuật toán lần nữa (từ bước 1).
    Nếu lớn hơn thì kết thúc thuật toán.
    '''
    iter = 0
    while not converged:
        time_here = timeit.default_timer()
        iter+=1
        centroids, moves = _k_presentative_iter(X, centroids, cl_attr_freq, membership)
        labels, ncost = _labels_cost(X, centroids)
        converged = (moves == 0)
        cost = ncost
        if verbose>=2: print("Iter ", iter," cost:", ncost , " Move:", moves," Timelapse: ", "%.2f"%(timeit.default_timer() - time_here ))

    # Store result of current runt
    all_centroids.append(centroids)
    all_labels.append(labels)
    all_costs.append(cost)

    '''
    Tìm giá trị best là bước thực hiện cho tổng số dissimilarity là nhỏ nhất
    '''
    best = np.argmin(all_costs)
    if n_init > 1 and verbose:
        print("Best run was number {}, min cost is {}" . format(best + 1, all_costs[best]))

    return all_centroids[best], all_labels[best], all_costs[best]


class KRepresentative(object):

    '''k-representative clustering algorithm for categorical data

    Parameters
    -----------
    K : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.

    max_iter : int, default: 300
        Maximum number of iterations of the k-modes algorithm for a
        single run.

    n_init : int, default: 10
        Number of time the k-modes algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of cost.

    init : 'random'
        'random': choose k observations (rows) at random from data for
        the initial centroids.

    verbose : boolean, optional
        Verbosity mode.

    Attributes
    ----------
    cluster_centroids_ : array, [K, n_features]
        Categories of cluster centroids

    labels_ :
        Labels of each point

    cost_ : float
        Clustering cost, defined as the sum distance of all points to
        their respective cluster centroids.
    '''

    def __init__(self, n_clusters, n_init, init = 'random', max_iter = 1, verbose = 1):
        if verbose:
            print ("Number of clusters: {0}" . format(n_clusters))
            print ("Init type: {0}" . format(init))
            print ("Local loop: {0}" . format(n_init))
            print ("Max iterations: {0}" . format(max_iter))

        if hasattr(init, '__array__'):
            n_clusters = init.shape[0];
            init = np.asarray(init, dtype = np.float64);

        self.n_clusters = n_clusters;
        self.init = init;
        self.n_init = n_init
        self.verbose = verbose
        self.max_iter = max_iter

    def fit(self, X, **kwargs):
        '''Compute k-representative clustering.

        Parameters
        ----------
        X : array-like, shape=[n_samples, n_features]
        '''

        self.cluster_centroids_, self.labels_, self.cost_ = \
            k_representative(X, self.n_clusters, self.init, self.n_init, self.verbose)
        return self

    def fit_predict(self, X, **kwargs):
        '''Compute cluster centroids and predict cluster index for each sample.

        Convenience method; equivalent to calling fit(X) followed by
        predict(X).
        '''
        return self.fit(X, **kwargs).labels_

    def predict(self, X, **kwargs):
        '''Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            New data to predict.

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        '''
        # assert hasattr(self, 'cluster_centroids_'), "Model not yet fitted"
        # return _labels_cost(X, self.cluster_centroids_)[0]
class kRepresentatives_ATai(ClusteringAlgorithm):
    def test(self):
        print("a234 " + str(self.k))
    def DoCluster(self):
        self.name = 'kRepresentatives_ATai'
        start_time = timeit.default_timer()
        anhtai = KRepresentative(n_clusters=self.k, n_init = TDef.n_init, init = 'random', max_iter = TDef.n_iter, verbose = TDef.verbose)
        anhtai.fit(self.X)
        self.labels = anhtai.labels_
        #self.km = KModes(n_clusters=self.k, init='Huang',max_iter=self.n_iter, n_init=self.n_init, verbose=TDef.verbose)
        #self.clusters = self.km.fit_predict(self.X)
        #self.time_score = (timeit.default_timer() - start_time)/ self.n_init
        #self.labels = self.km.labels_
        #self.iter = self.km.n_iter_
        self.time_score = (timeit.default_timer() - start_time)/ self.n_init
        return self.labels
def Test(): 
    #MeasureManager.CURRENT_DATASET = 'tae_c.csv'
    MeasureManager.CURRENT_DATASET = 'lung.csv'
    MeasureManager.CURRENT_MEASURE = 'Overlap'
    if TDef.data!='': MeasureManager.CURRENT_DATASET = TDef.data
    if TDef.measure!='': MeasureManager.CURRENT_MEASURE = TDef.measure
    if TDef.test_type == 'syn':
        DB = tulti.LoadSynthesisData(TDef.n,  TDef.d, TDef.k)
        MeasureManager.CURRENT_DATASET= DB['name']
    else:
        DB = tulti.LoadRealData(MeasureManager.CURRENT_DATASET)
    print("\n\n############## kRepresentatives_ATai ###################")
    lshkrepresentatives = kRepresentatives_ATai(DB['DB'],DB['labels_'] ,dbname=MeasureManager.CURRENT_DATASET ,k=TDef.k)
    #lshkrepresentatives.SetupMeasure(MeasureManager.CURRENT_MEASURE)
    #lshkrepresentatives.SetupLSH(measure=MeasureManager.CURRENT_MEASURE)
    lshkrepresentatives.DoCluster()
    lshkrepresentatives.CalcScore()


def TestDatasets(): 
    for dbname in MeasureManager.DATASET_LIST:
        DB = tulti.LoadRealData(dbname)
        MeasureManager.CURRENT_DATASET = dbname
        MeasureManager.CURRENT_MEASURE = 'Overlap'
        print("\n\n############## kRepresentatives_ATai ###################")
        alo = kRepresentatives_ATai(DB['DB'],DB['labels_'],dbname=MeasureManager.CURRENT_DATASET )
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