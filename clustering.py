import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
import fastcluster



class Clustering:

    @staticmethod
    def hierarchical_cluster(data, cluster_threshold):
        """
        data -> shape(steps, n_features, n_stocks)
        分层聚类算法, 对过去的m个时间点进行股票数据分类, 得到的分类结果作为 edge index
        :return:
        """
        # data -> (stocks, steps * features)
        data = data.permute((2, 0, 1))
        data = data.reshape(data.size(0), -1)
        # distance_matrix -> shape(n_points*(n_points-1)/2)
        # Z -> shape(n_points-1, 4), 前两列是合并的聚类(点索引), 第三列是这次合并的距离, 第四列是合并后聚类中的数据点数量
        Z = fastcluster.linkage_vector(data, method='ward')
        # t=1.5 -> 合并的聚类之间的距离不能超过这个值
        # label -> shape(n_points), 每个数据点的聚类标签
        labels = fcluster(Z, t=cluster_threshold, criterion='distance')
        return labels


if __name__ == '__main__':
    n_points = 100
    n_features = 10
    n_stocks = 20
    cluster_threshold = 5
    data = torch.rand(n_points, n_features, n_stocks)
    label = Clustering.hierarchical_cluster(data, cluster_threshold)

