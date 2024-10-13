import torch
from multiprocessing import Pool
from clustering import Clustering
import numpy as np


class EdgeInfo:
    def __init__(self, data):
        self.num_features = data.size(1)
        self.num_stocks = data.size(2)
        self.num_steps = data.size(0)
        self.last_m = 100  # 聚类的时候取过去的时间步长
        self.data = data

    def generate_edge_info(self, labels):
        """
        将分层聚类算法得到的labels整合到edge index/edge weight中
        :return:
        """
        edge_index = []
        edge_weight = []
        for i in range(self.num_stocks):
            for j in range(i + 1, self.num_stocks):
                if labels[i] == labels[j]:  # 如果两个股票在同一个类别
                    edge_index.append((i, j))
                    edge_weight.append(1)

        edge_index = torch.tensor(edge_index).t().contiguous()
        edge_weight = torch.tensor(edge_weight, dtype=torch.float)
        return edge_index, edge_weight

    def edge_info(self, cluster_threshold):
        """
        将 generate_edge_info 得到的 edge index/edge weight 整理
        :return:
        """
        tasks = []

        for t in range(self.num_steps):
            # 每last_m步或在时间序列开始时进行聚类
            if t % self.last_m == 0 or t < self.last_m:
                # 计算窗口数据的开始索引，确保不会有负索引
                start_index = max(0, t - self.last_m)
                window_data = self.data[start_index:t + 1]  # 提取窗口数据
                tasks.append((window_data, cluster_threshold))

        # 使用 multiprocessing 进行并行聚类计算

        pool = Pool(processes=4)  # 创建进程池
        clustering_results = pool.starmap(Clustering.hierarchical_cluster, tasks)
        pool.close()  # 关闭进程池
        pool.join()  # 等待所有进程结束

        # except Exception as e:
        #     print(f"Error during multiprocessing: {e}")
        #
        # finally:
        #     print(1)
        #     pool.terminate()  # 无论如何都确保进程池被终止

        edge_indices = []
        edge_weights = []

        # 使用聚类结果生成边信息
        for t in range(self.num_steps):
            i = 0
            if t % self.last_m == 0 or t < self.last_m:
                labels = clustering_results[i]
                i += 1

            edge_index, edge_weight = self.generate_edge_info(labels)
            edge_indices.append(edge_index)
            edge_weights.append(edge_weight)

        return edge_indices, edge_weights

    def get_full_connected_edges(self):
        """
        生成全连接的边信息
        :return: edge_indices, edge_weights
        """
        edge_indices = []
        edge_weights = []

        # 对于每个股票与其他所有股票创建边
        for i in range(self.num_stocks):
            for j in range(self.num_stocks):
                if i != j:
                    edge_indices.append((i, j))
                    edge_weights.append(1.0)  # 可以调整边权重，这里假设为1

        edge_index = torch.tensor(edge_indices).t().contiguous()
        edge_weight = torch.tensor(edge_weights, dtype=torch.float)

        # 将边信息复制为每一个时间步的边信息
        edge_indices = [edge_index for _ in range(self.num_steps)]
        edge_weights = [edge_weight for _ in range(self.num_steps)]
        return edge_indices, edge_weights

    def get_edge_index_by_industry(self, industry_info, stock_ids):
        """
        通过行业信息得到图的edge index
        :return:
        """
        df = industry_info.reindex(list(stock_ids))

        industry_codes = df['industry_code'].values

        edge_list = []
        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                if industry_codes[i] == industry_codes[j]:
                    edge_list.append((i, j))
                    edge_list.append((j, i))  # 因为图是无向的，所以也需要添加反向边

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        # 生成 (num_stocks, num_stocks) 的边信息
        edge_index_matrix = np.zeros((len(df), len(df)))
        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                if industry_codes[i] == industry_codes[j]:
                    edge_index_matrix[i, j] = edge_index_matrix[j, i] = 1
        edge_index_matrix = torch.tensor(edge_index_matrix, dtype=torch.long)

        return edge_index, edge_index_matrix

