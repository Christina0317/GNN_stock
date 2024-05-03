# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.optim as optim
import torch.nn as nn
from model import DynamicGNN
from data_local import FeatureType, StockData
from data_process import DataProcess
from clustering import Clustering
from tqdm import tqdm
from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np
import datetime


class TrainProcess:
    def __init__(self, data, return_data, num_epochs):   # data -> tensor[time_steps, features, stocks]
        self.num_features = data.size(1)
        self.num_stocks = data.size(2)
        self.num_steps = data.size(0)
        self.data = data
        self.last_m = 100   # 聚类的时候取过去的时间步长
        self.num_epochs = num_epochs
        self.return_data = return_data

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
        with Pool(processes=4) as pool:  # 调整进程数以适应您的系统
            clustering_results = pool.starmap(Clustering.hierarchical_cluster, tasks)

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

    def apply_to_model(self):
        """
        将分层聚类算法和 dynamic GNN 相结合
        :return:
        """
        # generate edge information
        start_time = datetime.datetime.now()
        cluster_threshold = 5
        edge_indices, edge_weights = self.edge_info(cluster_threshold)
        end_time = datetime.datetime.now()
        print('The edge information is generated successfully. Used time is ', end_time-start_time)

        start_time = datetime.datetime.now()
        # init model
        model = DynamicGNN(self.data, edge_indices, edge_weights, consider_time_steps=10)
        # init optimizer
        criterion = nn.MSELoss()  # 使用均方误差作为损失函数
        optimizer = optim.Adam(model.parameters(), lr=0.05)
        losses = []

        with tqdm(range(self.num_epochs), desc='Processing') as pbar:
            for epoch in pbar:
                optimizer.zero_grad()

                predictions = model(self.data)  # forward

                loss = criterion(predictions, self.return_data)  # loss

                loss.backward()  # 反向传播，计算参数的梯度
                optimizer.step()  # 使用梯度更新参数

                if epoch % 10 == 0:  # 每10个epoch打印一次损失
                    print(f"Epoch {epoch}, Loss: {loss.item()}")

                losses.append(loss.detach().numpy())

        end_time = datetime.datetime.now()
        print('The training process is finished. Used time is ', end_time - start_time)

        return model, losses


if __name__ == '__main__':
    stock_data = StockData(
        start_date='20230301',
        end_date='20230501',
        features=[FeatureType.OPEN, FeatureType.CLOSE]
    )
    data = stock_data.daily_data_from_h5()
    return_data = stock_data.calculate_return(data)
    dp = DataProcess(data, return_data, stock_data.features, stock_data.dates, stock_data.stock_ids)

    num_epochs = 100
    data1 = dp.data1[:, :, :]
    data2 = dp.data2[:, :]

    tp = TrainProcess(data1, data2, num_epochs)

    model, losses = tp.apply_to_model()

    plt.scatter(np.arange(len(losses)), losses)