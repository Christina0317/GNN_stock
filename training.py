# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
from model import *
from edge_model import EdgeInfo
from data_local import FeatureType, StockData
from data_process import DataProcess, TimeseriesDataset
from tqdm import tqdm
import datetime
from torch.utils.data import DataLoader, TensorDataset


class TrainProcess:
    """
    feature_data: dataframe -> [len(dates)/features, num_stocks]
    return_data: dataframe -> [len(dates)/return, num_stocks]
    features: list[FeatureType]
    dates: list[str]
    stock_ids: index
    num_epochs: int
    batch_size: int

    Parameter of class
    feature_tensor: tensor -> [len(dates), len(features), num_stocks]
    return_tensor: tensor -> [len(dates), num_stocks]
    """
    def __init__(self, feature_data, return_data, features, dates, stock_ids, num_epochs, batch_size):
        self.feature_data = feature_data
        self.num_features = len(features)
        self.num_stocks = None   # 去除了都是nan的列
        self.num_steps = None    # 去除了都是nan的行
        self.num_epochs = num_epochs
        self.return_data = return_data
        self.features = features
        self.dates = dates
        self.stock_ids = stock_ids  # 未删除列之前的 stock index
        self.batch_size = batch_size

        self.feature_tensor, self.return_tensor = self.data_cleaning()

    def data_cleaning(self):
        start_time = datetime.datetime.now()
        dp = DataProcess(self.feature_data, self.return_data, self.features, self.dates, self.stock_ids)
        self.feature_data = self.feature_data.drop(columns=dp.over_threshold_stocks)  # data -> dataframe
        data1 = dp.data1[:, :, :]
        data2 = dp.data2[:, :]
        self.num_stocks = data1.size(2)
        self.num_steps = data1.size(0)
        self.stock_ids = self.feature_data.columns   # 删除列之后的 stock index
        end_time = datetime.datetime.now()
        print('Data is cleaned successfully. Used time is', end_time - start_time)
        return data1, data2

    def train_dynamic_gnn(self):
        """
        将分层聚类算法和 dynamic GNN 相结合
        :return:
        """
        # generate edge information
        start_time = datetime.datetime.now()
        cluster_threshold = 5
        edgeinfo = EdgeInfo(self.feature_tensor)
        # edge_indices, edge_weights = edgeinfo.fully_connected_edge_info()
        edge_indices, edge_weights = edgeinfo.edge_info(cluster_threshold)
        end_time = datetime.datetime.now()
        print('The edge information is generated successfully. Used time is ', end_time-start_time)

        start_time = datetime.datetime.now()
        # init model
        model = DynamicGNN(self.feature_tensor, edge_indices, edge_weights, consider_time_steps=10)
        # init optimizer
        criterion = nn.MSELoss()  # 使用均方误差作为损失函数
        optimizer = optim.Adam(model.parameters(), lr=0.05)
        losses = []

        with tqdm(range(self.num_epochs), desc='Processing') as pbar:
            for epoch in pbar:
                optimizer.zero_grad()

                predictions = model(self.feature_tensor)  # forward

                loss = criterion(predictions, self.return_data)  # loss

                loss.backward()  # 反向传播，计算参数的梯度
                optimizer.step()  # 使用梯度更新参数

                if epoch % 10 == 0:  # 每10个epoch打印一次损失
                    print(f"Epoch {epoch}, Loss: {loss.item()}")

                losses.append(loss.detach().numpy())

        end_time = datetime.datetime.now()
        print('The training process is finished. Used time is ', end_time - start_time)

        return model, losses

    def train_gnn_gru_model(self, in_channels, out_channels, hidden_size, hidden_size_gru):
        """
        用gnn去聚合邻近节点的信息, 并且结合gru进行时序分析
        :return:
        """
        industry_info = pd.read_csv('/Volumes/E/quant_data/zjw300_code.csv', index_col=0)
        edgeinfo = EdgeInfo(self.feature_tensor)
        edge_index, edge_index_matrix = edgeinfo.get_edge_index_by_industry(industry_info, self.stock_ids)

        model = GNNAndGRU(self.num_stocks, in_channels, out_channels, hidden_size, hidden_size_gru, edge_index)
        optimizer = optim.Adam(model.parameters(), lr=0.1)
        # loss_func = nn.L1Loss()  # 二元交叉熵损失函数
        loss_func = nn.MSELoss()

        td = TimeseriesDataset(self.return_tensor, in_channels, out_channels)
        x_tensor, y_tensor = td.x_timeseries, td.y_timeseries

        # 转化为batch进行训练
        dataset = TensorDataset(x_tensor, y_tensor)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        model.train()
        losses = []
        for epoch in range(self.num_epochs):
            for x, y in data_loader:
                optimizer.zero_grad()
                pred, A_hats = model(x)
                loss = loss_func(pred.view(y.shape), y)
                loss.backward()
                optimizer.step()
                losses.append(loss)
            print(f"Epoch {epoch}, Loss: {loss.item()}")

        return model, losses


if __name__ == '__main__':
    stock_data = StockData(
        start_date='20230301',
        end_date='20230501',
        features=[FeatureType.OPEN, FeatureType.CLOSE]
    )
    data = stock_data.daily_data_from_h5()
    return_data = stock_data.calculate_return(data)
    # dp = DataProcess(data, return_data, stock_data.features, stock_data.dates, stock_data.stock_ids)
    # over_threshold_stocks = dp.over_threshold_stocks
    # data = data.drop(columns=over_threshold_stocks)   # data -> dataframe

    num_epochs = 100
    batch_size = 1
    # data1 = dp.data1[:, :, :]
    # data2 = dp.data2[:, :]

    tp = TrainProcess(data, return_data, stock_data.features, stock_data.dates, stock_data.stock_ids, num_epochs, batch_size)

    # model, losses = tp.apply_to_model()
    #
    # plt.scatter(np.arange(len(losses)), losses)

    model, losses = tp.train_gnn_gru_model()
