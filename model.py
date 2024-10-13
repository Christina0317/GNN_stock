import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.nn import GCNConv, MessagePassing
import datetime
from torch_geometric.utils import add_self_loops, degree
from data_local import FeatureType, StockData
from data_process import DataProcess
from torch.nn import MSELoss, L1Loss, Linear, Flatten, BatchNorm1d, BatchNorm2d, LeakyReLU, Dropout, Sigmoid


class DynamicGNN(nn.Module):
    def __init__(self, data, edge_index, edge_weight, consider_time_steps):
        # data -> shape(date, features, stocks)
        super(DynamicGNN, self).__init__()
        self.num_features = data.size(1)
        self.num_stocks = data.size(2)
        self.num_steps = data.size(0)
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.consider_time_steps = consider_time_steps

        # 图卷积层
        self.gcn1 = GCNConv(self.num_features, 64)
        self.gcn2 = GCNConv(64, 64)

        # GRU, input_size -> (seq_len, batch_size, features)
        self.gru = nn.GRU(input_size=64, hidden_size=32, num_layers=1)

        # 预测层
        self.predictor = nn.Linear(32, 1)

    def forward(self, x):
        # x -> [date, features, stocks]

        # 把日期当作序列，对每一个时间点的图进行操作
        h = None  # GRU的隐藏状态初始化
        predictions = []
        gcn_outputs = []
        for t in range(self.num_steps):
            x_t = x[t].T   # x_t -> (stocks, features)
            edge_index_t = self.edge_index[t]
            edge_weight_t = self.edge_weight[t]

            x_t = self.gcn1(x_t, edge_index_t, edge_weight_t)
            start_time = datetime.datetime.now()
            x_t = F.relu(x_t)
            x_t = self.gcn2(x_t, edge_index_t, edge_weight_t)
            x_t = F.relu(x_t)
            gcn_outputs.append(x_t)

            inputs_for_gru = gcn_outputs[-self.consider_time_steps:] if len(
                gcn_outputs) >= self.consider_time_steps else gcn_outputs[:]
            x_gru = torch.stack(inputs_for_gru, dim=0)  # Add batch dimension
            _, h = self.gru(x_gru)  # Process with GRU
            prediction = self.predictor(h.squeeze(0))  # Generate output
            predictions.append(prediction)

        predictions = torch.stack(predictions, dim=0)
        predictions = predictions.squeeze(2)
        # predictions = torch.cat(predictions, dim=0)
        return predictions


class GraphAutoencoder(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphAutoencoder, self).__init__()
        self.encoder = GCNConv(in_features, out_features)
        self.decoder = GCNConv(out_features, in_features)  # 简化示例，实际可以自定义更合适的解码器

    def forward(self, x, edge_index):
        x = x.T
        z = self.encoder(x, edge_index)
        adj_pred = torch.sigmoid(torch.mm(z, z.t()))  # 使用点积后应用sigmoid获取边权重
        return adj_pred


class GNNAndGRU(torch.nn.Module):
    def __init__(self, num_nodes, in_channels, out_channels, hidden_size, hidden_size_gru, edge_index):
        super(GNNAndGRU, self).__init__()

        self.gru = nn.GRU(input_size=num_nodes, hidden_size=hidden_size_gru, batch_first=True)  # 输入和输出的形状为 (batch_size, seq_length, feature_dim)
        self.gcn = GCNConv(in_channels, out_channels, add_self_loops=False)

        self.linear = Linear(hidden_size_gru, num_nodes)
        self.linearf1 = Linear(in_channels, 2*in_channels)
        self.linearf2 = Linear(in_channels, 2*in_channels)
        self.lineara1 = Linear(2*in_channels, hidden_size)
        self.bna1 = BatchNorm2d(hidden_size)
        self.lineara2 = Linear(hidden_size, hidden_size)
        self.bna2 = BatchNorm2d(hidden_size)
        self.lineara_last = Linear(hidden_size, 1)
        self.bna_last = BatchNorm2d(num_nodes)
        self.bnz = BatchNorm1d(num_nodes)
        self.linearz = Linear(out_channels, out_channels)

        self.lrelu = LeakyReLU(0.1)
        self.dropout = Dropout(0.1)

        self.adj1 = torch.rand((num_nodes, num_nodes)).to('cpu')
        self.edge_index = edge_index

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.linear(x)
        x = torch.transpose(x, 1, 2)
        f1 = self.lrelu(self.linearf1(torch.abs(x.unsqueeze(2) - x.unsqueeze(1))))
        f2 = self.lrelu(self.linearf2(torch.mul(x.unsqueeze(2), x.unsqueeze(1))))
        A_hat = f1 + f2
        A_hat = self.lineara1(A_hat)
        A_hat = A_hat.permute(0, 3, 1, 2)
        A_hat = self.lrelu(self.bna1(A_hat))
        A_hat = A_hat.permute(0, 2, 3, 1)
        A_hat = self.lineara2(A_hat)
        A_hat = A_hat.permute(0, 3, 1, 2)
        A_hat = self.dropout(self.lrelu(self.bna2(A_hat)))
        A_hat = A_hat.permute(0, 2, 3, 1)
        A_hat = torch.mean(self.lineara_last(A_hat), dim=3)
        A_hat = torch.sigmoid((A_hat + A_hat.transpose(-1, -2)) / 2)

        # 将 A_hat 转换为和 edge_index 匹配的格式
        edge_weights = []
        for i in range(self.edge_index.shape[1]):
            for batch in range(x.shape[0]):
                A_batch = A_hat[batch]
                weight = A_batch[self.edge_index[0, i]][self.edge_index[1, i]]
                edge_weights.append(weight)
        edge_weights = torch.tensor(edge_weights)
        edge_weights = edge_weights.view(self.edge_index.shape[1], -1)

        preds = []
        for batch in range(x.shape[0]):
            edge_weight = edge_weights[:, batch]
            x_input = x[batch, :, :]
            pred = self.gcn(x_input, self.edge_index, edge_weight)
            preds.append(pred)
        preds = torch.stack(preds, dim=0)
        # out = torch.mean(preds, dim=0)
        preds = preds.transpose(1, 2)

        return preds, A_hat

