import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import datetime


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
        self.gcn1 = GCNConv(self.num_features, 128)
        self.gcn2 = GCNConv(128, 128)

        # GRU, input_size -> (seq_len, batch_size, features)
        self.gru = nn.GRU(input_size=128, hidden_size=64, num_layers=1)

        # 预测层
        self.predictor = nn.Linear(64, 1)

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
