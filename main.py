import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
from data_local import FeatureType, StockData
from training import TrainProcess
import matplotlib.pyplot as plt
import numpy as np
import datetime


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    stock_data = StockData(
        start_date='20220101',
        end_date='20221231',
        features=[FeatureType.CLOSE]
    )
    data = stock_data.daily_data_from_h5()
    return_data = stock_data.calculate_return(data)
    end_time = datetime.datetime.now()
    print('Data is loaded successfully. Used time is', end_time - start_time)

    num_epochs = 100
    batch_size = 64

    tp = TrainProcess(data, return_data, stock_data.features, stock_data.dates, stock_data.stock_ids, num_epochs, batch_size)

    print('Training begins')
    in_channels, out_channels, hidden_size, hidden_size_gru = 10, 10, 32, 2*len(stock_data.stock_ids)
    model, losses = tp.train_gnn_gru_model(in_channels, out_channels, hidden_size, hidden_size_gru)

    losses_arr = []
    for l in losses:
        losses_arr.append(l.detach().numpy())
    plt.scatter(np.arange(len(losses_arr)), losses_arr)
    plt.show()

