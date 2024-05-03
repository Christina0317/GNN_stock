# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
from data_local import FeatureType, StockData
from data_process import DataProcess
from training import TrainProcess
import matplotlib.pyplot as plt
import numpy as np
import datetime


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    stock_data = StockData(
        start_date='20230301',
        end_date='20230501',
        features=[FeatureType.OPEN, FeatureType.CLOSE]
    )
    data = stock_data.daily_data_from_h5()
    return_data = stock_data.calculate_return(data)
    end_time = datetime.datetime.now()
    print('Data is loaded successfully. Used time is', end_time - start_time)

    # 数据清洗
    start_time = datetime.datetime.now()
    dp = DataProcess(data, return_data, stock_data.features, stock_data.dates, stock_data.stock_ids)
    end_time = datetime.datetime.now()
    print('Data is cleaned successfully. Used time is', end_time - start_time)

    num_epochs = 50
    data1 = dp.data1[:, :, :]
    data2 = dp.data2[:, :]

    tp = TrainProcess(data1, data2, num_epochs)

    model, losses = tp.apply_to_model()

    plt.scatter(np.arange(len(losses)), losses)


