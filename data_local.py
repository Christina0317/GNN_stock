from typing import List, Union, Optional, Tuple
from enum import IntEnum
import pickle
import pandas as pd
import torch
import h5py


class FeatureType(IntEnum):
    OPEN = 0
    CLOSE = 1
    HIGH = 2
    LOW = 3
    VOLUME = 4
    VWAP = 5
    RETURN = 6


class StockData:
    def __init__(self,
                 start_date: str,
                 end_date: str,
                 max_backtrack_days: int = 100,
                 max_future_days: int = 30,
                 features: Optional[List[FeatureType]] = None,
                 path: str = '/Volumes/E/quant_internship_sgd/data/db_jq_candle_minute_w_nan.h5',
                 device: torch.device = torch.device('cpu')
                 ):
        self._start_date = start_date
        self._end_date = end_date
        self.max_backtrack_days = max_backtrack_days
        self.max_future_days = max_future_days
        self.device = device
        # self.features = [f.name.lower() for f in features]
        self.features = features if features is not None else list(FeatureType)
        self.path = path
        self.stock_ids = None
        self.dates = None

    def daily_data_from_h5(self):
        with h5py.File(self.path, 'r') as file:
            dates = [date for date in file.keys() if (int(date) >= int(self._start_date)) * (int(date) <= int(self._end_date))]

        daily_data = pd.DataFrame()
        for date in dates:
            data = pd.read_hdf(self.path, key=date)
            data = data.groupby('code').apply(lambda x: x.iloc[-1], include_groups=False)
            data['time'] = data['time'].dt.strftime("%Y-%m-%d")   # data.columns = ['code', 'time', 'open'...]
            # 筛选 feature
            data = data[['time'] + [f.name.lower() for f in self.features]]
            data = data.reset_index()
            data = data.melt(id_vars=['code', 'time'], var_name='feature', value_name='value')
            data.set_index(['time', 'feature', 'code'], inplace=True)
            data = data['value'].unstack(level='code')
            daily_data = pd.concat([daily_data, data])

        self.stock_ids = daily_data.columns
        self.dates = dates
        return daily_data

    def daily_data_from_pkl(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            pkl_data = pickle.load(f)
        dates = [date for date in pkl_data.keys() if (int(date) >= int(self._start_date)) * (int(date) <= int(self._end_date))]

        daily_data = pd.DataFrame()
        for date in dates:
            data = pkl_data[date]
            # 筛选 feature
            data = data[['time'] + [f.name.lower() for f in self.features]]
            data = data.reset_index()
            data = data.melt(id_vars=['code', 'time'], var_name='feature', value_name='value')
            data.set_index(['time', 'feature', 'code'], inplace=True)
            data = data['value'].unstack(level='code')
            daily_data = pd.concat([daily_data, data])
        return daily_data

    def calculate_return(self, df):
        """
        用输入的数据data去计算return (未来20天收益率)
        :return:
        """
        df = df.reset_index('feature')
        df = df[df['feature'] == 'close']
        df['feature'] = 'return'
        df = df.reset_index()
        df = df.set_index(['time', 'feature'])
        return_df = df.shift(-20) / df - 1
        return_df = return_df.reindex(columns=self.stock_ids)
        return return_df


if __name__ == '__main__':
    stock_data = StockData(
        start_date='20230301',
        end_date='20230501',
        features=[FeatureType.OPEN, FeatureType.CLOSE]
        )
    data = stock_data.daily_data_from_h5()
    return_data = stock_data.calculate_return(data)
