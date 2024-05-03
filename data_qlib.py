import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
from typing import List, Union, Optional, Tuple
from enum import IntEnum
import numpy as np
import pandas as pd


class FeatureType(IntEnum):
    OPEN = 0
    CLOSE = 1
    HIGH = 2
    LOW = 3
    VOLUME = 4
    VWAP = 5
    RETURN = 6


class StockData:
    _qlib_initialized: bool = False

    def __init__(self,
                 instrument: Union[str, List[str]],
                 start_time: str,
                 end_time: str,
                 max_backtrack_days: int = 100,
                 max_future_days: int = 30,
                 features: Optional[List[FeatureType]] = None,
                 device: torch.device = torch.device('cpu')) -> None:
        self._init_qlib()

        self._instrument = instrument
        self.max_backtrack_days = max_backtrack_days
        self.max_future_days = max_future_days
        self._start_time = start_time
        self._end_time = end_time
        self._features = features if features is not None else list(FeatureType)
        self.device = device
        self.data, self.dates, self.stock_ids = self._get_data()


    @classmethod
    def _init_qlib(cls) -> None:
        if cls._qlib_initialized:
            return
        import qlib
        from qlib.config import REG_CN
        qlib.init(provider_uri="~/.qlib/qlib_data/cn_data_rolling", region=REG_CN)
        cls._qlib_initialized = True

    def _load_exprs(self, exprs: Union[str, List[str]]) -> pd.DataFrame:
        # This evaluates an expression on the data and returns the dataframe
        # It might throw on illegal expressions like "Ref(constant, dtime)"
        from qlib.data.dataset.loader import QlibDataLoader
        from qlib.data import D
        if not isinstance(exprs, list):
            exprs = [exprs]
        cal: np.ndarray = D.calendar()
        start_index = cal.searchsorted(pd.Timestamp(self._start_time))  # type: ignore
        end_index = cal.searchsorted(pd.Timestamp(self._end_time))  # type: ignore
        real_start_time = cal[start_index - self.max_backtrack_days]
        if cal[end_index] != pd.Timestamp(self._end_time):
            end_index -= 1
        real_end_time = cal[end_index + self.max_future_days]
        return (QlibDataLoader(config=exprs)  # type: ignore
                .load(self._instrument, real_start_time, real_end_time))

    def _get_data(self) -> Tuple[torch.Tensor, pd.Index, pd.Index]:
        features = ['$' + f.name.lower() for f in self._features]
        df = self._load_exprs(features)
        df = df.stack().unstack(level=1)
        df = df.sort_index(level=0)
        filtered_df = df.loc[(slice(pd.Timestamp(self._start_time), pd.Timestamp(self._end_time)), slice(None)), :]
        stock_ids = filtered_df.columns
        dates = filtered_df.index.get_level_values(0).unique()
        return filtered_df, dates, stock_ids

    def calculate_return(self):
        """
        用输入的数据data去计算return (未来20天收益率)
        :return:
        """
        feature = ['$' + FeatureType.CLOSE.name.lower()]
        df = self._load_exprs(feature)
        df = df.stack().unstack(level=1)
        df = df.reset_index()
        df['level_1'] = '$' + FeatureType.RETURN.name.lower()
        df = df.set_index(['datetime', 'level_1'])
        df = df.reindex(columns=self.stock_ids)
        return_df = df.shift(-20) / df - 1
        filtered_return_df = return_df.loc[(slice(pd.Timestamp(self._start_time), pd.Timestamp(self._end_time)), slice(None)), :]
        filtered_return_df = filtered_return_df.round(2)
        return filtered_return_df

    @property
    def n_features(self) -> int:
        return len(self._features)

    @property
    def n_stocks(self) -> int:
        return self.data.shape[-1]

    @property
    def n_days(self) -> int:
        return self.data.shape[0] - self.max_backtrack_days - self.max_future_days

    def make_dataframe(
        self,
        data: Union[torch.Tensor, List[torch.Tensor]],
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
            Parameters:
            - `data`: a tensor of size `(n_days, n_stocks[, n_columns])`, or
            a list of tensors of size `(n_days, n_stocks)`
            - `columns`: an optional list of column names
            """
        if isinstance(data, list):
            data = torch.stack(data, dim=2)
        if len(data.shape) == 2:
            data = data.unsqueeze(2)
        if columns is None:
            columns = [str(i) for i in range(data.shape[2])]
        n_days, n_stocks, n_columns = data.shape
        if self.n_days != n_days:
            raise ValueError(f"number of days in the provided tensor ({n_days}) doesn't "
                             f"match that of the current StockData ({self.n_days})")
        if self.n_stocks != n_stocks:
            raise ValueError(f"number of stocks in the provided tensor ({n_stocks}) doesn't "
                             f"match that of the current StockData ({self.n_stocks})")
        if len(columns) != n_columns:
            raise ValueError(f"size of columns ({len(columns)}) doesn't match with "
                             f"tensor feature count ({data.shape[2]})")
        if self.max_future_days == 0:
            date_index = self.dates[self.max_backtrack_days:]
        else:
            date_index = self.dates[self.max_backtrack_days:-self.max_future_days]
        index = pd.MultiIndex.from_product([date_index, self.stock_ids])
        data = data.reshape(-1, n_columns)
        return pd.DataFrame(data.detach().cpu().numpy(), index=index, columns=columns)


if __name__ == '__main__':
    try:
        # 初始化 StockData 实例
        stock_data = StockData(
            instrument='all',  # 示例股票代码
            start_time='2020-11-01',
            end_time='2020-12-31',
            features=[FeatureType.OPEN, FeatureType.CLOSE]  # 仅加载开盘和收盘价
        )
        # data type -> tensor(date, features, stocks)

        ret = stock_data.calculate_return()

        # 打印一些基本信息以确认加载正确
        print("Number of features:", stock_data.n_features)
        print("Number of stocks:", stock_data.n_stocks)
        print("Number of days:", stock_data.n_days)
        print("Sample data:\n", stock_data.data[:5])  # 打印前5天的数据

    except Exception as e:
        print("Failed to load stock data:", str(e))


