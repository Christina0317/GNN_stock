import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import pandas as pd
import numpy as np
from data_local import FeatureType, StockData


class DataProcess:
    def __init__(self, df1, df2, features, dates, stock_ids):
        self.df1 = df1
        self.df2 = df2
        self._features = features if features is not None else list(FeatureType)
        self.device = torch.device('cpu')
        self._dates = dates
        self._stock_ids = stock_ids

        self._stock_ids_cleaned = None
        self._dates_cleaned = None
        self.over_threshold_stocks = None   # nan值太多的stock

        # 清洗数据
        self.data1, self.data2 = self.validate_data()

        # 归一化数据
        self.data1 = self.normalized_data(self.data1)

    def df_to_tensor(self, df, num_features, num_stocks):
        """
        将 df 转换为 tensor
        :param df: [date/feature, stock]
        """
        values = df.values
        values = values.reshape((-1, num_features, num_stocks))
        return torch.tensor(values, dtype=torch.float, device=self.device)

    def tensor_to_df(self, data):
        """
        将 PyTorch 张量转换为 DataFrame
        :return: df -> [date, stock_id/feature]
        """
        data = data.numpy()
        data = np.transpose(data, (0,2,1))
        feature_name = [f.name.lower() for f in self._features]
        feature_name.append('return')
        multi_columns = pd.MultiIndex.from_product([self._stock_ids, feature_name], names=['stock_id', 'feature'])
        df = pd.DataFrame(data.reshape(len(self._dates), -1), index=self._dates, columns=multi_columns)
        return df

    def validate_data(self):
        """
        清洗数据
        :return: data1, data2 -> tensor[date, feature, stock]
        其中 data1 作为 x, data2 作为 y
        """
        data1, data2 = self.df_to_tensor(self.df1, len(self._features), len(self._stock_ids)), self.df_to_tensor(self.df2, 1, len(self._stock_ids))
        data = self.combine_data(data1, data2)

        if torch.isnan(data).any() or torch.isinf(data).any():
            nan_proportion, inf_proportion = self.calculate_nan_inf_proportions(data)
            print('The nan proportion / inf proportion is ', nan_proportion, inf_proportion)

            if nan_proportion < 0.5 and inf_proportion < 0.5:
                df = self.tensor_to_df(data)
                df = df.replace([np.inf, -np.inf], np.nan)

                # 前值填充
                df.ffill(inplace=True)

                # 判断是否有一列50%都是nan, 有则删除
                nan_ratios = df.isna().mean()
                over_threshold_stocks = nan_ratios[nan_ratios > 0.5].index.get_level_values(0).unique()
                df = df.drop(columns=over_threshold_stocks, level=0)

                # 删除有nan的行
                df = df.dropna(how='any', axis=0)

                self._stock_ids_cleaned = df.columns.get_level_values(0).unique()
                self._dates_cleaned = df.index
                self.over_threshold_stocks = over_threshold_stocks

                # df to tensor
                clean_data = self.df_to_tensor(df, len(self._features)+1, len(self._stock_ids_cleaned))

            else:
                raise ValueError("There are too many nan or inf values")

            if torch.isnan(clean_data).any() or torch.isinf(clean_data).any():
                raise ValueError("Input data contains NaN or Inf values")

            # 分离清洗后的数据回 data1 和 data2
            clean_data1 = clean_data[:, :-1, :]  # 所有行，除最后一个特征外的所有特征，所有股票
            clean_data2 = clean_data[:, -1, :]  # 所有行，最后一个特征，所有股票

            return clean_data1, clean_data2
        else:
            return data1, data2

    def calculate_nan_inf_proportions(self, data):
        """
        data -> tensor[date, feature, stock]
        计算并返回数据中 NaN 和 Inf 的比例
        :return: (float, float): 第一个值是 NaN 的比例，第二个值是 Inf 的比例
        """
        total_elements = data.numel()  # 总元素数量
        nan_count = torch.isnan(data).sum().item()  # NaN 元素数量
        inf_count = torch.isinf(data).sum().item()  # Inf 元素数量

        nan_proportion = nan_count / total_elements  # NaN 比例
        inf_proportion = inf_count / total_elements  # Inf 比例

        return nan_proportion, inf_proportion

    def combine_data(self, data1, data2):
        """
        data1: x -> tensor[date, feature, stock]
        data2: y -> tensor[date, 1, stock]
        将两个data合并, 这样在删除nan值的时候方便对齐data & stock_id
        :return: combined_data -> tensor[date, feature, stock],  feature[-1] = return
        """
        combined_data = torch.cat((data1, data2), dim=1)
        return combined_data

    def normalized_data(self, data):
        """
        对去除nan和inf的数据进行归一化处理, 截面归一化
        :param data: tensor[date, feature, stock]
        :return:
        """
        # print(data[0,0,0].dtype)
        mean_vals = data.mean(dim=2, keepdim=True)
        std_vals = data.std(dim=2, keepdim=True)

        normalized_data = (data - mean_vals) / std_vals
        normalized_data[torch.isnan(normalized_data)] = 0   # 处理分母为0的情况
        normalized_data = normalized_data.round(decimals=2)
        # print(normalized_data[0,0,0].dtype)
        return normalized_data


if __name__ == '__main__':
    # qlib
    # stock_data = StockData(
    #     instrument='all',  # 示例股票代码
    #     start_time='2020-11-01',
    #     end_time='2020-12-31',
    #     features=[FeatureType.OPEN, FeatureType.CLOSE]  # 仅加载开盘和收盘价
    # )
    # return_data = stock_data.calculate_return()

    # local
    stock_data = StockData(
        start_date='20230301',
        end_date='20230501',
        features=[FeatureType.OPEN, FeatureType.CLOSE]
    )
    data = stock_data.daily_data_from_h5()
    return_data = stock_data.calculate_return(data)
    dp = DataProcess(data, return_data, stock_data.features, stock_data.dates, stock_data.stock_ids)
