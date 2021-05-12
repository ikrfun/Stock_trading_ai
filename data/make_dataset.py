import numpy as np
import pandas
import pandas_datareader
from pandas_datareader import data

class make_datasets():
    def __init__(self) -> None:
        pass
    def make_data_for_lstm(self, y, num_window): #LSTMのためのデータセットを作成する
        num_data = len(y)
        seq_data = []
        target_data = []
        for i in range(num_data - num_window):
            seq_data.append(y[i:i+num_window])
            target_data.append(y[i+num_window:1+i+num_window])
        seq_arr = np.array(seq_data)
        target_arr = np.array(target_data)
        return seq_arr, target_arr

    def make_trading_data(self,):