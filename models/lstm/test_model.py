import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optimizer as optim


class make_datasets:#データの前処理を行えるクラス
    def __init__(self,csv_file,col_name, type, num_window=0):
        raw_data = pd.read_csv(csv_file)
        self.price = raw_data[col_name]
        self.date = raw_data["date"]
        self.seq_arr = []
        self.target_arr = []
        self.seq_train = []
        self.target_train = []
        self.seq_test = []
        self.target_test = []
        if type == "lstm":
            self.make_data_for_lstm(num_window)
        elif type == "mlp":
            self.make_data_for_mlp()
        else:
            print("type:{}は存在しません".format(type))
        
    def make_data_for_lstm(self, num_window): #LSTMのためのデータセットを作成する
        num_data = len(self.price)
        seq_data = []
        target_data = []
        for i in range(num_data - num_window):
            seq_data.append(self.price[i:i+num_window])
            target_data.append(self.price[i+num_window:1+i+num_window])
        self.seq_arr = np.array(seq_data)
        self.target_arr = np.array(target_data)
    
    def make_data_for_mlp(self):
        pass#今後実装
    
    def lstm_train_data(self,num_test):
        self.seq_train = self.seq_arr[:-num_test]
        self.seq_target = self.target_arr[:-num_test]
        self.seq_test = self.seq_arr[-num_test :]
        self.target_test = self.target_arr[-num_test :]
        return self.seq_train, self.target_train, self.seq_test, self.target_test
    
#モデル定義
class LSTM(nn.Module):
    def __init__(self,hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size = 1, hidden_size = self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, 1)
    def forward(self,x):
        x,_ = self.lstm(x)
        x_last = x[-1]
        x = self.linear(x_last)
        return x

# 学習
def learn(seq_data,target_data,num_epoch):
    losses = []
    for epoch in range(num_epoch):
        optimizer.zero_grad()
        output = model(seq_data)
        loss = criterion(output,target_data)
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        if epoch % 10 == 0:
            print("epoch{} : loss{}".format(epoch,loss.item()))
        plt.plot(losses)

data_set = make_datasets("./test_model","opening_price","lstm",30) #インスタンス作成 30日分のデータを１区切りとして学習する
seq_train, target_train, seq_test, target_test = data_set.lstm_train_data(10)
seq_train_t = seq_train.FloatTensor() #　GPU処理のためにTensor型に変換する
target_train_t = target_train.FloatTensor()
seq_train_t = seq_train_t.permute(1,0) #次元の入れ替え
target_train_t= target_train_t.permute(1,0) 
seq_train_t = seq_train_t.unsequeeze(dim = -1) #３次元にする
target_train_t = target_train_t.unsequeeze(dim = -1)
criterion = nn.MSELoss()
num_hidden = 100 #モデルのLSTMブロックの数を定義
lr = 0.001 #ラーニングレートを設定
num_epoch = 100
model = LSTM(num_hidden)
optimizer = optim.Adam(model.parameters(),lr = lr)
#学習
learn(seq_train_t,target_train_t,num_epoch)
#テスト
#テストデータの整形
seq_test_t = seq_test.FloatTensor()
seq_test_t= seq_test_t.permute(1,0)
seq_test_t = seq_test_t.unsequeeze(dim = -1)
#modelにテストデータを流し、予測を行う
y_pred = model(seq_test_t)
#予測したデータをPlotしてみる
x = data_set.date #使用したデータの時系列情報
y = data_set.price #使用した株価のrawデータ
plt.plot(x,y)
plt.plot("日付の範囲(pandas_date_data)",y_pred.detach())
plt.xlim(["日付の範囲(pandas_date_data"])