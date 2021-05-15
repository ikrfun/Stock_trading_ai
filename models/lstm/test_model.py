import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optimizer as optim


class make_datasets:
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
        num_data = len(self.y)
        seq_data = []
        target_data = []
        for i in range(num_data - num_window):
            seq_data.append(y[i:i+num_window])
            target_data.append(y[i+num_window:1+i+num_window])
        seq_arr = np.array(seq_data)
        target_arr = np.array(target_data)
    
    def make_data_for_mlp(self):
        pass#今後実装
    
    def lstm_train_data(self,num_test):
        return self.seq_train, self.target_train
    def lstm_test_data(self):
        return self.seq_test, self.target_test
    
        
    
    
    
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
    

    

data_set = make_datasets("./test_model","opening_price") #インスタンス作成
y_seq, y_target = data_set.make_data_for_lstm(30)#30日分のデータを１区切りとして学習する
y_seq_t = y_seq.FloatTensor() #　GPU処理のためにTensor型に変換する
y_target_t = y_target.FloatTensor()
y_seq_t = y_seq_t.permute(1,0) #次元の入れ替え
y_target_t = y_target_t.permute(1,0) 
y_seq_t = y_seq_t.unsequeeze(dim = -1) #３次元にする
y_target_t = y_target_t.unsequeeze(dim = -1)
criterion = nn.MSELoss()

num_hidden = 100 #モデルのLSTMブロックの数を定義
lr = 0.001 #ラーニングレートを設定
num_epoch = 100
model = LSTM(num_hidden)
optimizer = optim.Adam(model.parameters(),lr = lr)

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
        
#テスト
#テストデータの整形
y_seq_test_t = y_seq_test.FloatTensor()
y_seq_test_t = y_seq_test_t.permute(1,0)
y_seq_test_t = y_seq_test_t.unsequeeze(dim = -1)
#modelにテストデータを流し、予測を行う
y_pred = model(y_seq_test_t)
#予測したデータをPlotしてみる
x = time_data #使用したデータの時系列情報
y = trading_price #使用した株価のrawデータ
plt.plot(x,y)
plt.plot("日付の範囲(pandas_date_data)",y_pred.detach())
plt.xlim(["日付の範囲(pandas_date_data"])