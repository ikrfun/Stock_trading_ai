import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optimizer as optim
import meke_dataset.py 

y_seq, y_target =make_data_for_lstm(data, window_size) #importした前処理関数呼び出し
num_hidden = 100 #モデルのLSTMブロックの数を定義
lr = 0.001 #ラーニングレートを設定
num_epoch = 100

y_seq_t = y_seq.FloatTensor() #　GPU処理のためにTensor型に変換する
y_target_t = y_target.FloatTensor()
y_seq_t = y_seq_t.permute(1,0) #次元の入れ替え
y_target_t = y_target_t.permute(1,0) 
y_seq_t = y_seq_t.unsequeeze(dim = -1) #３次元にする
y_target_t = y_target_t.unsequeeze(dim = -1)
criterion = nn.MSELoss()

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




        
