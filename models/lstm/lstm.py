import torch
import torch.nn as nn
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

stock_data = pd.read_csv(
    "stock_data/^GSPC.csv",
    index_col=0,
    parse_dates=True
    
)

stock_data.drop(
    ["Open","High","Low","Close","Volume"],
    axis= "columns",
    inplace=True
)

stock_data.plot(figsize=(12,4))

y = stock_data["Adj Close"].values

from sklearn.preprocessing import MinMaxScaler　#正規化
scaler = MinMaxScaler(feature_range = (-1,1)) #インスタンス化
scaler.fit(y.reshape(-1,1))　#fitさせる
y = scaler.transform(y.reshape(-1,1))　＃yを正規化

y = torch.FloatTensor(y).view(-1) #FloatTensorに型変換　view(-1)で行列をいい感じに並べる

test_size = 24 #テスト使うデータの数

train_seq = y[:-test_size]#最後の２４個以外のデータをトレーニングに用いる
test_seq = y[-test_size:] #最後の２４個のデータをテストデータとする

train_window_size = 12

def input_data(seq, ws): #LSTM用に訓練データ生成関数
    out = []
    L = len(seq)
    
    for i in range(L - ws):
        window  = seq[i:i+ws]
        label = seq[i+ws : i+ws+1]
        out.append((window, label))
    
    return out

train_data = input_data(train_seq, train_window_size)

class Model(nn.Module): #nn.Moduleクラスを継承してモデルの作成（LSTM）
    def __init__(self, input = 1, h = 50, output = 1):
        """
        hiddeen_size は隠れ層にあるLSTMブロックの個数　増やすと計算コストが爆増する
        
        """
        super().__init__()
        self.hidden_size = h
        self.lstm = nn.LSTM(input, h)
        self.fc = nn.Linear(h,output)
        self.hidden = (
            torch.zeros(1,1,h),
            torch.zeros(1,1,h)
        )
    
    def forward(self,seq):
        out,_ = self.lstm(
            seq.view(len(seq), 1, -1),
            self.hidden
        )
        out = self.fc(
            out.view(len(seq),-1)
        )
        
        return out[-1]
    
torch.manual_seed(123)
model = Model()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001) 
epoch = 10
train_losses = []
test_losses = []

run_train():
    model.train() #trainモードに切り替える drop outなどの挙動に影響する
    for train_window, correct_label in train_data:
        optimizer.zero_grad()
        model.hidden = (
            torch.zeros(1,1,model.hidden_size),
            torch.zeros(1,1,model.hidden_size)
        )
    
    train_predicted_label = model.forward(train_window)
    train_loss = criterion(train_predicted_label, correct_label)
    train_loss.backward()
    optimizer.step()
    train_losses.append(train_loss)

def run_test():
    model.eval()
    for i in range(test_size):
        test_window = torch.FloatTensor(extending_seq[-test_size:])
        with torch.no_grad():#勾配計算を行わない　＝　メモリの節約
            model.hidden = (
                torch.zeros(1,1,model.hidden_size),
                torch.zeros(1,1,model.hidden_size)
            )
            test_predicted_label = model.forward(test_window)
            extending_seq.append(test_predicted_label.item())
    
    test_loss = criterion(
        torch.FloatTensor(extending_seq[-test_size:]),
        y[len(y)-test_size:]
    )
    test_losses.append(test_loss)
    
train_seq[-test_size:].tolist()

for epoch in range(epochs):
    print()
    print(f'Epoch:{epoch+1}')
    
    run_train()
    extending_seq = train_seq[-test_size:].tolist()
    run_test()
    
    plt.figure(figsize=(12,4))
    plt.xlim(-20, len(y)+20)
    plt.grid(True)
    
    plt.plot(y.numpy())
    plt.plot(
        range(len(y)-test_size, len(y)),
        extending_seq[-test_size:]
    )
    
    plt.show()