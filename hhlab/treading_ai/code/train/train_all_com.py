#ライブラリインポート
import torch
import torch.nn as nn
import numpy as np 
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
from torch import FloatTensor
from sklearn import preprocessing as prp 

#全ての企業の証券コードを受け取り、そのデータからLSTMパラメータを作成する関数
def train_comps():
    train_list = np.array(pd.read_csv(
        '/home/ikrfun/Projects/hhlab/raw_data/com_code.csv',
        usecols=[1])
    ).flatten()
    len(train_list)
    for i in train_list:
        print('学習開始：{}'.format(i))
        make_param = Make_param(i)

#６次元の株価データを格納し、好きな時に取り出すことできるクラス
class DataLoader():
    def __init__(self,com_code,dir_path):
        self.seq = []
        self.target = []
        file_path = dir_path+str(com_code)+'.csv'
        self.raw = np.array(pd.read_csv(
            file_path,
            index_col = "date",
            parse_dates = True
        ))
        self.make_window(5)
        self.split(60)
        
    #5こずつの窓にしている（週５の取引に合わせて窓サイズは５とした）インスタンス化時に自動呼び出し
    def make_window(self,n_w):
        mmscaler = prp.MinMaxScaler(feature_range=(-1,1))
        normed = mmscaler.fit_transform(np.array(self.raw))
        for i in range(len(self.raw)-(n_w+1)):
            self.seq.append(normed[i:i+n_w])
            self.target.append([self.raw[i+n_w][0]])

    #訓練データと検証データを分けるメソッド　同じく自動呼び出し
    def split(self,num_test):
        self.seq_train = self.seq[:-num_test]
        self.seq_test = self.seq[-num_test:]
        self.target_train = self.target[:-num_test]
        self.target_test = self.target[-num_test:]
    #訓練に使うデータを取り出すメソッド
    def get_traindata(self):
        seq = FloatTensor(self.seq_train).permute(1,0,-1)
        target= FloatTensor(self.target_train).permute(0,1)
        return seq, target
    #検証ようデータを取り出すメソッド
    def get_validationdata(self):
        seq = FloatTensor(self.seq_test).permute(1,0,-1)
        target = FloatTensor(self.target_test)
        return seq, target

#LSTMモデルの定義　6次元データ専用
class Lstm(nn.Module):
    def __init__(self,num_hidden):
        super().__init__()
        self.hidden_size = num_hidden
        self.lstm = nn.LSTM(
            input_size = 6,
            hidden_size = self.hidden_size,
            )
        self.fc = nn.Linear(self.hidden_size,1)
    #順伝播
    def forward(self,x):
        x,_ = self.lstm(x)
        x_last = x[-1]
        x = self.fc(x_last)
        return x

#LSTM用学習済みパラメータの訓練、検証と保存を行う　関数にするつもりだったがいろいろありクラスにした
class Make_param():
    def __init__(self,com_code):
        self.com_code = com_code
        self.model = Lstm(1024)
        self.data = DataLoader(self.com_code,"/home/ikrfun/Projects/hhlab/raw_data/")
        self.gpu = 'cuda'
        self.criterion = nn.MSELoss()
        self.min_loss = -1
        self.train_min_loss = -1
        self.train()
    #パラメータのトレーニングを行うメソッド　エポック数、lrはハードコーディング
    def train(self):
        self.model.to(self.gpu)
        x,y = self.data.get_traindata()
        x = x.to(self.gpu)
        y = y.to(self.gpu)
        num_epochs = 100001
        optimizer = optim.AdamW(self.model.parameters(),lr = 0.0001)
        for epoch in range(num_epochs):
            self.model.train()
            optimizer.zero_grad()
            output = self.model(x)
            loss = self.criterion(output,y)
            loss.backward()
            optimizer.step()
            if epoch%10000 == 0:
                if self.train_min_loss < 0 or self.train_min_loss>loss.item():
                    self.train_min_loss = loss.item()
                    
                print("min_loss:{}".format(self.min_loss))
                self.save_params()
                print("train_min_loss:{}".format(self.train_min_loss))
    #パラメータをpthファイルに保存する、検証用データに対する誤差が最小のモデルが保存される事になる
    def save_params(self):
        dir_path = "/home/ikrfun/Projects/hhlab/treading_ai/trained_params"
        if self.validation():
            torch.save(self.model.state_dict(),dir_path+'/{}.pth'.format(self.com_code))
    #訓練したモデルの精度を測定（検証）を行う。最も誤差の少ないモデルが産まれたらTrueを返し、パラメータを保存する
    def validation(self):
        x,y = self.data.get_validationdata()
        x = x.to(self.gpu)
        y = y.to(self.gpu)
        self.model.eval()
        pred = self.model(x)
        loss = self.criterion(pred,y).item()
        if self.min_loss<0 or self.min_loss>loss:
            self.min_loss = loss
            return True

#これ一行で全部動く
train_comps()