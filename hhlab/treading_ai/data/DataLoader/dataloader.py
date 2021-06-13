import torch
import torch.nn as nn
import numpy as np 
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
from torch import FloatTensor
from sklearn import preprocessing as prp 

class DataLoader():
    def __init__(self,com_code,dir_path):
        file_path = dir_path+str(com_code)+'.csv'
        self.raw = pd.read_csv(
            file_path,
            index_col = "date",
            parse_dates = True
        )
        self.device = 'cuda'
    def split(self,split_date):
        self.train = np.array(self.raw[:split_date])
        self.eval = np.array(self.raw[split_date:])

    def make_traindata(self,norm,n_w):
        normed_train = mmscaler.fit_transform(self.train)
        seq = []
        target = []
        for i in range(len(self.train)-n_w):
            seq.append(normed_train[i:i+n_w])
            target.append([self.train[i+n_w][0]])
        seq_t = FloatTensor(seq)
        target_t = FloatTensor(target)
        seq_t_t= (seq_t.permute(1,0,-1)).to(self.device)
        target_t_t= (target_t.permute(0,1)).to(self.device)
        return seq_t_t,target_t_t

    def make_validationdata(self):
        pass
        