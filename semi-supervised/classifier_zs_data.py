import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):  # 继承 torch 的 Module（固定）
    def __init__(self, n_feature, n_hidden, n_output,num_of_hidden_layers):  # 定义层的信息，n_feature多少个输入, n_hidden每层神经元, n_output多少个输出
        super(Net, self).__init__()  # 继承 __init__ 功能（固定）
        self.num_of_hidden_layers = num_of_hidden_layers
        self.hidden = nn.Linear(n_feature, n_hidden)  # 定义隐藏层，线性输出
        self.hidden_list = nn.ModuleList([nn.Linear(n_hidden, n_hidden) for i in range(num_of_hidden_layers)])
        self.BN = nn.BatchNorm1d(n_feature)
        self.smooth = nn.Linear(n_hidden,int(n_hidden/4))
        self.predict = nn.Linear(int(n_hidden/4), n_output)  # 定义输出层线性输出
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):  # x是输入信息就是data，同时也是 Module 中的 forward 功能，定义神经网络前向传递的过程，把__init__中的层信息一个一个的组合起来
        x = self.BN(x)
        x = F.relu(self.hidden(x))
        x = self.dropout(x)
        for i in range(self.num_of_hidden_layers):
            x = F.relu(self.hidden_list[i](x))  # 定义激励函数(隐藏层的线性值)
            x = self.dropout(x)
            #x = self.batch_norm_list[i](x)
        x = F.relu(self.smooth(x))
        x = self.predict(x)  # 输出层，输出值
        return self.sigmoid(x).squeeze()
    
def train_epoch(train_loader,model,epoch,device,loss_fn,optimizer):
    model.train()
    
    for batch_idx,(batch_X,batch_Y) in enumerate(train_loader):
        global_step = len(train_loader)*epoch + batch_idx
        batch_X,batch_Y = batch_X.to(device),batch_Y.to(device)
        batch_pred_y = model(batch_X)
        loss = loss_fn(batch_pred_y,batch_Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
from torch.utils.data import DataLoader
import logging
import torch.utils.data as data
class Dataset_SRU(data.Dataset):
    def __init__(self,total_data):
        self.total_data = total_data
    def __getitem__(self, index):
        return self.total_data[index]['X'],self.total_data[index]['Y']
    def __len__(self):
        return len(self.total_data)
class Dataset_class(data.Dataset):
    def __init__(self,total_X,total_Y):
        self.total_X = total_X.numpy()
        self.total_Y = total_Y.numpy()
    def __getitem__(self, index):
        return self.total_X[index],self.total_Y[index]
    def __len__(self):
        return self.total_X.shape[0]
def collate_train(data):
    batch_X,batch_Y = zip(*data)
    batch_X = torch.tensor(np.array(batch_X),dtype=torch.float)
    batch_Y = torch.tensor(np.array(batch_Y),dtype=torch.float)
    return batch_X,batch_Y
def collate_fn(data):
    batch_X,batch_Y = zip(*data)
    batch_X = torch.tensor(batch_X,dtype=torch.float)
    batch_Y = torch.tensor(batch_Y,dtype=torch.float)
    return batch_X,batch_Y
def eval_epoch(val_loader,model,epoch,device,loss_fn,evaluate_score_fn):
    model.eval()
    batch_pred_Y = None
    for batch_idx,(batch_X,batch_Y) in enumerate(val_loader):
        batch_X,batch_Y = batch_X.to(device),batch_Y.to(device)
        batch_pred_Y = model(batch_X)
    score = evaluate_score_fn(batch_pred_Y,batch_Y)
    #print(batch_pred_Y,batch_Y)
    return score

import random
import numpy as np

import pandas as pd
def get_data():
    df = pd.read_csv('dataset/zd-data.CSV',encoding="gb2312")
    df = df.dropna(axis=0,how='any')
    total_X = np.array(df[['一反温度','二反温度','重量空速','反应压力','剂油比']],dtype = np.float32)
    total_Y = np.array(df.汽油收率,dtype = np.float32)
    return [{'X':total_X[i],'Y':int(total_Y[i]>39)} for i in range(len(total_X))]
import tqdm
def evaluate_score_fn(y_pred,Y):
    return sum([int(y_pred[i]>=0.5)==Y[i]for i in range(len(y_pred))])/len(Y)
if __name__=='__main__':
    seed_name = 12138
    random.seed(seed_name)
    np.random.seed(seed_name)
    torch.manual_seed(seed=seed_name)
    multi = True
    total_data = get_data()
    random.shuffle(total_data)
    select_indexes = int(len(total_data)*0.8)
    val_text_same = True
    if val_text_same:
        train_data,val_data,test_data = total_data[:select_indexes],total_data[select_indexes:],total_data[select_indexes:]
    else:
        test_indexs = int(len(total_data)*0.9)
        train_data,val_data,test_data = total_data[:select_indexes],total_data[select_indexes:test_indexs],total_data[test_indexs:]
    #n_epochs = 400
    train_dataset,val_dataset,test_dataset = Dataset_SRU(train_data),Dataset_SRU(val_data),Dataset_SRU(test_data)
    train_loader = DataLoader(train_dataset,batch_size = 32,shuffle=True,collate_fn=collate_train,drop_last=True)
    val_loader = DataLoader(val_dataset,batch_size = val_dataset.__len__(),shuffle=False,collate_fn=collate_train,drop_last=True)
    test_loader = DataLoader(test_dataset,batch_size = test_dataset.__len__(),shuffle=False,collate_fn=collate_train,drop_last=True)

    learn_rate_ls = [1e-5,5e-5,1e-4]
    num_of_hidden_layers_ls = [2,4,6,8,10]
    num_of_hidden_layers = 1
    n_hidden = 1024
    n_epochs = 100
    best_eval_score = 0
    model = Net(n_feature=5,n_output=1,n_hidden=n_hidden,num_of_hidden_layers=num_of_hidden_layers)
    loss_fn = nn.BCELoss()
    device = torch.device('cpu')
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay = 0.1)
    
    for epoch in tqdm.tqdm(range(n_epochs),colour = 'red'):
        train_epoch(train_loader,model,epoch,device,loss_fn,optimizer)
        with torch.no_grad():
            MSE_score = eval_epoch(val_loader,model,epoch,device,loss_fn,evaluate_score_fn)
            print('epoch:{},accuracy:{} '.format(epoch,MSE_score))
            if MSE_score >=  best_eval_score:
                best_eval_score = MSE_score
                checkpoint = model.state_dict()
                torch.save({'checkpoint':checkpoint},'model.ckpt')
    print('eval_best_score model0:{}'.format(best_eval_score))
    #writer.close()
    opt_dict = {
        'learning_rate':learning_rate,
        'num_of_hidden_layers':num_of_hidden_layers,
        'n_hidden':n_hidden}
    checkpoint = torch.load('model.ckpt')
    model.load_state_dict(checkpoint['checkpoint'])
    _0 = eval_epoch(test_loader,model,epoch,device,loss_fn,evaluate_score_fn)

