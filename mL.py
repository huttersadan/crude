import pandas as pd
import numpy as np
import random
import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import scipy.io as scio
def get_single_XandY(read_data_squeeze,write_data_squeeze):
    read_data_name_ls = [
    "TIC10701_PV",#控温1
    "TI10702_PV",#控温2
    "TI10703_PV" ,#控温3
    "TI10704_PV" ,#控温4
    "ATMFEED_PV" ,#进料流量和
    "draw1" ,#回流温度1
    "draw2" ,#回流温度2
    "draw3" ,#回流温度3
    "draw1r",#回流温度1r
    "draw2r" ,#回流温度2r
    "draw3r" ,#回流温度3r
    "ATOPKK_PV",#汽油100%
    "ATM1HK_PV" ,#煤油出溜点
    "ATM1KK_PV" ,#煤油100%
    "ATM1FLH_PV",#煤油闪点
    "ATM2KK_PV" ,#柴油95%
    "ATM3KK_PV" ,#AGO 95%
    ]
    read_data_name_dict = {}
    for i in range(len(read_data_name_ls)):
        read_data_name_dict[read_data_name_ls[i]] = i 
    write_data_name_ls = [
        'PI10701.PV', #压力
        'FIC10701.RSP',#出料1
        'FIC10702_RSP',#出料2
        'FIC10703_RSP',#出料3
        'TIC11300_PV',#温度
        'TIC_102_RSP',#控温
        'FIC10705_RSP',#回流量1
        'FIC10706_RSP',#回流量2
        'FIC10707_RSP',#回流量3
        'medium',
        'heavy',
        'light',
    ]
    write_data_name_dict = {}
    for i in range(len(write_data_name_ls)):
        write_data_name_dict[write_data_name_ls[i]] = i
    
    total_X = np.array([
        np.squeeze(read_data_squeeze[read_data_name_dict['draw1r']])*np.squeeze(write_data_squeeze[write_data_name_dict['FIC10705_RSP']]/235),
        np.squeeze(read_data_squeeze[read_data_name_dict['draw2r']])*np.squeeze(write_data_squeeze[write_data_name_dict['FIC10706_RSP']]/235),
        np.squeeze(read_data_squeeze[read_data_name_dict['draw3r']])*np.squeeze(write_data_squeeze[write_data_name_dict['FIC10707_RSP']]/235),        
        np.squeeze(read_data_squeeze[read_data_name_dict['draw1']])*np.squeeze(write_data_squeeze[write_data_name_dict['FIC10705_RSP']]/235),
        np.squeeze(read_data_squeeze[read_data_name_dict['draw2']])*np.squeeze(write_data_squeeze[write_data_name_dict['FIC10706_RSP']]/235),
        np.squeeze(read_data_squeeze[read_data_name_dict['draw3']])*np.squeeze(write_data_squeeze[write_data_name_dict['FIC10707_RSP']]/235),                
        np.squeeze(read_data_squeeze[read_data_name_dict['draw1r']]),
        np.squeeze(read_data_squeeze[read_data_name_dict['draw2r']]),
        np.squeeze(read_data_squeeze[read_data_name_dict['draw3r']]),
        np.squeeze(read_data_squeeze[read_data_name_dict['draw1']]),
        np.squeeze(read_data_squeeze[read_data_name_dict['draw2']]),
        np.squeeze(read_data_squeeze[read_data_name_dict['draw3']]),
        np.squeeze(write_data_squeeze[write_data_name_dict['FIC10705_RSP']]/235),
        np.squeeze(write_data_squeeze[write_data_name_dict['FIC10706_RSP']]/235),
        np.squeeze(write_data_squeeze[write_data_name_dict['FIC10707_RSP']]/235),
        np.squeeze(read_data_squeeze[read_data_name_dict['TIC10701_PV']]),# 四个温度
        np.squeeze(read_data_squeeze[read_data_name_dict['TI10702_PV']]),# 四个温度
        np.squeeze(read_data_squeeze[read_data_name_dict['TI10703_PV']]),# 四个温度
        np.squeeze(read_data_squeeze[read_data_name_dict['TI10704_PV']]),# 四个温度
        np.squeeze(write_data_squeeze[write_data_name_dict['PI10701.PV']]),#压力
        np.squeeze(write_data_squeeze[write_data_name_dict['TIC11300_PV']]),#温度
        np.squeeze(write_data_squeeze[write_data_name_dict['TIC_102_RSP']]),#控温
        np.squeeze(write_data_squeeze[write_data_name_dict['FIC10701.RSP']]/235),
        np.squeeze(write_data_squeeze[write_data_name_dict['FIC10702_RSP']]/235),
        np.squeeze(write_data_squeeze[write_data_name_dict['FIC10703_RSP']]/235),
    ])
    total_Y = np.array([
        read_data_squeeze[read_data_name_dict['ATOPKK_PV']],
        read_data_squeeze[read_data_name_dict['ATM1HK_PV']],
        read_data_squeeze[read_data_name_dict['ATM1KK_PV']],
        read_data_squeeze[read_data_name_dict['ATM1FLH_PV']],
        read_data_squeeze[read_data_name_dict['ATM2KK_PV']],
        read_data_squeeze[read_data_name_dict['ATM3KK_PV']],
    ])
    
    total_X = np.squeeze(total_X)
    total_Y = np.squeeze(total_Y)
    total_Y = total_Y[0:1,:]
    return total_X,total_Y
import matplotlib.pyplot as plt
def data_provider(multi):
    
    if not multi:
        path_dataset = 'USD_OS 3/softsensor_data3'
        read_data = scio.loadmat(path_dataset+'/read.mat')
        write_data = scio.loadmat(path_dataset+'/write.mat')
        read_data_squeeze = read_data['rs_read_data'][0][0]
        write_data_squeeze = write_data['write_data'][0][0]
        # write_refine_data_squeeze = []
        # for inst in write_data_squeeze:
        #     write_refine_data_squeeze.append(inst[:500])
        # write_data_squeeze = write_refine_data_squeeze
        total_X,total_Y = get_single_XandY(read_data_squeeze,write_data_squeeze)
        corr_ls = [np.corrcoef(total_X[i],total_Y[0]) for i in range(len(total_X))]

        print(corr_ls)
        plt_data = copy.deepcopy(total_X[7])[:200]
        total_Y[0]
        x = [i for i in range(len(plt_data))]
        plt_data1 = plt_data+ np.random.randn(len(x))
        plt.plot(x,plt_data,label = 'Original temperature')
        plt.plot(x,plt_data1,label = 'New temperature')
        
        plt.ylabel('Temperature/℃')
        plt.legend()
        plt.savefig('2_3_2.png')
        plt.show()
        
        print(total_X.shape,total_Y.shape)
    else:
        path_dataset = 'USD_OS 3/softsensor_data2'
        read_data_ls = [scio.loadmat(path_dataset+'/read{}.mat'.format(i)) for i in range(1,7)]
        write_data_ls = [scio.loadmat(path_dataset+'/write{}.mat'.format(i)) for i in range(1,7)] 
        read_data_squeeze_ls = [read_data_ls[i]['rs_read_data'][0][0] for i in range(0,6)]
        write_data_squeeze_ls = [write_data_ls[i]['write_data'][0][0] for i in range(0,6)]
        total_X,total_Y = [],[]
        for i in range(0,6):
            single_X,single_Y = get_single_XandY(read_data_squeeze_ls[i],write_data_squeeze_ls[i])
            total_X.append(single_X)
            total_Y.append(single_Y)
        total_X = np.concatenate(total_X,axis=1)
        total_Y = np.concatenate(total_Y,axis=1)
       
        
    #print(total_X.shape,total_Y.shape)
    rs_dict_ls = []
    for i in range(total_X.shape[1]):
        rs_dict_ls.append({'X':total_X[:,i],'Y':total_Y[:,i]})
    return rs_dict_ls
import torch.utils.data as data
class Dataset_SRU(data.Dataset):
    def __init__(self,total_data):
        self.total_data = total_data
    def __getitem__(self, index):
        return self.total_data[index]['X'],self.total_data[index]['Y']
    def __len__(self):
        return len(self.total_data)
from torch.utils.data import DataLoader

class Dataset_class(data.Dataset):
    def __init__(self,total_X,total_Y):
        self.total_X = total_X.cpu().numpy()
        self.total_X = (self.total_X - np.mean(self.total_X))/np.std(self.total_X)
        self.total_Y = total_Y.cpu().numpy()
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
def eval_epoch(val_loader,model,epoch,device,loss_fn,evaluate_score_fn,type_model):
    model.eval()
    batch_pred_Y = None
    for batch_idx,(batch_X,batch_Y) in enumerate(val_loader):
        batch_X,batch_Y = batch_X.to(device),batch_Y.to(device)
        batch_pred_Y = model(batch_X)
    #print('batch_pred_Y:{},batch_Y:{}'.format(batch_pred_Y.shape,batch_Y.shape))
    i = random.randint(0,len(batch_Y)-1)
    print('\n i = {}\n pred:{},\n orginal:{}'.format(i,batch_pred_Y[1],batch_Y[1]))
    score = evaluate_score_fn(batch_pred_Y,batch_Y)
    logging.info('eval_score_model{}:{}'.format(type_model,score))

    return score


def train_epoch(train_loader,model,epoch,device,loss_fn,optimizer,writer,type_model,early_print):
    model.train()
    
    for batch_idx,(batch_X,batch_Y) in enumerate(train_loader):
        global_step = len(train_loader)*epoch + batch_idx
        batch_X,batch_Y = batch_X.to(device),batch_Y.to(device)
        
        batch_pred_y = model(batch_X)
        loss = loss_fn(batch_pred_y,batch_Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch >=early_print:
            writer.add_scalar("Train,model{}/MSELoss".format(type_model), loss,global_step)
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import argparse
import time
import json
import logging
from classfymodel import linear_model_pytorch
# class linear_model_pytorch(nn.Module):
#     def __init__(self,in_hsz,hidden_size,out_hsz, dropout=0.1):
#         super(linear_model_pytorch,self).__init__()
#         self.linear1 = nn.Linear(in_hsz,hidden_size)
#         self.BN = nn.BatchNorm1d(in_hsz)
        
#         self.smooth = nn.Linear(hidden_size,int(hidden_size/4))
#         self.predict = nn.Linear(int(hidden_size/4),out_hsz)
#         self.dropout1 = nn.Dropout(dropout)
#     def forward(self,x,y):
#         #print(x[1])
#         x = self.BN(x)
#         x = F.relu(self.linear1(x))
#         x = self.dropout1(x)
#         x = F.relu(self.smooth(x))
#         x = self.predict(x)
#         y_pred = F.softmax(x, dim=-1)
#         loss_fn = nn.CrossEntropyLoss()
#         loss = loss_fn(x,y.type(torch.long))
#         return loss,y_pred

class Net(nn.Module):  # 继承 torch 的 Module（固定）
    def __init__(self, n_feature, n_hidden, n_output,num_of_hidden_layers,dropout_prob):  # 定义层的信息，n_feature多少个输入, n_hidden每层神经元, n_output多少个输出
        super(Net, self).__init__()  # 继承 __init__ 功能（固定）
        self.num_of_hidden_layers = num_of_hidden_layers
        self.hidden = nn.Linear(n_feature, n_hidden)  # 定义隐藏层，线性输出
        self.hidden_list = nn.ModuleList([nn.Linear(n_hidden, n_hidden) for i in range(num_of_hidden_layers)])
        self.BN = nn.BatchNorm1d(n_feature)
        self.smooth = nn.Linear(n_hidden,int(n_hidden/4))
        self.predict = nn.Linear(int(n_hidden), n_output)  # 定义输出层线性输出
        self.dropout = nn.Dropout(dropout_prob)
        self.dropout_list = nn.ModuleList([nn.Dropout(dropout_prob) for i in range(num_of_hidden_layers)])
    def forward(self, x):  # x是输入信息就是data，同时也是 Module 中的 forward 功能，定义神经网络前向传递的过程，把__init__中的层信息一个一个的组合起来
        x = self.BN(x)
        x = F.relu(self.hidden(x))
        x = self.dropout(x)
        for i in range(self.num_of_hidden_layers):
            x = x + F.relu(self.hidden_list[i](x))  # 定义激励函数(隐藏层的线性值)
            x = self.dropout_list[i](x)
        x = self.predict(x)  # 输出层，输出值
        return x
import copy
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
if __name__=='__main__':
    seed_name = 12138
    random.seed(seed_name)
    np.random.seed(seed_name)
    torch.manual_seed(seed=seed_name)
    multi = False
    total_data = data_provider(multi)
    random.shuffle(total_data)
    X,y = [total_data[i]['X'] for i in range(len(total_data))],[total_data[i]['Y'] for i in range(len(total_data))]
    X_train, X_test, y_train, y_test \
        = train_test_split(X, y, test_size=0.3, random_state=0)
    #model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    model = PLSRegression(n_components=18)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse=  np.sqrt(mse)
    print("Mean squared error: %.2f" % rmse)
    df_dict = {'n_components':[],'RMSE':[]}
    for i in range(1,25):
        model = PLSRegression(n_components=i)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse=  np.sqrt(mse)
        
        print("Mean squared error: %.2f" % rmse)
        df_dict['n_components'].append(i)
        df_dict['RMSE'].append(rmse)
    df = pd.DataFrame(df_dict)
    fig, ax = plt.subplots()
    #for i in range(2,5):
    #     ax.scatter(df[df['min_samples_leaf'] == msl]['max_depth'], df[df['min_samples_leaf'] == msl]['RMSE'], cmap='viridis',label = "min_samples_leaf = {}".format(msl))
    ax.scatter(df['n_components'],df['RMSE'],cmap='viridis')
    ax.legend()
    ax.set_xlabel('max_depth')
    ax.set_ylabel('RMSE')
    ax.set_title('RMSE vs. max_depth')
    plt.show()
    # df_dict = {'max_depth':[],'min_samples_leaf':[],'RMSE':[]}
    # for md in range(2,10,1):
    #     for msl in range(2,5):
    #         model = DecisionTreeRegressor(max_depth = md,min_samples_leaf=msl)
    #         #model = RandomForestRegressor(n_estimators=100, max_depth=3)
    #         model.fit(X_train, y_train)
    #         y_pred = model.predict(X_test)
    #         mse = mean_squared_error(y_test, y_pred)
    #         df_dict['max_depth'].append(md)
    #         df_dict['min_samples_leaf'].append(msl)
    #         rmse=  np.sqrt(mse)
    #         df_dict['RMSE'].append(mse)
        
    #         print("Mean squared error: %.2f" % mse)
    # df = pd.DataFrame(df_dict)
    # y = df['RMSE']
    # # 绘制散点图
    # fig, ax = plt.subplots()
    # for msl in range(2,5):
    #     ax.scatter(df[df['min_samples_leaf'] == msl]['max_depth'], df[df['min_samples_leaf'] == msl]['RMSE'], cmap='viridis',label = "min_samples_leaf = {}".format(msl))
    # ax.legend()
    # ax.set_xlabel('max_depth')
    # ax.set_ylabel('RMSE')
    # ax.set_title('RMSE vs. max_depth')
    # plt.show()

    # fig, ax = plt.subplots()
    # for md in range(2,10,2):
    #     ax.scatter(df[df['max_depth'] == md]['min_samples_leaf'], df[df['max_depth'] == md]['RMSE'], cmap='viridis',label = "max_depth = {}".format(md))
    # ax.legend()
    # ax.set_xlabel('min_samples_leaf')
    # ax.set_ylabel('RMSE')
    # ax.set_title('RMSE vs. min_samples_leaf')
    # plt.show()