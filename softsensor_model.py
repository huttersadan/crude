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
    # print(read_data_squeeze[read_data_name_dict['ATM1HK_PV']][0])
    # print(read_data_squeeze[read_data_name_dict['ATM1KK_PV']][0])
    # print(read_data_squeeze[read_data_name_dict['ATM1FLH_PV']][0])
    # print(read_data_squeeze[read_data_name_dict['ATM1HK_PV']][1])
    # print(read_data_squeeze[read_data_name_dict['ATM1KK_PV']][1])
    # print(read_data_squeeze[read_data_name_dict['ATM1FLH_PV']][1])
    # print(np.squeeze(write_data_squeeze[write_data_name_dict['medium']])[1])
    # print(np.squeeze(write_data_squeeze[write_data_name_dict['heavy']])[1])
    # print(np.squeeze(write_data_squeeze[write_data_name_dict['light']])[1])
    # print(np.squeeze(write_data_squeeze[write_data_name_dict['medium']])[2])
    # print(np.squeeze(write_data_squeeze[write_data_name_dict['heavy']])[2])
    # print(np.squeeze(write_data_squeeze[write_data_name_dict['light']])[2])
    # total_X = np.array([
    #     np.squeeze((read_data_squeeze[read_data_name_dict['draw1r']]-read_data_squeeze[read_data_name_dict['draw1']])*write_data_squeeze[write_data_name_dict['FIC10705_RSP']]/235),
    #     np.squeeze((read_data_squeeze[read_data_name_dict['draw2r']]-read_data_squeeze[read_data_name_dict['draw2']])*write_data_squeeze[write_data_name_dict['FIC10706_RSP']]/235),
    #     np.squeeze((read_data_squeeze[read_data_name_dict['draw3r']]-read_data_squeeze[read_data_name_dict['draw3']])*write_data_squeeze[write_data_name_dict['FIC10707_RSP']]/235),
    #     np.squeeze(read_data_squeeze[read_data_name_dict['TIC10701_PV']]),# 四个温度
    #     np.squeeze(read_data_squeeze[read_data_name_dict['TI10702_PV']]),# 四个温度
    #     np.squeeze(read_data_squeeze[read_data_name_dict['TI10703_PV']]),# 四个温度
    #     np.squeeze(read_data_squeeze[read_data_name_dict['TI10704_PV']]),# 四个温度
    #     np.squeeze(write_data_squeeze[write_data_name_dict['PI10701.PV']]),#压力
    #     np.squeeze(write_data_squeeze[write_data_name_dict['TIC11300_PV']]),#温度
    #     np.squeeze(write_data_squeeze[write_data_name_dict['TIC_102_RSP']]),#控温
    #     np.squeeze(write_data_squeeze[write_data_name_dict['medium']]/235),#进料
    #     np.squeeze(write_data_squeeze[write_data_name_dict['heavy']]/235),#进料
    #     np.squeeze(write_data_squeeze[write_data_name_dict['light']]/235)#进料
    # ])
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
        
        plt_data = copy.deepcopy(total_X[6])[:50]
        x = [i for i in range(len(plt_data))]
        plt_data1 = plt_data+ np.random.randn(len(x))*2
        plt.plot(x,plt_data,label = 'Original temperature')
        plt.plot(x,plt_data1,label = 'New temperature')
        
        plt.ylabel('Temperature/℃')
        plt.legend()
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
    def __init__(self, n_feature, n_hidden, n_output,num_of_hidden_layers):  # 定义层的信息，n_feature多少个输入, n_hidden每层神经元, n_output多少个输出
        super(Net, self).__init__()  # 继承 __init__ 功能（固定）
        self.num_of_hidden_layers = num_of_hidden_layers
        self.hidden = nn.Linear(n_feature, n_hidden)  # 定义隐藏层，线性输出
        self.hidden_list = nn.ModuleList([nn.Linear(n_hidden, n_hidden) for i in range(num_of_hidden_layers)])
        self.BN = nn.BatchNorm1d(n_feature)
        self.smooth = nn.Linear(n_hidden,int(n_hidden/4))
        self.predict = nn.Linear(int(n_hidden), n_output)  # 定义输出层线性输出
        self.dropout = nn.Dropout(0.07)
        
    def forward(self, x):  # x是输入信息就是data，同时也是 Module 中的 forward 功能，定义神经网络前向传递的过程，把__init__中的层信息一个一个的组合起来
        x = self.BN(x)
        x = F.relu(self.hidden(x))
        x = self.dropout(x)
        for i in range(self.num_of_hidden_layers):
            x = F.relu(self.hidden_list[i](x))  # 定义激励函数(隐藏层的线性值)
            #x = self.dropout(x)
            #x = self.batch_norm_list[i](x)
        #x = F.relu(self.smooth(x))
        x = self.predict(x)  # 输出层，输出值
        return x
import copy
if __name__=='__main__':
    seed_name = 12138
    random.seed(seed_name)
    np.random.seed(seed_name)
    torch.manual_seed(seed=seed_name)
    multi = False
    total_data = data_provider(multi)
    random.shuffle(total_data)
    select_indexes = int(len(total_data)*0.8)
    val_text_same = False
    if val_text_same:
        train_data,val_data,test_data = total_data[:select_indexes],total_data[select_indexes:],total_data[select_indexes:]
    else:
        test_indexs = int(len(total_data)*0.9)
        train_data,val_data,test_data = total_data[:select_indexes],total_data[select_indexes:test_indexs],total_data[test_indexs:]
    train_dataset,val_dataset,test_dataset = Dataset_SRU(train_data),Dataset_SRU(val_data),Dataset_SRU(test_data)
    #train_loader = DataLoader(train_dataset,batch_size = 32,shuffle=True,collate_fn=collate_train,drop_last=True)
    total_dataset = Dataset_SRU(total_data)
    total_loader = DataLoader(total_dataset,batch_size = total_dataset.__len__(),shuffle = False,collate_fn=collate_train)

    learn_rate_ls = [1e-5,5e-5,1e-4]
    num_of_hidden_layers_ls = [2,4,6,8,10]
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id',help = 'experiment_id',type=str,default='test_')
    parser.add_argument('--num_of_hidden_layers',help = 'num_of_hidden_layers',type=int,default=4)
    parser.add_argument('--n_hidden',help = 'hidden size 4x ',type = int,default=128)
    parser.add_argument('--batch_size',help = 'batch_size',type = int,default=32)
    parser.add_argument('--n_epoch0',type=int,default=50,help = 'number of epochs0')
    parser.add_argument('--n_epoch1',type=int,default=50,help = 'number of epochs1')
    parser.add_argument('--n_epoch2',type=int,default=50,help = 'number of epochs2')
    parser.add_argument('--learning_rate0',help = 'learning_rate',type=float,default=1e-5)
    parser.add_argument('--learning_rate1',help = 'learning_rate',type=float,default=1e-5)
    parser.add_argument('--learning_rate2',help = 'learning_rate',type=float,default=1e-5)
    parser.add_argument('--learning_rate_total',help = 'learning_rate',type=float,default=1e-5)
    parser.add_argument('--n_epoch_total',type=int,default=50,help = 'number of epochs_total')
    parser.add_argument('--early_print',type = int,default = 30,help = 'when to write in writer')
    opt = parser.parse_args()
    opt.result_dir = 'results/' + opt.exp_id +time.strftime("%Y_%m_%d_%H_%M_%S")
    
    train_loader = DataLoader(train_dataset,batch_size = train_dataset.__len__(),shuffle=False,collate_fn=collate_train,drop_last=True)
    val_loader = DataLoader(val_dataset,batch_size = val_dataset.__len__(),shuffle=False,collate_fn=collate_train,drop_last=True)
    test_loader = DataLoader(test_dataset,batch_size = test_dataset.__len__(),shuffle=False,collate_fn=collate_train,drop_last=True)
    
    device = torch.device('cpu')
    classify_model = linear_model_pytorch(3,100,3).to(device)
    classify_model.load_state_dict(torch.load('model1_dict.ckpt3')['model'])
    classify_model.eval()
    for batch_idx,(batch_X,batch_Y) in enumerate(total_loader):
        batch_x = (batch_X[:,[-3,-2,-1]].to(device))* 1000
        batch_y = torch.zeros(batch_x.shape[0]).to(device)
        _,y_pred = classify_model(batch_x,batch_y)
        y_pred = torch.argmax(y_pred,dim = -1)
    total_X0 = copy.deepcopy(batch_X[y_pred == 0])
    total_X1 = copy.deepcopy(batch_X[y_pred == 1])
    total_X2 = copy.deepcopy(batch_X[y_pred == 2])
    total_Y0 = copy.deepcopy(batch_Y[y_pred == 0])
    total_Y1 = copy.deepcopy(batch_Y[y_pred == 1])
    total_Y2 = copy.deepcopy(batch_Y[y_pred == 2])
    total_X = copy.deepcopy(batch_X)
    total_Y = copy.deepcopy(batch_Y)
    total_X0 = copy.deepcopy(total_X0[:150])
    total_Y0 = copy.deepcopy(total_Y0[:150])
    X_index = random.sample([i for i in range(len(total_X))],int(len(total_X)*0.8))
    X0_index = random.sample([i for i in range(len(total_X0))],int(len(total_X0)*0.8))
    X1_index = random.sample([i for i in range(len(total_X1))],int(len(total_X1)*0.8))
    X2_index = random.sample([i for i in range(len(total_X2))],int(len(total_X2)*0.8))
    no_X_index = []
    for i in range(len(X_index)):
        if i not in X_index:
            no_X_index.append(i)
    no_X0_index = []
    for i in range(len(X0_index)):
        if i not in X0_index:
            no_X0_index.append(i)
    no_X1_index = []
    for i in range(len(X1_index)):
        if i not in X1_index:
            no_X1_index.append(i)
    no_X2_index = []
    for i in range(len(X2_index)):
        if i not in X2_index:
            no_X2_index.append(i)
    train_X = total_X[X_index]
    train_Y = total_Y[X_index]
    train_X0 = total_X0[X0_index]
    train_Y0 = total_Y0[X0_index]
    train_X1 = total_X1[X1_index]
    train_Y1 = total_Y1[X1_index]
    train_X2 = total_X2[X2_index]
    train_Y2 = total_Y2[X2_index]
    val_X = total_X[no_X_index]
    val_Y = total_Y[no_X_index]
    val_X0 = total_X0[no_X0_index]
    val_Y0 = total_Y0[no_X0_index]
    val_X1 = total_X1[no_X1_index]
    val_Y1 = total_Y1[no_X1_index]
    val_X2 = total_X2[no_X2_index]
    val_Y2 = total_Y2[no_X2_index]
    test_X = val_X
    test_Y = val_Y
    test_X0 = val_X0
    test_Y0 = val_Y0
    test_X1 = val_X1
    test_Y1 = val_Y1
    test_X2 = val_X2
    test_Y2 = val_Y2
    train_X = torch.tensor(np.concatenate((train_X0,train_X1,train_X2)))
    train_Y = torch.tensor(np.concatenate((train_Y0,train_Y1,train_Y2)))
    val_X = torch.tensor(np.concatenate((val_X0,val_X1,val_X2)))
    val_Y = torch.tensor(np.concatenate((val_Y0,val_Y1,val_Y2)))
    test_X = torch.tensor(np.concatenate((test_X0,test_X1,test_X2)))
    test_Y = torch.tensor(np.concatenate((test_Y0,test_Y1,test_Y2)))
    total_X = torch.cat((train_X,val_X))
    total_Y = torch.cat((train_Y,val_Y))
    print('shape:{}'.format(total_X.shape))
    X_index = random.sample([i for i in range(len(total_X))],int(len(total_X)*0.8))
    no_X_index = []
    for i in range(len(X_index)):
        if i not in X_index:
            no_X_index.append(i)
    train_X = total_X[X_index]
    train_Y = total_Y[X_index]
    val_X = total_X[no_X_index]
    val_Y = total_Y[no_X_index]
    test_X = val_X
    test_Y = val_Y
    # train_X = train_X1
    # train_Y = train_Y1
    
    # val_X = val_X1
    # val_Y = val_Y1
    
    # test_X = test_X1
    # test_Y = test_Y1
    print('\ndata0 size:{}\ndata1 size:{}\ndata2 size:{}\n'.format(len(total_X0),len(total_X1),len(total_X2)))
    print('\ntotal_data size:{}'.format(train_X.shape))
    print('\ntotal_data size:{}'.format(train_Y.shape))
    # for batch_idx,(batch_X,batch_Y) in enumerate(train_loader):
    #     batch_x = (batch_X[:,[-3,-2,-1]].to(device))* 1000
    #     print(batch_x)
    #     # b = torch.sum(batch_x,axis = 1)
    #     # b = 1/b
    #     # c = torch.cat((b.view(-1,1),b.view(-1,1),b.view(-1,1)),axis = 1)
    #     # batch_x = batch_x * c
    #     batch_y = torch.zeros(batch_x.shape[0]).to(device)
    #     _,y_pred = classify_model(batch_x,batch_y)
    #     y_pred = torch.argmax(y_pred,dim = -1)
    # #print('---------------y_pred:{}'.format(y_pred))
        
    
    # train_X0 = copy.deepcopy(batch_X[y_pred == 0])
    # train_X1 = copy.deepcopy(batch_X[y_pred == 1])
    # train_X2 = copy.deepcopy(batch_X[y_pred == 2])
    # train_Y0 = copy.deepcopy(batch_Y[y_pred == 0])
    # train_Y1 = copy.deepcopy(batch_Y[y_pred == 1])
    # train_Y2 = copy.deepcopy(batch_Y[y_pred == 2])
    # train_X = copy.deepcopy(batch_X)
    # train_Y = copy.deepcopy(batch_Y)
    # for batch_idx,(batch_X,batch_Y) in enumerate(val_loader):
    #     batch_x = (batch_X[:,[-3,-2,-1]].to(device))* 1000
    #     # b = torch.sum(batch_x,axis = 1)
    #     # b = 1/b
    #     # c = torch.cat((b.view(-1,1),b.view(-1,1),b.view(-1,1)),axis = 1)
    #     # batch_x = batch_x * c
    #     batch_y = torch.zeros(batch_x.shape[0]).to(device)
    #     _,y_pred = classify_model(batch_x,batch_y)
    #     y_pred = torch.argmax(y_pred,dim = -1)
    #     #print(y_pred)
    # val_X0 = copy.deepcopy(batch_X[y_pred == 0])
    # val_X1 = copy.deepcopy(batch_X[y_pred == 1])
    # val_X2 = copy.deepcopy(batch_X[y_pred == 2])
    # val_Y0 = copy.deepcopy(batch_Y[y_pred == 0])
    # val_Y1 = copy.deepcopy(batch_Y[y_pred == 1])
    # val_Y2 = copy.deepcopy(batch_Y[y_pred == 2])
    # val_X = copy.deepcopy(batch_X)
    # val_Y = copy.deepcopy(batch_Y)
    # for batch_idx,(batch_X,batch_Y) in enumerate(test_loader):
    #     batch_x = (batch_X[:,[-3,-2,-1]].to(device))* 1000
    #     # b = torch.sum(batch_x,axis = 1)
    #     # b = 1/b
    #     # c = torch.cat((b.view(-1,1),b.view(-1,1),b.view(-1,1)),axis = 1)
    #     # batch_x = batch_x * c
    #     batch_y = torch.zeros(batch_x.shape[0]).to(device)
    #     _,y_pred = classify_model(batch_x,batch_y)
    #     y_pred = torch.argmax(y_pred,dim = -1)
    #     #print(y_pred)
    # test_X0 = copy.deepcopy(batch_X[y_pred == 0])
    # test_X1 = copy.deepcopy(batch_X[y_pred == 1])
    # test_X2 = copy.deepcopy(batch_X[y_pred == 2])
    # test_Y0 = copy.deepcopy(batch_Y[y_pred == 0])
    # test_Y1 = copy.deepcopy(batch_Y[y_pred == 1])
    # test_Y2 = copy.deepcopy(batch_Y[y_pred == 2])
    # test_X = copy.deepcopy(batch_X)
    # test_Y = copy.deepcopy(batch_Y)
    
    
    train_loader0 = DataLoader(Dataset_class(train_X0,train_Y0),batch_size = opt.batch_size,shuffle=True,collate_fn=collate_fn,drop_last=True)
    train_loader1 = DataLoader(Dataset_class(train_X1,train_Y1),batch_size = opt.batch_size,shuffle=True,collate_fn=collate_fn,drop_last=True)
    train_loader2 = DataLoader(Dataset_class(train_X2,train_Y2),batch_size = opt.batch_size,shuffle=True,collate_fn=collate_fn,drop_last=True)
    train_loader = DataLoader(Dataset_class(train_X,train_Y),batch_size = opt.batch_size,shuffle=True,collate_fn=collate_fn,drop_last=True)
    
    
    val_loader0 = DataLoader(Dataset_class(val_X0,val_Y0),batch_size = val_X0.shape[0],shuffle=False,collate_fn=collate_fn,drop_last=True)
    val_loader1 = DataLoader(Dataset_class(val_X1,val_Y1),batch_size = val_X1.shape[0],shuffle=False,collate_fn=collate_fn,drop_last=True)
    val_loader2 = DataLoader(Dataset_class(val_X2,val_Y2),batch_size = val_X2.shape[0],shuffle=False,collate_fn=collate_fn,drop_last=True)
    val_loader = DataLoader(Dataset_class(val_X,val_Y),batch_size = val_X.shape[0],shuffle=False,collate_fn=collate_fn,drop_last=True)
    
    
    test_loader0 = DataLoader(Dataset_class(test_X0,test_Y0),batch_size = test_X0.shape[0],shuffle=False,collate_fn=collate_fn,drop_last=True)
    test_loader1 = DataLoader(Dataset_class(test_X1,test_Y1),batch_size = test_X1.shape[0],shuffle=False,collate_fn=collate_fn,drop_last=True)
    test_loader2 = DataLoader(Dataset_class(test_X2,test_Y2),batch_size = test_X2.shape[0],shuffle=False,collate_fn=collate_fn,drop_last=True)
    test_loader = DataLoader(Dataset_class(test_X,test_Y),batch_size = test_X.shape[0],shuffle=False,collate_fn=collate_fn,drop_last=True)
    
    model0 = Net(n_feature=25,n_output=1,n_hidden=opt.n_hidden,num_of_hidden_layers=opt.num_of_hidden_layers)
    model1 = Net(n_feature=25,n_output=1,n_hidden=opt.n_hidden,num_of_hidden_layers=opt.num_of_hidden_layers)
    model2 = Net(n_feature=25,n_output=1,n_hidden=opt.n_hidden,num_of_hidden_layers=opt.num_of_hidden_layers)
    model_total = Net(n_feature=25,n_output=1,n_hidden=opt.n_hidden,num_of_hidden_layers=opt.num_of_hidden_layers)
    model0 = model0.to(device)
    model1 = model1.to(device)
    model2 = model2.to(device)
    model_total = model_total.to(device)
    loss_fn = nn.MSELoss()
    
    #loss_fn = nn.KLDivLoss()
    optimizer0 = torch.optim.Adam(model0.parameters(),lr=opt.learning_rate0,weight_decay = 0.1)
    if os.path.exists(opt.result_dir) == 0:
        os.mkdir(opt.result_dir)
    opt.tensorboard_log_dir = opt.result_dir+'/tensorboard_log_dir'
    # opt.tensorboard_log_dir = 'results/test'
    writer = SummaryWriter(opt.tensorboard_log_dir)
    evaluate_score_fn = nn.MSELoss()
    
    logging.basicConfig(level=logging.INFO,
                    filename=opt.result_dir + '/evaluate.log',
                    filemode='a',
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    best_eval_score = 1e10
    for epoch in tqdm.tqdm(range(opt.n_epoch0),colour = 'red'):
        train_epoch(train_loader0,model0,epoch,device,loss_fn,optimizer0,writer,type_model = 0,early_print=opt.early_print)
        with torch.no_grad():
            MSE_score = eval_epoch(val_loader0,model0,epoch,device,loss_fn,evaluate_score_fn,type_model=0)
            if epoch >= opt.early_print:
                writer.add_scalar('Eval_model0/MSE:',MSE_score,epoch)
            if MSE_score<= best_eval_score:
                best_eval_score = MSE_score
                checkpoint = model0.state_dict()
                torch.save({'checkpoint':checkpoint},opt.result_dir+'/model0.ckpt')
    print('eval_best_score model0:{}'.format(best_eval_score))
    #writer.close()
    
    checkpoint = torch.load(opt.result_dir+'/model0.ckpt')
    model0.load_state_dict(checkpoint['checkpoint'])
    _0 = eval_epoch(test_loader0,model0,epoch,device,loss_fn,evaluate_score_fn,type_model = 0)

    opt_dict = {
        'learning_rate0':opt.learning_rate0,
        'learning_rate1':opt.learning_rate1,
        'learning_rate2':opt.learning_rate2,
        'learning_rate_total':opt.learning_rate_total,
        'num_of_hidden_layers':opt.num_of_hidden_layers,
        'n_hidden':opt.n_hidden,
        'n_epoch0':opt.n_epoch0,
        'n_epoch1':opt.n_epoch1,
        'n_epoch2':opt.n_epoch2,
        'n_epoch_total':opt.n_epoch_total
        }
    opt.config_path = opt.result_dir + '/config.json'
    with open(opt.config_path,'w+') as file:
        json.dump(opt_dict,file)
    best_eval_score = 1e10
    optimizer1 = torch.optim.Adam(model1.parameters(),lr=opt.learning_rate1,weight_decay = 0.1)
    for epoch in tqdm.tqdm(range(opt.n_epoch1),colour = 'red'):
        train_epoch(train_loader1,model1,epoch,device,loss_fn,optimizer1,writer,type_model = 1,early_print=opt.early_print)
        with torch.no_grad():
            MSE_score = eval_epoch(val_loader1,model1,epoch,device,loss_fn,evaluate_score_fn,type_model = 1)
            if epoch >= opt.early_print:
                writer.add_scalar('Eval_model1/MSE:',MSE_score,epoch)
            if MSE_score<= best_eval_score:
                best_eval_score = MSE_score
                checkpoint = model1.state_dict()
                torch.save({'checkpoint':checkpoint},opt.result_dir+'/model1.ckpt')
    print('eval_best_score model1:{}'.format(best_eval_score))

    checkpoint = torch.load(opt.result_dir+'/model1.ckpt')
    model1.load_state_dict(checkpoint['checkpoint'])
    _1 = eval_epoch(test_loader1,model1,epoch,device,loss_fn,evaluate_score_fn,type_model = 1)
# use logging
    
    best_eval_score = 1e10
    optimizer2 = torch.optim.Adam(model2.parameters(),lr=opt.learning_rate2,weight_decay = 0.1)
    for epoch in tqdm.tqdm(range(opt.n_epoch2),colour = 'red'):
        train_epoch(train_loader2,model2,epoch,device,loss_fn,optimizer2,writer,type_model = 2,early_print=opt.early_print)
        with torch.no_grad():
            MSE_score = eval_epoch(val_loader2,model2,epoch,device,loss_fn,evaluate_score_fn,type_model = 2)
            if epoch >= opt.early_print:
                writer.add_scalar('Eval_model2/MSE:',MSE_score,epoch)
            if MSE_score<= best_eval_score:
                best_eval_score = MSE_score
                checkpoint = model2.state_dict()
                torch.save({'checkpoint':checkpoint},opt.result_dir+'/model2.ckpt')
    print('eval_best_score model2:{}'.format(best_eval_score))

    checkpoint = torch.load(opt.result_dir+'/model2.ckpt')
    model2.load_state_dict(checkpoint['checkpoint'])
    _2 = eval_epoch(test_loader2,model2,epoch,device,loss_fn,evaluate_score_fn,type_model = 2)
# use logging

    best_eval_score = 1e10
    optimizer_total = torch.optim.Adam(model_total.parameters(),lr=opt.learning_rate_total,weight_decay = 0.1)
    for epoch in tqdm.tqdm(range(opt.n_epoch_total),colour = 'red'):
        train_epoch(train_loader,model_total,epoch,device,loss_fn,optimizer_total,writer,type_model = 'total',early_print=opt.early_print)
        with torch.no_grad():
            MSE_score = eval_epoch(val_loader,model_total,epoch,device,loss_fn,evaluate_score_fn,type_model = 'total')
            if epoch >= opt.early_print:
                writer.add_scalar('Eval_model_total/MSE:',MSE_score,epoch)
            if MSE_score<= best_eval_score:
                best_eval_score = MSE_score
                checkpoint = model_total.state_dict()
                torch.save({'checkpoint':checkpoint},opt.result_dir+'/model_total.ckpt')
    print('eval_best_score model_total:{}'.format(best_eval_score))

    checkpoint = torch.load(opt.result_dir+'/model_total.ckpt')
    model_total.load_state_dict(checkpoint['checkpoint'])
    _total = eval_epoch(test_loader2,model_total,epoch,device,loss_fn,evaluate_score_fn,type_model = 'total')
    logging.info('                      ')
    logging.info('test_score_model0:{}'.format(_0))
    logging.info('test_score_model1:{}'.format(_1))
    logging.info('test_score_model2:{}'.format(_2))
    logging.info('test_score_model_total:{}'.format(_total))
    print('test_score_model0:{}'.format(_0))
    print('test_score_model1:{}'.format(_1))
    print('test_score_model2:{}'.format(_2))
    print('test_score_model_total:{}'.format(_total))
