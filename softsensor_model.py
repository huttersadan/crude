import pandas as pd
import numpy as np
import random
import os
import torch
import torch.nn as nn
import scipy.io as scio
def data_provider():
    path_dataset = 'D:/大四上/毕设/crude/crude/softsensor_data'
    # sru_data = pd.read_csv(path_dataset,header=None, skiprows=1).iloc[:,0].apply(lambda x: pd.Series(x.split()))
    # sru_data.columns = ['MEA GAS',
    #                     'AIR MEA1',
    #                     'AIR MEA 2',
    #                     'AIR SWS',
    #                     'SWS GAS',
    #                     'H2S',
    #                     'SO2']
    read_data = scio.loadmat(path_dataset+'/read1.mat')
    write_data = scio.loadmat(path_dataset+'/write1.mat')
    '''
    'draw1', ...%回流温度1——pred
    'draw2', ...%回流温度2——pred
    'draw3', ...%回流温度3——pred
    'draw1r', ...%回流温度1——next
    'draw2r', ...%回流温度2——next
    'draw3r', ...%回流温度3——next
    '''
    #print(read_data)
    read_data_name_dict = [
    "TIC10701_PV",
    "TI10702_PV",
    "TI10703_PV" ,
    "TI10704_PV" ,
    "ATMFEED_PV" ,
    "draw1" ,
    "draw2" ,
    "draw3" ,
    "draw1r",
    "draw2r" ,
    "draw3r" ,
    "ATOPKK_PV",
    "ATM1HK_PV" ,
    "ATM1KK_PV" ,
    "ATM1FLH_PV",
    "ATM2KK_PV" ,
    "ATM3KK_PV" ,
    ]
    read_data_squeeze = read_data['rs_read_data'][0][0]
    write_data_squeeze = write_data['write']
    print(write_data['write_data'][0][0])
    #print(len(read_data['rs_read_data'][0][0]))
    total_X = np.array([read_data['rs_read_data'][0][0][0],read_data['rs_read_data'][0][0][1],read_data['rs_read_data'][0][0][2],write_data['write_data'][0][0][0]])
    total_Y = np.array([read_data['rs_read_data'],write_data['write_data'][0][0][1]])
    #total_X,total_Y = np.array(sru_data[['MEA GAS','AIR MEA1','AIR MEA 2','AIR SWS','SWS GAS']]),np.array(sru_data[['H2S','SO2']])
    rs_dict_ls = []
    total_X = np.squeeze(total_X)
    total_Y = np.squeeze(total_Y)
    #print(total_X.shape)
    #print(total_Y.shape)
    for i in range(len(total_X)):
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
def collate_train(data):
    batch_X,batch_Y = zip(*data)
    return batch_X,batch_Y

def eval_epoch(val_loader,model,epoch,device,loss_fn):
    pass
def train_epoch(train_loader,model,epoch,device,loss_fn,optimizer,writer):
    model.train()
    for batch_idx,(batch_X,batch_Y) in enumerate(train_loader):
        global_step = len(train_loader)*epoch + batch_idx
        batch_X,batch_Y = batch_X.to(device),batch_Y.to(device)
        batch_pred_y = model(batch_X)
        loss = loss_fn(batch_pred_y,batch_Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar("Train/MSELoss", loss, global_step)
if __name__=='__main__':
    print(data_provider())
if __name__!='__main__':
    seed_name = 114514
    random.seed(seed_name)
    np.random.seed(seed_name)
    torch.manual_seed(seed=seed_name)
    total_data = data_provider()
    random.shuffle(total_data)
    select_indexes = int(len(total_data)*0.8)
    val_text_same = True
    if val_text_same:
        train_data,val_data,test_data = total_data[:select_indexes],total_data[select_indexes:],total_data[select_indexes:]
    else:
        test_indexs = int(len(total_data)*0.9)
        train_data,val_data,test_data = total_data[:select_indexes],total_data[select_indexes:test_indexs],total_data[test_indexs:]
    n_epochs = 50
    train_dataset,val_dataset,test_dataset = Dataset_SRU(train_data),Dataset_SRU(val_data),Dataset_SRU(test_data)
    train_loader = DataLoader(train_dataset,batch_size = 16,shuffle=True,collate_fn=collate_train,drop_last=True)
    val_loader = DataLoader(val_dataset,batch_size = val_dataset.__len__,shuffle=False,collate_fn=collate_train,drop_last=True)
    test_loader = DataLoader(test_dataset,batch_size = test_dataset.__len__,shuffle=False,collate_fn=collate_train,drop_last=True)
    
    for epoch in n_epochs:
        train_epoch(train_loader,model,epoch,device,loss_fn,optimizer,writer)