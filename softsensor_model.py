import pandas as pd
import numpy as np
import random
import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import scipy.io as scio
def data_provider():
    path_dataset = 'D:/yuketang/USD_OS 3/softsensor_data2'
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
    read_data_squeeze = read_data['rs_read_data'][0][0]
    write_data_squeeze = write_data['write_data'][0][0]
    
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
        np.squeeze(write_data_squeeze[write_data_name_dict['medium']]/235),#进料
        np.squeeze(write_data_squeeze[write_data_name_dict['heavy']]/235),#进料
        np.squeeze(write_data_squeeze[write_data_name_dict['light']]/235)#进料
    ])
    total_Y = np.array([
        read_data_squeeze[read_data_name_dict['ATOPKK_PV']],
        read_data_squeeze[read_data_name_dict['ATM1HK_PV']],
        read_data_squeeze[read_data_name_dict['ATM1KK_PV']],
        read_data_squeeze[read_data_name_dict['ATM1FLH_PV']],
        read_data_squeeze[read_data_name_dict['ATM2KK_PV']],
        read_data_squeeze[read_data_name_dict['ATM3KK_PV']],
    ])
    rs_dict_ls = []
    total_X = np.squeeze(total_X)
    total_Y = np.squeeze(total_Y)

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
def collate_train(data):
    batch_X,batch_Y = zip(*data)
    batch_X = torch.tensor(np.array(batch_X),dtype=torch.float)
    batch_Y = torch.tensor(np.array(batch_Y),dtype=torch.float)
    return batch_X,batch_Y

def eval_epoch(val_loader,model,epoch,device,loss_fn,evaluate_score_fn):
    model.eval()
    batch_pred_Y = None
    for batch_idx,(batch_X,batch_Y) in enumerate(val_loader):
        batch_X,batch_Y = batch_X.to(device),batch_Y.to(device)
        batch_pred_Y = model(batch_X)

    score = evaluate_score_fn(batch_pred_Y,batch_Y)
    

    return score


def train_epoch(train_loader,model,epoch,device,loss_fn,optimizer,writer):
    model.train()
    for batch_idx,(batch_X,batch_Y) in enumerate(train_loader):
        global_step = len(train_loader)*epoch + batch_idx
        batch_X,batch_Y = batch_X.to(device),batch_Y.to(device)
        #print(batch_X[0])
        #print(batch_Y[0])
        batch_pred_y = model(batch_X)
        #print(batch_pred_y[0])
        #print('\n')
        loss = loss_fn(batch_pred_y,batch_Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar("Train/MSELoss", loss,global_step)
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):  # 继承 torch 的 Module（固定）
    def __init__(self, n_feature, n_hidden, n_output,num_of_hidden_layers):  # 定义层的信息，n_feature多少个输入, n_hidden每层神经元, n_output多少个输出
        super(Net, self).__init__()  # 继承 __init__ 功能（固定）
        self.num_of_hidden_layers = num_of_hidden_layers
        self.hidden = nn.Linear(n_feature, n_hidden)  # 定义隐藏层，线性输出
        self.hidden_list = nn.ModuleList([nn.Linear(n_hidden, n_hidden) for i in range(num_of_hidden_layers)])
        self.BN = nn.BatchNorm1d(n_feature)
        self.predict = nn.Linear(n_hidden, n_output)  # 定义输出层线性输出
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):  # x是输入信息就是data，同时也是 Module 中的 forward 功能，定义神经网络前向传递的过程，把__init__中的层信息一个一个的组合起来
        x = self.BN(x)
        x = F.relu(self.hidden(x))
        x = self.dropout(x)
        for i in range(self.num_of_hidden_layers):
            x = F.relu(self.hidden_list[i](x))  # 定义激励函数(隐藏层的线性值)
            x = self.dropout(x)
            #x = self.batch_norm_list[i](x)
        x = self.predict(x)  # 输出层，输出值
        return x
import tqdm
import argparse
import time
if __name__=='__main__':
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
    n_epochs = 100
    train_dataset,val_dataset,test_dataset = Dataset_SRU(train_data),Dataset_SRU(val_data),Dataset_SRU(test_data)
    train_loader = DataLoader(train_dataset,batch_size = 32,shuffle=True,collate_fn=collate_train,drop_last=True)
    val_loader = DataLoader(val_dataset,batch_size = val_dataset.__len__(),shuffle=False,collate_fn=collate_train,drop_last=True)
    test_loader = DataLoader(test_dataset,batch_size = test_dataset.__len__(),shuffle=False,collate_fn=collate_train,drop_last=True)
    model = Net(n_feature=25,n_output=6,n_hidden=128,num_of_hidden_layers=3)
    
    device = torch.device('cuda')
    model = model.to(device)
    loss_fn = nn.MSELoss()
    #loss_fn = nn.KLDivLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr= 0.05,weight_decay = 0.1)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id',help = 'experiment_id',type=str,default='test')
    opt = parser.parse_args()
    opt.result_dir = 'D:/crude/' + opt.exp_id +time.strftime("%Y_%m_%d_%H_%M_%S")
    if os.path.exists(opt.result_dir) == 0:
        os.mkdir(opt.result_dir)
    
    opt.tensorboard_log_dir = opt.result_dir+'/tensorboard_log_dir'
    writer = SummaryWriter(opt.tensorboard_log_dir)
    evaluate_score_fn = nn.MSELoss()
    best_eval_score = 1e10
    for epoch in tqdm.tqdm(range(n_epochs),colour = 'red'):
        train_epoch(train_loader,model,epoch,device,loss_fn,optimizer,writer)
        with torch.no_grad():
            MSE_score = eval_epoch(val_loader,model,epoch,device,loss_fn,evaluate_score_fn)
            writer.add_scalar('Eval/MSE:',MSE_score,epoch)
            if MSE_score<= best_eval_score:
                best_eval_score = MSE_score
    print(best_eval_score)
    writer.close()