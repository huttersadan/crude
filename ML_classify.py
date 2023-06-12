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

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
def data_provider(multi):
    
    if not multi:
        path_dataset = 'USD_OS 3/softsensor_data5'
        read_data = scio.loadmat(path_dataset+'/read.mat')
        write_data = scio.loadmat(path_dataset+'/write.mat')
        read_data_squeeze = read_data['rs_read_data'][0][0]
        write_data_squeeze = write_data['write_data'][0][0]
        # write_refine_data_squeeze = []
        # for inst in write_data_squeeze:
        #     write_refine_data_squeeze.append(inst[:500])
        # write_data_squeeze = write_refine_data_squeeze
        total_X,total_Y = get_single_XandY(read_data_squeeze,write_data_squeeze)
        #绘制辅助变量和主导变量之间的相关系数
        # corr_ls = [np.corrcoef(total_X[i],total_Y[0]) for i in range(len(total_X))]
        # fig, ax = plt.subplots(figsize=(9,6))
        # print(corr_ls[0])
        # for idx,names in enumerate(corr_ls):
        #     ax.bar(idx,corr_ls[idx][0][1],label = idx,align='center',width=0.3)
        # #plt.plot(x,y,label='Correlation')
        # ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])
        # plt.title('辅助变量和主导变量之间的相关系数',fontsize=20)
        # #plt.savefig('4_2.png')
        # plt.show()

        # # 绘制高斯噪声下的结果
        # plt_data = copy.deepcopy(total_X[9])[:150]
        # random.shuffle(plt_data)
        # x = [i for i in range(len(plt_data))]
        # plt_data1 = plt_data+ np.random.randn(len(x))
        # plt.plot(x,plt_data,label = '无噪声的仿真值')
        # plt.plot(x,plt_data1,label = '有噪声的仿真值')
        # plt.title('顶循回流采出温度')
        # plt.ylabel('温度/℃')
        # plt.legend()

        # plt.show()
        # #绘制散点图
        # plt.figure(figsize=(12,4))
        # plt.title('顶循回流返塔温度')
        # plt.scatter(x,plt_data,s=10, c='green', marker='o')
        # plt.xlabel('采样')
        # plt.ylabel('温度 ℃')
        # plt.show()
        
        #df = pd.DataFrame({'二中回流采出温度':total_X[8][:1000],'二中回流返塔温度':total_X[11][:1000],'一中回流采出温度':total_X[7][:1000],'一中回流返塔温度':total_X[10][:1000]})
        
        # box_colors = ['r', 'g','pink','blue']

        # # # 使用 Matplotlib 绘制盒图
        # bp = plt.boxplot(df.values, patch_artist=True, boxprops=dict(facecolor='w'), capprops=dict(color='k'), whiskerprops=dict(color='k'), flierprops=dict(marker='o', markersize=5), medianprops=dict(color='k'))

        # # # 为每个箱子设置不同的颜色
        # for i, box in enumerate(bp['boxes']):
        #     box.set_facecolor(box_colors[i % len(box_colors)])
        # plt.boxplot(df.values)
        # # 添加标题和标签
        # plt.title('一中与二中回流采出温度和返塔温度箱线图')
        # plt.xticks(range(1, len(df.columns) + 1), df.columns)
        # plt.ylabel('温度')
        # plt.show()
        # df = {}
        # for idx,inst in enumerate(total_X[6:]):
        #     df[idx] = inst
        # df = pd.DataFrame(df)
        # corr_matrix = df.corr()

        # # 使用 imshow() 方法显示相关系数矩阵
        # plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')

        # # 添加颜色条和标签
        # plt.colorbar()
        # tick_marks = np.arange(len(corr_matrix.columns))
        # plt.xticks(tick_marks, corr_matrix.columns, rotation=0)
        # plt.yticks(tick_marks, corr_matrix.columns)
        # # for i in range(len(corr_matrix.columns)):
        # #     for j in range(len(corr_matrix.columns)):
        # #         plt.text(j, i, corr_matrix.iloc[i, j].round(2), ha='center', va='center', fontsize=14, color='w')
        # # 显示图形
        # plt.title('辅助变量之间的相关系数矩阵')
        # plt.show()
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
from torchsummary import summary
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
    parser.add_argument('--dropout',type = float,default = 0.1,help = 'Dropout')
    opt = parser.parse_args()
    opt.result_dir = 'results/' + opt.exp_id +time.strftime("%Y_%m_%d_%H_%M_%S")
    
    train_loader = DataLoader(train_dataset,batch_size = train_dataset.__len__(),shuffle=False,collate_fn=collate_train,drop_last=True)
    val_loader = DataLoader(val_dataset,batch_size = val_dataset.__len__(),shuffle=False,collate_fn=collate_train,drop_last=True)
    test_loader = DataLoader(test_dataset,batch_size = test_dataset.__len__(),shuffle=False,collate_fn=collate_train,drop_last=True)
    
    device = torch.device('cpu')
    classify_model = linear_model_pytorch(3,40,3).to(device)
    classify_model.load_state_dict(torch.load('model_dict.ckpt3')['model'])
    classify_model.eval()
    for batch_idx,(batch_X,batch_Y) in enumerate(total_loader):
        batch_x = (batch_X[:,[-3,-2,-1]].to(device))* 1000
        batch_y = torch.zeros(batch_x.shape[0]).to(device)
        _,y_pred = classify_model(batch_x,batch_y)
        y_pred = torch.argmax(y_pred,dim = -1)
    total_X0 = copy.deepcopy(batch_X[y_pred == 0])
    total_X1 = copy.deepcopy(batch_X[y_pred == 1])[:150]
    total_X2 = copy.deepcopy(batch_X[y_pred == 2])[:150]
    total_Y0 = copy.deepcopy(batch_Y[y_pred == 0])
    total_Y1 = copy.deepcopy(batch_Y[y_pred == 1])[:150]
    total_Y2 = copy.deepcopy(batch_Y[y_pred == 2])[:150]
    
    total_X0 = copy.deepcopy(total_X0[:150])
    total_Y0 = copy.deepcopy(total_Y0[:150])
    total_X = torch.cat((total_X0,total_X1,total_X2),dim=0)
    total_Y = torch.cat((total_Y0,total_Y1,total_Y2),dim=0)
    

    X_all,Y_all = [x.detach().cpu().numpy() for x in total_X],[Y.detach().cpu().numpy() for Y in total_Y]
    X0,Y0 = [x.detach().cpu().numpy() for x in total_X0],[Y.detach().cpu().numpy() for Y in total_Y0]
    X1,Y1 = [x.detach().cpu().numpy() for x in total_X1],[Y.detach().cpu().numpy() for Y in total_Y1]
    X2,Y2 = [x.detach().cpu().numpy() for x in total_X2],[Y.detach().cpu().numpy() for Y in total_Y2]
    
    X_train, X_test, y_train, y_test \
        = train_test_split(X_all, Y_all, test_size=0.1, random_state=0)
    X_train = X_train[:150]
    X_test = X_test[:150]
    y_train = y_train[:150]
    y_test = y_test[:150]
    X_train0, X_test0, y_train0, y_test0 \
        = train_test_split(X0, Y0, test_size=0.5, random_state=0)
    X_train1, X_test1, y_train1, y_test1 \
        = train_test_split(X1, Y1, test_size=0.5, random_state=0)
    X_train2, X_test2, y_train2, y_test2 \
        = train_test_split(X2, Y2, test_size=0.5, random_state=0)
    print('X_train:{}'.format(len(X_train)))
    print('X_train0:{}'.format(len(X_train0)))
    print('X_train1:{}'.format(len(X_train1)))
    print('X_train2:{}'.format(len(X_train2)))
    #import ipdb;ipdb.set_trace()
    #model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    model = PLSRegression(n_components=10)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse=  np.sqrt(mse)
    print("total: %.2f\n" % rmse)

    model.fit(X_train0, y_train0)
    y_pred0 = model.predict(X_test0)
    mse = mean_squared_error(y_test0, y_pred0)
    rmse=  np.sqrt(mse)
    print("Light: %.2f\n" % rmse)

    model.fit(X_train1, y_train1)
    y_pred1 = model.predict(X_test1)
    mse = mean_squared_error(y_test1, y_pred1)
    rmse=  np.sqrt(mse)
    print("Medium: %.2f\n" % rmse)
    
    model.fit(X_train2, y_train2)
    y_pred2 = model.predict(X_test2)
    for i in range(len(y_pred2)):
        print('y_pred2:{},y_test2:{}'.format(y_pred2[i],y_test2[i]))
    sum_square_error = [(y_pred2[i]-y_test2[i])**2 for i in range(len(y_pred2))]
    print('sse = {},num = {}'.format(sum(sum_square_error),len(y_pred2)))
    mse = mean_squared_error(y_test2, y_pred2)
    rmse=  np.sqrt(mse)
    print("Heavy: %.2f" % rmse)

    # df_dict = {'n_components':[],'RMSE':[]}
    # for i in range(1,25):
    #     model = PLSRegression(n_components=i)
    #     model.fit(X_train0, y_train0)
    #     y_pred0 = model.predict(X_test0)
    #     mse = mean_squared_error(y_test0, y_pred0)
    #     rmse=  np.sqrt(mse)
        
    #     print("Mean squared error: %.2f" % rmse)
    #     df_dict['n_components'].append(i)
    #     df_dict['RMSE'].append(rmse)
    # df = pd.DataFrame(df_dict)
    # fig, ax = plt.subplots()
    # #for i in range(2,5):
    # #     ax.scatter(df[df['min_samples_leaf'] == msl]['max_depth'], df[df['min_samples_leaf'] == msl]['RMSE'], cmap='viridis',label = "min_samples_leaf = {}".format(msl))
    # ax.scatter(df['n_components'],df['RMSE'],cmap='viridis')
    # ax.legend()
    # ax.set_xlabel('max_depth')
    # ax.set_ylabel('RMSE')
    # ax.set_title('RMSE vs. max_depth')
    # plt.show()
