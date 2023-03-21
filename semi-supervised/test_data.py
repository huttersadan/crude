import pandas as pd
import urllib.request 

import numpy as np
import random
random.seed(114514)
# url = 'https://raw.githubusercontent.com/onecoinbuybus/Database_chemoinformatics/master/SRU_data.txt'
# urllib.request.urlretrieve(url, 'SRU_data.txt') 

sru_data = pd.read_csv('SRU_data.txt',header=None, skiprows=1).iloc[:,0].apply(lambda x: pd.Series(x.split()))
sru_data.columns = ['MEA GAS',
                    'AIR MEA1',
                    'AIR MEA2',
                    'AIR SWS',
                    'SWS GAS',
                    'H2S',
                    'SO2']

total_X,total_Y = np.array(sru_data[['MEA GAS','AIR MEA1','AIR MEA2','AIR SWS','SWS GAS']]),np.array(sru_data[['H2S','SO2']])
length = int(0.7*total_X.shape[0])
total_indexs = [i for i in range(total_X.shape[0])]
train_indexs = random.sample(total_indexs,k=length)
total_indexs = [i for i in range(total_X.shape[0])]
val_test_indexs = []
for idx in total_indexs:
    if idx not in train_indexs:
        val_test_indexs.append(idx)
val_indexs = random.sample(val_test_indexs,k = int(0.67*len(val_test_indexs)))
test_indexs = []
for idx in val_test_indexs:
    if idx not in val_indexs:
        test_indexs.append(idx)
# print(sorted(val_indexs)[0:10],sorted(test_indexs)[0:10])
# print(sorted(train_indexs)[0:10])
train_X,train_Y,val_X,val_Y,test_X,test_Y = total_X[train_indexs],total_Y[train_indexs],total_X[val_indexs],total_Y[val_indexs],total_X[test_indexs],total_Y[test_indexs]

# from sklearn.ensemble import RandomForestRegressor
# best_val_score = 0.0
# best_md = 31
# best_ns = 90
# for md in range(1,32,2):
#     for ns in range(10,201,40):
#         regression_model = RandomForestRegressor(n_estimators=ns,max_depth = md)
#         regression_model.fit(train_X,train_Y)

#         print("Traing Score:%f" % regression_model.score(train_X, train_Y))
#         val_score = regression_model.score(val_X, val_Y)
#         print("Testing Score:%f" % val_score)
#         print('md:{},ns:{}'.format(md,ns))
#         print('         ')
#         if val_score >= best_val_score:
#             best_val_score = val_score
#             best_md = md
#             best_ns = ns

# print('best_md:{},best_ns:{}'.format(best_md,best_ns))
# regression_model = RandomForestRegressor(n_estimators=best_ns,max_depth = best_md)
# regression_model.fit(train_X,train_Y)
# print("Traing Score:%f" % regression_model.score(train_X, train_Y))
# val_score = regression_model.score(val_X, val_Y)
# print("valing Score:%f" % val_score)
# test_score = regression_model.score(test_X, test_Y)
# print("testing Score:%f" % test_score)

from easydict import EasyDict as edict

config = edict()
config.hidden_size = 5
config.intermediate_size = 5
config.num_attention_heads = 1
config.hidden_dropout_prob = 0.1
config.attention_probs_dropout_prob = 0.1

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number of attention heads (%d)" % (
                config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    

    def forward(self, query_states, key_states, value_states, attention_mask=None):
        """
        Args:
            query_states: (N, D)
            key_states: (N,D)
            value_states: (N,D)
            attention_mask: (N, Lq, L)
        """
        # only need to mask the dimension where the softmax (last dim) is applied, as another dim (second last)
        # will be ignored in future computation anyway
        # attention_mask = (1 - attention_mask.unsqueeze(1)) * -10000.  # (N, 1, Lq, L)
        query_layer = self.query(query_states)
        key_layer = self.key(key_states)
        value_layer = self.value(value_states)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # N x D   D x N
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.unsqueeze(1)) * -10000.  # (N, 1, Lq, L)
            attention_scores = attention_scores + attention_mask
        # attention_scores = attention_scores + attention_mask
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        # compute output context
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.contiguous()
        return context_layer



class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states



class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask=None):
        """
        Args:
            input_tensor: (N, L, D)
            attention_mask: (N, L)
        """
        self_output = self.self(input_tensor, input_tensor, input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

class tran_model(nn.Module):
    def __init__(self,config):
        super(tran_model,self).__init__()
        self.config = config
        self.attention_layer = BertAttention(config)
        self.linear_layer = nn.Linear(config.hidden_size,2)
    def reset_parameters(self):
        def re_init(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module, nn.Conv1d):
                module.reset_parameters()
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        self.apply(re_init)
    def forward(self,X_data,Y_data,attention_mask = None):
        x = self.attention_layer(X_data,attention_mask)
        x = self.linear_layer(x)
        #print('x.shape:{}'.format(x.shape))
        return x
    
import torch.utils.data as data
class Dataset_tran(data.Dataset):
    def __init__(self,X_data,Y_data):
        self.X_data = X_data
        self.Y_data = Y_data

    def __getitem__(self, index):
        return [self.X_data[index],self.Y_data[index]]
    def __len__(self):
        return self.X_data.shape[0]
    
train_X,train_Y = torch.tensor(np.array(train_X).astype('float32')),torch.tensor(np.array(train_Y).astype('float32'))
val_X,val_Y = torch.tensor(np.array(val_X).astype('float32')),torch.tensor(np.array(val_Y).astype('float32'))
test_X,test_Y = torch.tensor(np.array(test_X).astype('float32')),torch.tensor(np.array(test_Y).astype('float32'))


from torch.utils.data import DataLoader
def collate_train(data):
    #print('data[0]:{}'.format(data[0]))
    #print('data[1]:{}'.format(data[1]))
    return [data[0],data[1]]
train_dataset,val_dataset,test_dataset = Dataset_tran(train_X,train_Y),Dataset_tran(val_X,val_Y),Dataset_tran(test_X,test_Y)
train_loader = DataLoader(dataset = train_dataset,batch_size=8,shuffle = True)#,collate_fn= collate_train)
val_loader = DataLoader(dataset = val_dataset,batch_size=8,shuffle = True)#,collate_fn= collate_train)
test_loader = DataLoader(dataset = test_dataset,batch_size=8,shuffle = False)#,collate_fn= collate_train)

Model = tran_model(config)
from tqdm import *
optimizer = torch.optim.SGD(Model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

loss_fn = nn.MSELoss()
for batch_idx,batch in tqdm(enumerate(train_loader)):
    #print('batch.shape:{}'.format(batch[0].shape))
    pred_Y = Model(batch[0],batch[1])
    loss = loss_fn(pred_Y,batch[1])
    optimizer.zero_grad()
    loss.backward()
    #print('batch_idx:{},loss:{}'.format(batch_idx,loss.item()))
    optimizer.step()
    
    #print('y_pred.shape:{}'.format(pred_Y.shape))


for batch_idx,batch in tqdm(enumerate(test_loader)):
    #print('batch.shape:{}'.format(batch[0].shape))
    pred_Y = Model(batch[0],batch[1])
    loss = loss_fn(pred_Y,batch[1])
    #optimizer.zero_grad()
    #loss.backward()
    print('batch_idx:{},loss:{}'.format(batch_idx,loss.item()))
    #optimizer.step()