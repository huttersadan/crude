
from easydict import EasyDict as edict

config = edict()
config.hidden_size = 3
config.intermediate_size = 3
config.num_attention_heads = 1
config.hidden_dropout_prob = 0.1
config.attention_probs_dropout_prob = 0.1

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
torch.manual_seed(114514)

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
        self.linear_layer = nn.Linear(config.hidden_size,config.hidden_size)
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
    def forward(self,X_data,attention_mask = None):
        x = self.attention_layer(X_data,attention_mask)
        x = self.linear_layer(x)
        x = F.softmax(x,dim=-1)
        return x



import torch.utils.data as data
class Dataset_tran(data.Dataset):
    def __init__(self,X_data,Y_data,yclass_data):
        self.X_data = X_data
        self.Y_data = Y_data
        self.yclass_data = yclass_data
    def __getitem__(self, index):
        return [self.X_data[index],self.Y_data[index],self.yclass_data[index]]
    def __len__(self):
        return self.X_data.shape[0]
    


class linear_model_pytorch(nn.Module):
    def __init__(self,in_hsz,hidden_size,out_hsz, dropout=0.1, relu=True):
        super(linear_model_pytorch,self).__init__()
        self.relu = relu
        self.linear1 = nn.Linear(in_hsz,hidden_size)

        self.linear2 = nn.Linear(hidden_size,hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)
        self.linear5 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size,out_hsz)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    def forward(self,x):
        #print(x[1])
        x = torch.relu(self.linear1(x))
        #x = self.dropout1(x)
        x = torch.relu(self.linear2(x))
        #x = torch.sigmoid(self.linear4(x))
        #x = torch.sigmoid(self.linear5(x))
        #x = self.dropout2(x)
        x = torch.relu(self.linear3(x))
        #x = self.dropout3(x)
        # y_pred = F.softmax(x, dim=-1)
        # loss_fn = nn.CrossEntropyLoss()
        # loss = loss_fn(x,y.type(torch.long))
        #print('loss :{}'.format(loss))
        #print(y_pred)
        #print(y)
        return x
from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter('lo1gs')
# x = range(100)

# writer.close()

from tensorboardX import SummaryWriter
# 定义Summary_Writer
writer = SummaryWriter(r'D:\yuketang\Results_2')
x = range(100)
for i in x:
    writer.add_scalar('y=x+10', i+10, i)
writer.close()