import pandas as pd
 

import numpy as np
import random
import os
random.seed(114514)
import matplotlib.pyplot as plt
random_seed = 114514
random.seed(random_seed)
np.random.seed(random_seed)
from sklearn.mixture import GaussianMixture
def Gauss(total_random_generate,n_clusters):
    X = [inst['output'] for inst in total_random_generate]
    model = GaussianMixture(n_components=n_clusters)
    y_pred = model.fit_predict(X)
    rs_index_dict = {}
    for i in range(n_clusters):
        rs_index_dict[i] = []
    for idx in range(len(total_random_generate)):
        rs_index_dict[y_pred[idx]].append(total_random_generate[idx]['output'])
        total_random_generate[idx]['y_class'] = y_pred[idx]
    return rs_index_dict, total_random_generate
def get_random_input(mass_of_oil,rate_of_output,num_of_generation):
    rs_input_generation = []
    rs_output_generation = []
    while num_of_generation > 0:
        input_random1 = np.random.randn()
        input_random2 = np.random.randn()
        input_random3 = np.random.randn()

        input_100 = round(mass_of_oil[input_features[0]] + input_random1*100)
        input_101 = round(mass_of_oil[input_features[1]] + input_random2*100)
        input_102 = 235000-input_100 - input_101
        output_701 = round(rate_of_output[output_features[0]] + input_random1 * 200)
        output_702 = round(rate_of_output[output_features[1]] + input_random2 * 200)
        if input_random1 * 100 + input_random2 * 100 > 0:
            input_random3 = -abs(input_random3)
        else:
            input_random3 = abs(input_random3)
        output_703 = round(rate_of_output[output_features[2]] + input_random3 * 100)
        if input_100 > 0 and input_101 > 0 and input_102 > 0 and output_701 > 0 and output_702 > 0 and output_703 > 0:
            input_generation = [input_100/(input_100+input_101+input_102),input_101/(input_100+input_101+input_102),input_102/(input_100+input_101+input_102)]
            output_generation = [output_701,output_702,output_703]
            num_of_generation -= 1
            rs_input_generation.append(input_generation)
            rs_output_generation.append(output_generation)
    return rs_input_generation,rs_output_generation

def get_single_csv_random_generate(temp):
    mass_of_oil = temp[input_features].iloc[-1]
    rate_of_output = temp[output_features].iloc[-1]
    num_of_generation = 300
    random_input_ls,random_output_ls  = get_random_input(mass_of_oil,rate_of_output,num_of_generation)
    #random_output_ls = get_random_output(rate_of_output, num_of_generation)
    rs_ls = [{'input':i,'output':n} for i,n in zip(random_input_ls,random_output_ls)]
    return rs_ls
def print_plot(rs_index_dict):
    colours = ['dimgray','lightcoral','darkred','sienna',
               'darkorange','tan','gold','olive','lawngreen',
               'palegreen','lime','turquoise','teal','powderblue',
               'dodgerblue','blue','darkorchid','fuchsia',
               'deeppink','crimson']
    random.shuffle(colours)
    markers = ["o","v","+","o","v","+","o","v","+","o","v","+","o","v","+"
               ,"o","v","+","o","v","+","o","v","+","o","v","+","o","v","+","o","v","+"
               ,"o","v","+","o","v","+","o","v","+","o","v","+"]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for idx,values in rs_index_dict.items():
        ax.scatter([values[i][0]/1000 for i in range(len(values))] ,
                   [values[i][1]/1000 for i in range(len(values))],
                   [values[i][2]/1000 for i in range(len(values))] ,
                   c=colours[idx],
                   marker=markers[idx])
    ax.set_xlabel('Kero')
    ax.set_ylabel('Diesel')
    ax.set_zlabel('AGO')
    ax.set_title('clusters plot',color = 'red')
    plt.show()
path_dataset = 'D:/大四上/毕设/dataset'
path_list = os.listdir(path_dataset)
csv_file_path_list = []
for file_name in path_list:
    if file_name[-4:] == '.csv':
        csv_file_path_list.append(path_dataset + '/' + file_name)
input_features = ['FIC-100 - PV', 'FIC-101 - PV', 'FIC-102 - PV']
output_features = ['FIC10701 - PV', 'FIC10702 - PV', 'FIC10703 - PV'] 

csv_files_ls = []
for i in  range(len(csv_file_path_list)):
    csv_files_ls.append(pd.read_csv(csv_file_path_list[i], skiprows=[j for j in range(9)] + [10]))
total_random_generate = []
for temp in csv_files_ls:
    total_random_generate.extend(get_single_csv_random_generate(temp))
random.shuffle(total_random_generate)



n_clusters = 3
rs_index_dict, total_data = Gauss(total_random_generate, n_clusters)
print_plot(rs_index_dict)
total_X,total_Y,total_y_class = np.array([total_random_generate[i]['input'] for i in range(len(total_random_generate))]),np.array([total_random_generate[i]['output'] for i in range(len(total_random_generate))]),np.array([total_random_generate[i]['y_class'] for i in range(len(total_random_generate))])
length = int(0.7*total_X.shape[0])
total_indexs = [i for i in range(total_X.shape[0])]
train_indexs = random.sample(total_indexs,k=length)
total_indexs = [i for i in range(total_X.shape[0])]

val_indexs = []
for idx in total_indexs:
    if idx not in train_indexs:
        val_indexs.append(idx)

train_X,train_Y,val_X,val_Y = total_X[train_indexs],total_Y[train_indexs],total_X[val_indexs],total_Y[val_indexs]
train_yclass,val_yclass = total_y_class[train_indexs],total_y_class[val_indexs]
print(train_X)
print(total_y_class)
from model import *

train_X,train_Y,train_yclass = torch.tensor(np.array(train_X).astype('float32')),torch.tensor(np.array(train_Y).astype('float32')),torch.tensor(np.array(train_yclass).astype('int'))
val_X,val_Y,val_yclass = torch.tensor(np.array(val_X).astype('float32')),torch.tensor(np.array(val_Y).astype('float32')),torch.tensor(np.array(val_yclass).astype('int'))
from torch.utils.data import DataLoader
def collate_train(data):
    return [data[0],data[1],data[2]]

class entroy_min(nn.Module):
    def __init__(self):
        super(entroy_min,self).__init__()
    def forward(self,x):
        #x = batch_size x n_dim
        #loss = \sum pi * log pi
        loss = torch.mul(x , torch.log(x))
        return -torch.sum(torch.sum(loss,dim = -1),dim=-1)

train_dataset,val_dataset = Dataset_tran(train_X,train_Y,train_yclass),Dataset_tran(val_X,val_Y,val_yclass)
train_loader = DataLoader(dataset = train_dataset,batch_size=32,shuffle = True)#,collate_fn= collate_train)
val_loader = DataLoader(dataset = val_dataset,batch_size=len(val_X),shuffle = False)#,collate_fn= collate_train)
#test_loader = DataLoader(dataset = test_dataset,batch_size=8,shuffle = False)#,collate_fn= collate_train)

device = torch.device('cuda')
#Model = tran_model(config).to(device)
Model = linear_model_pytorch(3,40,3).to(device)
from tqdm import *
optimizer = torch.optim.SGD(Model.parameters(), lr=0.3)
#loss_fn = entroy_min()
loss_fn = nn.CrossEntropyLoss()
eval_acc_ls = []
stop_score = 0
prev_best_score = 0
es_cnt = 0
es_epoch_cnt = 20
n_epochs = 50
from sklearn.metrics import accuracy_score
for epoch in tqdm(range(n_epochs),colour = 'red'):
    Model.train(mode=True)
    train_total_y_pred = []
    train_total_y_target = []
    for batch_idx,batch in enumerate(train_loader):
        pred_Y_unsoftmax = Model(batch[0].to(device))
        y_pred = torch.argmax(pred_Y_unsoftmax,dim = -1).cpu().tolist()
        train_total_y_pred.extend(y_pred)
        train_total_y_target.extend(batch[2].cpu().tolist())
        loss = loss_fn(pred_Y_unsoftmax,batch[2].type(torch.long).to(device))
        optimizer.zero_grad()
        loss.backward()
        
        #print('loss:{}'.format(loss.item()))
        #print('batch_idx:{},loss:{}'.format(batch_idx,loss.item()))
        optimizer.step()
    train_acc_scores = accuracy_score(train_total_y_pred,train_total_y_target)
    #print('single_acc:{}'.format(accuracy_score(y_pred,batch[2].cpu().tolist())))
    Model.eval()
    for eval_batch in val_loader:
        batch_eval_x,batch_eval_y = eval_batch[0].to(device),eval_batch[2].to(device)
        pred_Y_unsoftmax = Model(batch_eval_x)
        y_pred = torch.argmax(pred_Y_unsoftmax,dim = -1).cpu().tolist()
        batch_eval_y = batch_eval_y.cpu().tolist()
        acc_scores = accuracy_score(y_pred,batch_eval_y)
        eval_acc_ls.append(acc_scores)
    stop_score = acc_scores
    print('step:{},train_acc_scores:{},eval_acc:{}\n'.format(epoch,round(train_acc_scores,4),round(stop_score,4)))
    if stop_score > prev_best_score:
        es_cnt = 0
        prev_best_score = stop_score
        checkpoint = {"model": Model.state_dict(), "model_cfg":n_clusters}
        torch.save(checkpoint, 'model_dict.ckpt'+str(n_clusters))
    else:
        es_cnt += 1
        if es_cnt > es_epoch_cnt:  # early stop
            pass


Model.load_state_dict(torch.load('model_dict.ckpt3')['model'])

for eval_batch in tqdm(val_loader,colour = 'blue'):
    batch_eval_x,batch_eval_y = eval_batch[0].to(device),eval_batch[2].to(device)
    pred_Y_unsoftmax = Model(batch_eval_x)
    y_pred = torch.argmax(pred_Y_unsoftmax,dim = -1).cpu().tolist()
    batch_eval_y = batch_eval_y.cpu().tolist()
    eval_acc_scores = accuracy_score(y_pred,batch_eval_y)
total_y_pred = []
total_y_target = []
for eval_batch in tqdm(train_loader,colour = 'green'):
    batch_eval_x,batch_eval_y = eval_batch[0].to(device),eval_batch[2].to(device)
    pred_Y_unsoftmax = Model(batch_eval_x)
    y_pred = torch.argmax(pred_Y_unsoftmax,dim = -1).cpu().tolist()
    total_y_pred.extend(y_pred)
    batch_eval_y = batch_eval_y.cpu().tolist()
    total_y_target.extend(batch_eval_y)
train_acc_scores = accuracy_score(total_y_pred,total_y_target)
print(train_acc_scores)
print(eval_acc_scores)
















