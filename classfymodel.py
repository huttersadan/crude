from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


def SVM_classify(total_data,n_clusters):
    X = [single_data['input'] for single_data in total_data]
    y = [single_data['y_class'] for single_data in total_data]
    seperate_index = int(len(X) * 0.8)
    train_X = X[:seperate_index]
    train_y = y[:seperate_index]
    test_X = X[seperate_index:]
    test_y = y[seperate_index:]
    clf = make_pipeline(StandardScaler(),LinearSVC(random_state=0, tol=1e-5))
    clf.fit(train_X, train_y)
    y_pred = clf.predict(test_X)
    train_score = accuracy_score(train_y,clf.predict(train_X))
    test_score = accuracy_score(test_y, y_pred)
    print('train_score:{},test_score:{}'.format(round(train_score,3),round(test_score,3)))
    return test_score


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
def decision_tree_classify(total_data,n_clusters):
    X = [single_data['input'] for single_data in total_data]
    y = [single_data['y_class'] for single_data in total_data]
    seperate_index = int(len(X) * 0.8)
    train_X = X[:seperate_index]
    train_y = y[:seperate_index]
    test_X = X[seperate_index:]
    test_y = y[seperate_index:]
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(train_X,train_y)
    y_pred = clf.predict(test_X)
    train_score = accuracy_score(train_y,clf.predict(train_X))
    test_score = accuracy_score(test_y, y_pred)
    print('train_score:{},test_score:{}'.format(round(train_score,3),round(test_score,3)))
    return test_score


from sklearn.linear_model import SGDClassifier

def SGD_classify(total_data,n_clusters):
    X = [single_data['input'] for single_data in total_data]
    y = [single_data['y_class'] for single_data in total_data]
    seperate_index = int(len(X) * 0.8)
    train_X = X[:seperate_index]
    train_y = y[:seperate_index]
    test_X = X[seperate_index:]
    test_y = y[seperate_index:]
    clf = make_pipeline(StandardScaler(),SGDClassifier(max_iter=1000, tol=1e-3))
    clf.fit(train_X, train_y)
    y_pred = clf.predict(test_X)
    train_score = accuracy_score(train_y,clf.predict(train_X))
    test_score = accuracy_score(test_y, y_pred)
    print('train_score:{},test_score:{}'.format(round(train_score,3),round(test_score,3)))
    return test_score


from sklearn.neighbors import KNeighborsClassifier
def knn_classify(total_data,n_clusters):
    X = [single_data['input'] for single_data in total_data]
    y = [single_data['y_class'] for single_data in total_data]
    seperate_index = int(len(X) * 0.8)
    train_X = X[:seperate_index]
    train_y = y[:seperate_index]
    test_X = X[seperate_index:]
    test_y = y[seperate_index:]
    neigh = KNeighborsClassifier(n_neighbors=12,weights='distance')
    neigh.fit(train_X,train_y)
    #clf = make_pipeline(StandardScaler(),SGDClassifier(max_iter=1000, tol=1e-3))
    #clf.fit(train_X, train_y)
    y_pred = neigh.predict(test_X)
    train_score = accuracy_score(train_y,neigh.predict(train_X))
    test_score = accuracy_score(test_y, y_pred)
    print('train_score:{},test_score:{}'.format(round(train_score,3),round(test_score,3)))
    return test_score


from sklearn.naive_bayes import GaussianNB
def GaussianNB_classify(total_data,n_clusters):
    X = [single_data['input'] for single_data in total_data]
    y = [single_data['y_class'] for single_data in total_data]
    seperate_index = int(len(X) * 0.8)
    train_X = X[:seperate_index]
    train_y = y[:seperate_index]
    test_X = X[seperate_index:]
    test_y = y[seperate_index:]
    clf = GaussianNB()
    clf.fit(train_X,train_y)
    #clf = make_pipeline(StandardScaler(),SGDClassifier(max_iter=1000, tol=1e-3))
    #clf.fit(train_X, train_y)
    y_pred = clf.predict(test_X)
    train_score = accuracy_score(train_y,clf.predict(train_X))
    test_score = accuracy_score(test_y, y_pred)
    print('train_score:{},test_score:{}'.format(round(train_score,3),round(test_score,3)))
    return test_score

from sklearn.naive_bayes import CategoricalNB
def CategoricalNB_classify(total_data,n_clusters):
    X = [single_data['input'] for single_data in total_data]
    y = [single_data['y_class'] for single_data in total_data]
    seperate_index = int(len(X) * 0.8)
    train_X = X[:seperate_index]
    train_y = y[:seperate_index]
    test_X = X[seperate_index:]
    test_y = y[seperate_index:]
    clf = CategoricalNB()
    clf.fit(train_X,train_y)
    #clf = make_pipeline(StandardScaler(),SGDClassifier(max_iter=1000, tol=1e-3))
    #clf.fit(train_X, train_y)
    y_pred = clf.predict(test_X)
    train_score = accuracy_score(train_y,clf.predict(train_X))
    test_score = accuracy_score(test_y, y_pred)
    print('train_score:{},test_score:{}'.format(round(train_score,3),round(test_score,3)))
    return test_score

from sklearn.naive_bayes import ComplementNB
def ComplementNB_classify(total_data,n_clusters):
    X = [single_data['input'] for single_data in total_data]
    y = [single_data['y_class'] for single_data in total_data]
    seperate_index = int(len(X) * 0.8)
    train_X = X[:seperate_index]
    train_y = y[:seperate_index]
    test_X = X[seperate_index:]
    test_y = y[seperate_index:]
    clf = ComplementNB()
    clf.fit(train_X,train_y)
    #clf = make_pipeline(StandardScaler(),SGDClassifier(max_iter=1000, tol=1e-3))
    #clf.fit(train_X, train_y)
    y_pred = clf.predict(test_X)
    train_score = accuracy_score(train_y,clf.predict(train_X))
    test_score = accuracy_score(test_y, y_pred)
    print('train_score:{},test_score:{}'.format(round(train_score,3),round(test_score,3)))
    return test_score

from sklearn.naive_bayes import MultinomialNB
def MultinomialNB_classify(total_data,n_clusters):
    X = [single_data['input'] for single_data in total_data]
    y = [single_data['y_class'] for single_data in total_data]
    seperate_index = int(len(X) * 0.8)
    train_X = X[:seperate_index]
    train_y = y[:seperate_index]
    test_X = X[seperate_index:]
    test_y = y[seperate_index:]
    clf = MultinomialNB()
    clf.fit(train_X,train_y)
    #clf = make_pipeline(StandardScaler(),SGDClassifier(max_iter=1000, tol=1e-3))
    #clf.fit(train_X, train_y)
    y_pred = clf.predict(test_X)
    train_score = accuracy_score(train_y,clf.predict(train_X))
    test_score = accuracy_score(test_y, y_pred)
    print('train_score:{},test_score:{}'.format(round(train_score,3),round(test_score,3)))
    return test_score

from sklearn.neural_network import BernoulliRBM
from sklearn import linear_model
from sklearn.pipeline import Pipeline
def BernoulliRBM_classify(total_data,n_clusters):
    X = [single_data['input'] for single_data in total_data]
    y = [single_data['y_class'] for single_data in total_data]
    seperate_index = int(len(X) * 0.8)
    train_X = X[:seperate_index]
    train_y = y[:seperate_index]
    test_X = X[seperate_index:]
    test_y = y[seperate_index:]
    logistic = linear_model.LogisticRegression(C=1)
    rbm1 = BernoulliRBM(n_components=100, learning_rate=0.1, n_iter=100, verbose=1, random_state=101)
    rbm2 = BernoulliRBM(n_components=100, learning_rate=0.1, n_iter=100, verbose=1, random_state=101)
    rbm3 = BernoulliRBM(n_components=100, learning_rate=0.1, n_iter=100, verbose=1, random_state=101)
    DBN3 = Pipeline(steps=[('rbm1', rbm1), ('rbm2', rbm2), ('rbm3', rbm3), ('logistic', logistic)])
    #DBN3 = Pipeline(steps=[ ('logistic', logistic)])
    DBN3.fit(train_X, train_y)

    #clf = make_pipeline(StandardScaler(),SGDClassifier(max_iter=1000, tol=1e-3))
    #clf.fit(train_X, train_y)
    y_pred = DBN3.predict(test_X)
    train_score = accuracy_score(train_y,DBN3.predict(train_X))
    test_score = accuracy_score(test_y, y_pred)
    print('train_score:{},test_score:{}'.format(round(train_score,3),round(test_score,3)))
    return test_score
import torch
import torch.nn as nn
import torch.nn.functional as F
class linear_model_pytorch(nn.Module):
    def __init__(self,in_hsz,hidden_size,out_hsz, dropout=0.1):
        super(linear_model_pytorch,self).__init__()
        self.linear1 = nn.Linear(in_hsz,hidden_size)
        self.BN = nn.BatchNorm1d(in_hsz)
        
        self.smooth = nn.Linear(hidden_size,int(hidden_size/4))
        self.predict = nn.Linear(int(hidden_size/4),out_hsz)
        self.dropout1 = nn.Dropout(dropout)
    def forward(self,x,y):
        #print(x[1])
        x = self.BN(x)
        x = F.relu(self.linear1(x))
        x = self.dropout1(x)
        x = F.relu(self.smooth(x))
        x = self.predict(x)
        y_pred = F.softmax(x, dim=-1)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(x,y.type(torch.long))
        return loss,y_pred

import torch.utils.data as data
from torch.utils.data import DataLoader
class simple_dataset(data.Dataset):
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y
    def __getitem__(self, index):
        return self.X[index],self.Y[index]
    def __len__(self):
        return len(self.X)

def collate_fn(data):
    rs_X = torch.zeros((len(data), 3))
    rs_y = torch.zeros(len(data))
    for idx,inst in enumerate(data):
        X,y = inst[0],inst[1]
        rs_X[idx] = torch.tensor(X)
        rs_y[idx] = torch.tensor(y)
    return rs_X,rs_y
from torch.optim import Optimizer
import tqdm
import numpy as np
def linear_classify(total_data,n_clusters,hidden_size):
    print(total_data)
    X = [[single_data['input'][0]/235,
          single_data['input'][1]/235,
          single_data['input'][2]/235]
         for single_data in total_data]
    
    y = [single_data['y_class'] for single_data in total_data]
    seperate_index = int(len(X) * 0.8)
    train_X = X[:seperate_index]
    train_y = y[:seperate_index]
    test_X = X[seperate_index:]
    test_y = y[seperate_index:]
    device = torch.device('cpu')
    model = linear_model_pytorch(3,hidden_size,n_clusters).to(device)

    train_dataset = simple_dataset(train_X,train_y)
    train_dataloader = DataLoader(train_dataset,batch_size=32,collate_fn=collate_fn,shuffle=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    n_epochs = 50
    es_epoch_cnt = 20
    eval_dataset = simple_dataset(test_X,test_y)
    eval_dataloader = DataLoader(eval_dataset,batch_size=len(test_X),collate_fn=collate_fn,shuffle=False)
    eval_acc_ls = []
    stop_score = 0
    prev_best_score = 0
    es_cnt = 0
    for epoch in tqdm.tqdm(range(n_epochs)):
        model.train(mode=True)
        for idx,batch in enumerate(train_dataloader):
            batch_x,batch_y = batch[0].to(device),batch[1].to(device)
            loss,_ = model(batch_x,batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        for eval_batch in eval_dataloader:
            batch_eval_x,batch_eval_y = eval_batch[0].to(device),eval_batch[1].to(device)
            _,y_pred = model(batch_eval_x,batch_eval_y)
            y_pred = torch.argmax(y_pred,dim = -1).cpu().tolist()
            batch_eval_y = batch_eval_y.cpu().tolist()
            acc_scores = accuracy_score(y_pred,batch_eval_y)
            #print('step:{},acc_score:{}\n'.format(epoch,acc_scores))
            eval_acc_ls.append(acc_scores)
        stop_score = acc_scores
        if stop_score > prev_best_score:
            es_cnt = 0
            prev_best_score = stop_score
            checkpoint = {"model": model.state_dict(), "model_cfg":n_clusters}
            torch.save(checkpoint, 'randomgenerate/model_file/model_dict.ckpt'+str(n_clusters))
        else:
            es_cnt += 1
            if es_cnt > es_epoch_cnt:  # early stop
                pass
    return eval_acc_ls,prev_best_score






