std = 1.56e+005
heavy = 1000
light = 7.8e+004

total = std + heavy+light
print(total)

import random
total_list = [i for i in range(100,int(total),100)]
random.seed(114514)
import numpy as np
rs_list = []



for a in range(10000,200000,4000):
    for c in range(1000,235000-a,4000):
        b = 235000-a-c
        rs_list.append([a,b,c])
np_rs = np.array(rs_list)
print(np_rs)

import pandas as pd
#df = pd.DataFrame({'a':[rs_list[i][0] for i in range(len(rs_list)) ],'b':[rs_list[i][1] for i in range(len(rs_list)) ],'c':[rs_list[i][2] for i in range(len(rs_list)) ]})
import scipy.io as scio
mat_dict = {'a':np_rs[:,0],'b':np_rs[:,1],'c':np_rs[:,2]}
#print(mat_dict)
scio.savemat('random_input1.mat', mat_dict)

data = scio.loadmat('random_input1.mat')
print(data['a'].shape)