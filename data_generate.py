import os

import pandas
import pandas as pd
import random
import numpy as np
from clusters import*
import matplotlib.pyplot as plt
from classfymodel import*

from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
random_seed = 114514
random.seed(random_seed)
np.random.seed(random_seed)
def get_random_input(mass_of_oil,rate_of_output,num_of_generation):
    rs_output_generation = []
    while num_of_generation > 0:
        input_random1 = np.random.randn()
        input_random2 = np.random.randn()
        input_random3 = np.random.randn()
        output_701 = round(rate_of_output[output_features[0]] + input_random1 * 700)
        output_702 = round(rate_of_output[output_features[1]] + input_random2 * 700)
        output_703 = round(rate_of_output[output_features[2]] + input_random3 * 700)
        if output_701 > 0 and output_702 > 0 and output_703 > 0:
            output_generation = [output_701,output_702,output_703]
            num_of_generation -= 1
            rs_output_generation.append(output_generation)
    return rs_output_generation,rs_output_generation
from mpl_toolkits.mplot3d import Axes3D

def get_single_csv_random_generate(temp):
    mass_of_oil = temp[input_features].iloc[-1]
    rate_of_output = temp[output_features].iloc[-1]
    num_of_generation = 300
    random_input_ls,random_output_ls  = get_random_input(mass_of_oil,rate_of_output,num_of_generation)

    #for i in range(num_of_generation):
        
    rs_ls = [{'input':i,'output':n} for i,n in zip(random_input_ls,random_output_ls)]
    
    # 生成1200组三维数据
    


    return rs_ls
plt.rcParams['font.family'] = 'Microsoft YaHei'
def print_plot(rs_index_dict):
    colours = ['dimgray','lightcoral','darkred','sienna',
               'darkorange','tan','gold','olive','lawngreen',
               'palegreen','lime','turquoise','teal','powderblue',
               'dodgerblue','blue','darkorchid','fuchsia',
               'deeppink','crimson']
    random.shuffle(colours)
    merge_crude = ['A','B','C']
    markers = ["o","v","+","o","v","+","o","v","+","o","v","+","o","v","+"
               ,"o","v","+","o","v","+","o","v","+","o","v","+","o","v","+","o","v","+"
               ,"o","v","+","o","v","+","o","v","+","o","v","+"]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for idx,values in rs_index_dict.items():
        ax.scatter([values[i][0]/23500 for i in range(len(values))] ,
                   [values[i][1]/23500 for i in range(len(values))],
                   [values[i][2]/23500 for i in range(len(values))] ,
                   c=colours[idx],
                   marker=markers[idx],label = merge_crude[idx])
    ax.legend(loc = "center left")
    ax.set_xlabel('Kero')
    ax.set_ylabel('Diesel')
    ax.set_zlabel('AGO')
    ax.set_title('聚类后的结果',color = 'red')
    #plt.savefig('3_2.png')
    plt.show()

def print_eval_acc(eval_acc_ls):
    plt.figure()
    plt.plot(eval_acc_ls)
    plt.title('eval accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()
import time
import pandas as pd
from memory_profiler import profile
import json
if __name__=='__main__':
    path_dataset = 'dataset'
    path_list = os.listdir(path_dataset)
    csv_file_path_list = []
    for file_name in path_list:
        if file_name[-4:] == '.csv':
            csv_file_path_list.append(path_dataset + '/' + file_name)
    input_features = ['FIC10701 - PV', 'FIC10702 - PV', 'FIC10703 - PV']
    output_features = ['FIC10701 - PV', 'FIC10702 - PV', 'FIC10703 - PV']
    clusters_model = {'kmeans':k_means_model,'sc':sc_Clustering,'Agg':Agglomerative,
                      'birch':birch,'Gauss':Gauss,'Spectral':Spectral}
    csv_files_ls = []
    for i in  range(len(csv_file_path_list)):
        csv_files_ls.append(pd.read_csv(csv_file_path_list[i], skiprows=[j for j in range(9)] + [10]))
    total_random_generate = []
    for temp in csv_files_ls:
        total_random_generate.extend(get_single_csv_random_generate(temp))
    random.shuffle(total_random_generate)
    data = np.array([[output['output'][0]/235000,output['output'][1]/235000,output['output'][2]/235000] for output in total_random_generate])

    
    df = pd.DataFrame({'常一线产率':np.round(data[:,0],3),'常二线产率':np.round(data[:,1],3),'常三线产率':np.round(data[:,2],3)})
    df.to_csv('Classification_dataset.csv', index=False,encoding='utf-8-sig')
    