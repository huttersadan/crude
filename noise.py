import os

import pandas
import pandas as pd
import random
import numpy as np
from clusters import*
import matplotlib.pyplot as plt
from Classification_model import*
import time
import pandas as pd
from memory_profiler import profile
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
random_seed = 114514
random.seed(random_seed)
np.random.seed(random_seed)
plt.rcParams['font.family'] = 'Microsoft YaHei'
# def get_random_input(mass_of_oil,rate_of_output,num_of_generation):
#     rs_output_generation = []
#     while num_of_generation > 0:
#         input_random1 = np.random.randn()
#         input_random2 = np.random.randn()
#         input_random3 = np.random.randn()
#         output_701 = round(rate_of_output[output_features[0]] + input_random1 * 700)
#         output_702 = round(rate_of_output[output_features[1]] + input_random2 * 700)
#         output_703 = round(rate_of_output[output_features[2]] + input_random3 * 700)
#         if output_701 > 0 and output_702 > 0 and output_703 > 0:
#             output_generation = [output_701,output_702,output_703]
#             num_of_generation -= 1
#             rs_output_generation.append(output_generation)
#     return rs_output_generation,rs_output_generation


# def get_single_csv_random_generate(temp):
#     mass_of_oil = temp[input_features].iloc[-1]
#     rate_of_output = temp[output_features].iloc[-1]
#     num_of_generation = 300
#     random_input_ls,random_output_ls  = get_random_input(mass_of_oil,rate_of_output,num_of_generation)
#     rs_ls = [{'input':i,'output':n} for i,n in zip(random_input_ls,random_output_ls)]
#     return rs_ls

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
        ax.scatter([values[i][0] for i in range(len(values))] ,
                   [values[i][1] for i in range(len(values))],
                   [values[i][2] for i in range(len(values))] ,
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

if __name__=='__main__':
    clusters_model = {'kmeans':k_means_model,'sc':sc_Clustering,'Agg':Agglomerative,
                      'birch':birch,'Gauss':Gauss,'Spectral':Spectral}
    means_data_df = pd.read_csv('Classification_dataset.csv',encoding='utf-8-sig')
    means_data = [{'input':[inst1,inst2,inst3],'output':[inst1,inst2,inst3]} for inst1,inst2,inst3 in zip(np.array(means_data_df['常一线产率']),np.array(means_data_df['常二线产率']),np.array(means_data_df['常三线产率']))]
    classify_model ={'SVM':SVM_classify,'DecisionTree':decision_tree_classify,'SGD_classifier':SGD_classify,
                         'knn_classify':knn_classify,'GaussNB':GaussianNB_classify}
    n_clusters = 3
    best_scores_dict = {}
    rs_index_dict, total_data = clusters_model['kmeans'](means_data, n_clusters)
    print_plot(rs_index_dict)
    for dropout in range(1,10,2):
        eval_acc_ls,best_score = linear_classify(total_data,n_clusters,10,dropout/100)
        best_scores_dict[str(dropout/100)] = [best_score]
    df = pd.DataFrame(best_scores_dict)
    for names in df.columns:
        plt.bar(names,df[names],label = names,align='center',width=0.3)
    plt.title('hidden_layers influence')
    plt.xlabel('hidden layers')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()