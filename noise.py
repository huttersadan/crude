import os

import pandas
import pandas as pd
import random
import numpy as np
from clusters import*
import matplotlib.pyplot as plt
from classfymodel import*

from mpl_toolkits.mplot3d import Axes3D
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
def get_single_csv_random_generate(temp):
    mass_of_oil = temp[input_features].iloc[-1]
    rate_of_output = temp[output_features].iloc[-1]
    num_of_generation = 300
    random_input_ls,random_output_ls  = get_random_input(mass_of_oil,rate_of_output,num_of_generation)
    rs_ls = [{'input':i,'output':n} for i,n in zip(random_input_ls,random_output_ls)]
    return rs_ls

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
        ax.scatter([values[i][0]/1000 for i in range(len(values))] ,
                   [values[i][1]/1000 for i in range(len(values))],
                   [values[i][2]/1000 for i in range(len(values))] ,
                   c=colours[idx],
                   marker=markers[idx],label = merge_crude[idx])
    ax.legend(loc = "right")
    ax.set_xlabel('Kero')
    ax.set_ylabel('Diesel')
    ax.set_zlabel('AGO')
    ax.set_title('clusters plot',color = 'red')
    
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
    classify_model ={'SVM':SVM_classify,'DecisionTree':decision_tree_classify,'SGD_classifier':SGD_classify,
                         'knn_classify':knn_classify,'GaussNB':GaussianNB_classify}
    n_clusters = 3
    best_scores_dict = {}
    rs_index_dict, total_data = clusters_model['kmeans'](total_random_generate, n_clusters)
    print_plot(rs_index_dict)
    for hidden_size in range(40,101,30):
        eval_acc_ls,best_score = linear_classify(total_data,n_clusters,hidden_size)
        best_scores_dict[str(hidden_size)] = [best_score]
    df = pd.DataFrame(best_scores_dict)
    print(df)
    for names in df.columns:
        plt.bar(names,df[names],label = names,align='center',width=0.3)
    plt.title('hidden_layers influence')
    plt.xlabel('hidden layers')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()