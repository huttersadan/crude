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
            input_generation = [input_100,input_101,input_102]
            output_generation = [output_701,output_702,output_703]
            num_of_generation -= 1
            rs_input_generation.append(input_generation)
            rs_output_generation.append(output_generation)
    return rs_input_generation,rs_output_generation
def get_random_output(rate_of_output,num_of_generation):
    rs_output_generation = []
    if rate_of_output[output_features[0]] < 26.1 * 1000:
        while num_of_generation > 0:
            output_random1 = np.random.randn()
            output_random2 = np.random.randn()
            output_random3 = np.random.randn()
            output_701 = round(rate_of_output[output_features[0]] - abs(output_random1) * 100)
            output_702 = round(rate_of_output[output_features[1]] + output_random2 * 100)
            output_703 = round(rate_of_output[output_features[2]] + output_random3 * 50)
            if output_701 > 0 and output_702 > 0 and output_703 > 0:
                output_generation = [output_701, output_702, output_703]
                num_of_generation -= 1
                rs_output_generation.append(output_generation)

    else:
        while num_of_generation > 0:
            output_random1 = np.random.randn()
            output_random2 = np.random.randn()
            output_random3 = np.random.randn()
            output_701 = round(rate_of_output[output_features[0]] + output_random1 * 0)
            output_702 = round(rate_of_output[output_features[1]] + output_random2 * 0)
            output_703 = round(rate_of_output[output_features[2]] + output_random3 * 0)
            if output_701 > 0 and output_702 > 0 and output_703 > 0:
                output_generation = [output_701,output_702,output_703]
                num_of_generation -=1
                rs_output_generation.append(output_generation)


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
    path_dataset = 'C:/Users/Cooler Master/Desktop/毕设/crude/dataset'
    path_list = os.listdir(path_dataset)

    csv_file_path_list = []
    for file_name in path_list:
        if file_name[-4:] == '.csv':
            csv_file_path_list.append(path_dataset + '/' + file_name)
    input_features = ['FIC-100 - PV', 'FIC-101 - PV', 'FIC-102 - PV']
    output_features = ['FIC10701 - PV', 'FIC10702 - PV', 'FIC10703 - PV']
    #import ipdb;ipdb.set_trace()
    clusters_model = {'kmeans':k_means_model,'sc':sc_Clustering,'Agg':Agglomerative,
                      'birch':birch,'Gauss':Gauss,'Spectral':Spectral}
    csv_files_ls = []
    for i in  range(len(csv_file_path_list)):
        csv_files_ls.append(pd.read_csv(csv_file_path_list[i], skiprows=[j for j in range(9)] + [10]))
    total_random_generate = []
    for temp in csv_files_ls:
        total_random_generate.extend(get_single_csv_random_generate(temp))
    random.shuffle(total_random_generate)

    # n_clusters = 3
    # method_times = {}
    # for cluster_name,model_name in clusters_model.items():
    #     st = time.time()
    #     rs_index_dict,total_data = model_name(total_random_generate,n_clusters)
    #     #print(rs_index_dict)
    #     et = time.time()
    #     print(cluster_name+': {}'.format(round(et-st,6)))
    #     method_times[cluster_name] = [round(et-st,6)]
    # df = pd.DataFrame(method_times)
    # print(df)
    # for names in df.columns:
    #     plt.barh(names,df[names],label = names,align='center')
    # plt.title('Time cost')
    # plt.legend()
    # plt.show()

    # memory_cost = {'kmeans':[8.5],'sc':[1.9],'Agg':[1.3],
    #                   'birch':[0.1],'Gauss':[0.4],'Spectral':[0.5]}
    # df = pd.DataFrame(memory_cost)
    # print(df)
    # for names in df.columns:
    #     plt.barh(names,df[names],label = names,align='center')
    # plt.title('Memory cost')
    # plt.legend()
    # plt.show()




    classify_model ={'SVM':SVM_classify,'DecisionTree':decision_tree_classify,'SGD_classifier':SGD_classify,
                         'knn_classify':knn_classify,'GaussNB':GaussianNB_classify}

    #'linear_classify':linear_classify
    #'DBN':BernoulliRBM_classify
    #'CategoricalNB':CategoricalNB_classify,'ComplementNB':ComplementNB_classify,'MultinomialNB':MultinomialNB_classify,


    # best_scores_dict = {}
    # for n_clusters in range(3,15):
    #     rs_index_dict, total_data = clusters_model['kmeans'](total_random_generate, n_clusters)
    #     best_score = classify_model['knn_classify'](total_data,n_clusters)
    #     print('n_clusters:{},best_score:{}\n'.format(n_clusters,round(best_score,3)))
    #     best_scores_dict[n_clusters] = [best_score]
    # print(best_scores_dict)
    # df = pd.DataFrame(best_scores_dict)
    # print(df)
    # for names in df.columns:
    #     plt.barh(names,df[names],label = names,align='center')
    # plt.title('n_clusters influence')
    # plt.ylabel('n_clusters')
    # plt.xlabel('accuracy')
    # plt.legend()
    # plt.show()
    n_clusters = 3
    best_scores_dict = {}
    rs_index_dict, total_data = clusters_model['kmeans'](total_random_generate, n_clusters)
    print_plot(rs_index_dict)
    # for names,model_name in classify_model.items():
    #     best_score = model_name(total_data,n_clusters)
    #     best_scores_dict[names] = [best_score]
    #eval_acc_ls, best_score = classify_model['SVM'](total_data, n_clusters)
    #print(sum([total_data[idx]['y_class'] == 0 for idx in range(len(total_data))]),sum([total_data[idx]['y_class'] == 1 for idx in range(len(total_data))]),sum([total_data[idx]['y_class'] == 2 for idx in range(len(total_data))]))
    for hidden_size in range(40,101,10):
        eval_acc_ls,best_score = linear_classify(total_data,n_clusters,hidden_size)
        best_scores_dict[str(hidden_size)] = [best_score]
    #hidden_size = 90
    eval_acc_ls, best_score = linear_classify(total_data, n_clusters, hidden_size)
    best_scores_dict[str(hidden_size)] = [best_score]
    print(best_score)
    #df = pandas.DataFrame({'1 hidden layer':[0.99881],'2 hidden layers':[0.42857142857142855],'3 hidden layers':[0.3619047619047619],'4 hidden layers':[0.3619047619047619]})
    #df = pandas.DataFrame({'relu':[0.9285714285714286],'sigmoid':[0.7988095238095239],'tanh':[0.7285714285714285]})
    df = pd.DataFrame(best_scores_dict)
    print(df)
    for names in df.columns:
        plt.bar(names,df[names],label = names,align='center',width=0.3)
    plt.title('hidden_layers influence')
    plt.xlabel('hidden layers')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()
    # print_eval_acc(eval_acc_ls)
    #best_scores_dict = {}
    # for n_clusters in range(2,3):
    #     #n_clusters = 5
    #     rs_index_dict,total_data = clusters_model['kmeans'](total_random_generate,n_clusters)
    #     #print_plot(rs_index_dict)
    #     #print(len(total_data))
    #     #聚类完成了
    #     #total_data是有X，Y和Y_label的
    #     #利用分类模型进行分类
    #     classify_model ={'SVM':SVM_classify,'DecisionTree':decision_tree_classify,'SGD_classifier':SGD_classify,
    #                      'knn_classify':knn_classify,'GaussNB':GaussianNB_classify,'CategoricalNB':CategoricalNB_classify,
    #                      'ComplementNB':ComplementNB_classify,'MultinomialNB':MultinomialNB_classify,'DBN':BernoulliRBM_classify,
    #                      'linear_classify':linear_classify}
    #     eval_acc_ls,best_score = classify_model['linear_classify'](total_data,n_clusters)
    #     #print_eval_acc(eval_acc_ls)
    #     best_scores_dict[n_clusters] = best_score
    # print(best_scores_dict)
    # #{2: 0.9291666666666667, 3: 0.85, 4: 0.7, 5: 0.7791666666666667, 6: 0.625, 7: 0.7208333333333333, 8: 0.7541666666666667, 9: 0.7, 10: 0.9375,
    # # 11: 0.8375, 12: 0.8166666666666667, 13: 0.7541666666666667, 14: 0.7583333333333333, 15: 0.675, 16: 0.6666666666666666, 17: 0.65, 18: 0.525, 19: 0.5291666666666667}
    # rs_index_dict, total_data = clusters_model['kmeans'](total_random_generate,10)
    # classify_model = {'SVM': SVM_classify, 'DecisionTree': decision_tree_classify, 'SGD_classifier': SGD_classify,
    #                   'knn_classify': knn_classify, 'GaussNB': GaussianNB_classify,
    #                   'CategoricalNB': CategoricalNB_classify,
    #                   'ComplementNB': ComplementNB_classify, 'MultinomialNB': MultinomialNB_classify,
    #                   'DBN': BernoulliRBM_classify,
    #                   'linear_classify': linear_classify}
    # eval_acc_ls, best_score = classify_model['linear_classify'](total_data, 10)
    # print(best_score)

