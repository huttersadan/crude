
import os
import pandas as pd
import numpy as np
def get_random(process_variable,quality_temperature,num_of_generation):
    random_input_ls,random_output_ls = [process_variable]*num_of_generation,[quality_temperature]*num_of_generation
    return random_input_ls,random_output_ls
def get_data(temp_file):
    process_variable = temp_file[input_features].iloc[-1]
    quality_temperature = temp_file[output_features].iloc[-1]
    num_of_generation = 300
    random_input_ls,random_output_ls  = get_random(process_variable,quality_temperature,num_of_generation)
    
    rs_ls = [{'input':i,'output':n} for i,n in zip(random_input_ls,random_output_ls)]
    return rs_ls

if __name__=='__main__':
    #path_dataset = 'C:/Users/Cooler Master/Desktop/毕设/crude/dataset'
    path_dataset = 'D:\大四上\毕设\dataset\dataset_softsensor'    
    path_list = os.listdir(path_dataset)
    csv_file_path_list = []
    for file_name in path_list:
        if file_name[-4:] == '.csv':
            csv_file_path_list.append(path_dataset + '/' + file_name)
    #input_features = ['FIC-100 - PV', 'FIC-101 - PV', 'FIC-102 - PV']
    #output_features = ['FIC10701 - PV', 'FIC10702 - PV', 'FIC10703 - PV'] 
    input_features = ['TS-1 - Top Stage Pressure',
                      'HMI - G25:',
                      'HMI - G24:',
                      'HMI - G23:',
                      'FIC10705 - PV',
                      'FIC10706 - PV',
                      'FIC10707 - PV',
                      'HMI - B18: Mass Flow',
                      'HMI - B17: Top Stage Pressure',
                      'TIC-102 - PV',
                      'TIC11300 - PV',
                      'TS-1 - Top Stage Temperature',
                      'TS-1 - Stage Temperature (20__TS-1)',
                      'TS-1 - Stage Temperature (12__TS-1)',
                      'TS-1 - Stage Temperature (7__TS-1)'
                      ]
    # input_features = ['TS-1 - Top Stage Pressure',
    #                   'HMI - G22: draw3_q',
    #                   'HMI - G21: draw2_q',
    #                   'HMI - G20: draw1_q',
    #                   'HMI - B18: Mass Flow',
    #                   'HMI - B17: Top Stage Pressure',
    #                   'TIC-102 - PV',
    #                   'TIC11300 - PV',
    #                   'TS-1 - Top Stage Temperature',
    #                   'TS-1 - Stage Temperature (20__TS-1)',
    #                   'TS-1 - Stage Temperature (12__TS-1)',
    #                   'TS-1 - Stage Temperature (7__TS-1)'
    # ]


    #G20-G22 是热量 G23-G25是温差
    output_features = ['Naptha Cut Point - ASTM D86 (Cut Pt-100.00%-Naptha)',
                       'Kero Cut Point - ASTM D86 (Cut Pt-0.0%-Kero Product)',
                       'Kero Cut Point - ASTM D86 (Cut Pt-100.00%-Kero Product)',
                       'Kero Cut Point - ASTM D93 Flash Pt (Kero Product)',
                       'Diesel Cut Point - ASTM D86 (Cut Pt-95.00%-Diesel Product)',
                       'AGO Cut Point - ASTM D86 (Cut Pt-95.00%-AGO Product)']
    csv_files_ls = []
    for i in  range(len(csv_file_path_list)):
        csv_files_ls.append(pd.read_csv(csv_file_path_list[i], skiprows=[j for j in range(9)] + [10]))
    #print(csv_files_ls[-1].head())
    total_random_generate = []
    for temp_file in csv_files_ls:
        total_random_generate.extend(get_data(temp_file))
    print(total_random_generate[-1]['input'])