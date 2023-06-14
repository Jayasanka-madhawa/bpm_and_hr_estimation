import pandas as pd
import os
import numpy as np
import itertools
import pickle
pd.options.display.max_columns = 100

hr_list = []
bvp_list = []
DBP_list = []
SBP_list = []
calculated_hr_list = []

def collect_data(output_file):

    dirs = os.listdir('./rgb_and_bvp_data')
    dirs_real = os.listdir('./raw_data')

    for directory in dirs:
        directory_path = f'./rgb_and_bvp_data/{directory}'
        windows = len(os.listdir(directory_path+'/time_series/bvp/bvp_B/'))    
        
        i = directory_path[-15:]

        ins = pd.read_csv(directory_path+'/instantaneous.csv')
            
        for d_real in dirs_real:
            gt = pd.read_excel(f'./raw_data/{d_real}')
            is_present_in_first_column = i in gt['Unnamed: 18'].values
            is_present_in_second_column = i in gt['Unnamed: 21'].values
            if is_present_in_first_column:
                dbp = gt.loc[gt['Unnamed: 18'] == i, 'Unnamed: 14'].values
                dbp = dbp[0]
                sbp = gt.loc[gt['Unnamed: 18'] == i, 'Ground Truth reading'].values
                sbp = sbp[0]
                hr = gt.loc[gt['Unnamed: 18'] == i, 'Unnamed: 17'].values
                hr = hr[0]
            elif is_present_in_second_column:
                dbp = gt.loc[gt['Unnamed: 25'] == i, 'Unnamed: 21'].values
                dbp = dbp[0]
                sbp = gt.loc[gt['Unnamed: 25'] == i, 'Ground Truth reading.1'].values
                sbp = sbp[0]
                hr = gt.loc[gt['Unnamed: 25'] == i, 'Unnamed: 24'].values
                hr = hr[0]

        for window_num in range(1,windows+1):
            file_path = f'/time_series/bvp/bvp_B/bvp_B_window_{window_num}.pckl'
            bvp = pd.read_pickle(directory_path+file_path)
            bvp = np.array(bvp)
                
            calculated_hr = ins[(ins['Window num']==window_num) & (ins['ROI No']==0)].HR.values
            hr_list.append(hr)
            if len(bvp)>256:
                bvp_list.append(bvp[:-1])
            else:
                bvp_list.append(bvp)
            DBP_list.append(dbp)
            SBP_list.append(sbp)
            calculated_hr_list.append(calculated_hr[0])
            
    df = pd.DataFrame(bvp_list)
    df['hr'] = hr_list
    df['calculated_hr'] = calculated_hr_list
    df['DBP']= DBP_list
    df['SBP']= SBP_list

    df.to_csv(output_file, index=False)
    
    
    
collect_data('dataframes/data.csv')
