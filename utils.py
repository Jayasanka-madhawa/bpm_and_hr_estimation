
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def save_data(out_dict, file_path):
    # Inputs:
    #  out_dict : A dictionary like {'layer_index':[0],'deltaT':[2],'acc':[10]}
    if os.path.exists(file_path):
        old_df = pd.read_csv(file_path)
        new_df = pd.DataFrame.from_dict(out_dict)
        out_df = pd.concat([old_df,new_df])
        out_df.to_csv(file_path,index=False)
    else:
        out_df = pd.DataFrame.from_dict(out_dict)
        out_df.to_csv(file_path,index=False)
        
def split_train_test(df_input ,df_target ,ratio):

    X_train, y_train, X_val, y_val, X_test, y_test = (df_input[:ratio[0]],df_target[:ratio[0]],
                                            df_input[ratio[0]:ratio[0]+ratio[1]],df_target[ratio[0]:ratio[0]+ratio[1]], 
                                            df_input[-ratio[2]:],df_target[-ratio[2]:])
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def plot_scatter_bpm(y_predicted_df,y_test,folder_path):
    plt.figure(figsize=(10,6))
    plt.scatter(y_predicted_df[0],y_test[0],marker='.')
    plt.scatter(y_predicted_df[1],y_test[1],marker='.')
    plt.ylabel('Actual values')
    plt.xlabel('Predicted values')
    plt.axline((40, 40), slope=1 ,color='g')
    plt.savefig(f'{folder_path}/predictions.jpeg')
    
def plot_loss(history_dict ,folder_path):
    plt.figure(figsize=(10,6))
    plt.plot(history_dict['loss'])
    plt.plot(history_dict['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    # plt.ylim(0,250)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f'{folder_path}/loss.jpeg')
    
def calculate_loss_bpm(y_predicted_df,y_test):
    mae_dbp = np.mean(np.abs(y_predicted_df[0].values - y_test[0].values))
    print("DBP-MAE:", mae_dbp)
    mae_sbp = np.mean(np.abs(y_predicted_df[1].values - y_test[1].values))
    print("SBP-MAE:", mae_sbp)
    mse_dbp = np.mean(np.square(y_predicted_df[0].values - y_test[0].values))
    print("DBP-MSE:", mse_dbp)
    mse_sbp = np.mean(np.square(y_predicted_df[1].values - y_test[1].values))
    print("SBP-MSE:", mse_sbp)
    
    return mae_dbp ,mae_sbp ,mse_dbp ,mse_sbp

def plot_scatter_hr(y_predicted_df,y_test,folder_path):
    plt.figure(figsize=(10,6))
    plt.scatter(y_predicted_df,y_test,marker='.')
    plt.ylabel('Actual values')
    plt.xlabel('Predicted values')
    plt.axline((40, 40), slope=1 ,color='g')
    plt.savefig(f'{folder_path}/predictions.jpeg')
    
def calculate_loss_hr(y_predicted_df,y_test):
    mae = np.mean(np.abs(y_predicted_df.values - y_test.values))
    print("MAE:", mae)

    mse = np.mean(np.square(y_predicted_df.values - y_test.values))
    print("MSE:", mse)

    return mae , mse
    
