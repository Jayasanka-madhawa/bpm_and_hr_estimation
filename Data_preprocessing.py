import pandas as pd
import numpy as np
random_seed = 42
pd.np.random.seed(random_seed)

def split_train_val_test(input_path):
    df = pd.read_csv(input_path)
    
    df_shuffled = df.sample(frac=1, random_state=random_seed)

    # Calculate the sizes for each subset
    total_samples = len(df_shuffled)
    train_size = int(0.6 * total_samples)
    val_size = int(0.2 * total_samples)
    test_size = total_samples - train_size - val_size

    # Split the DataFrame into three subsets
    train_df = df_shuffled[:train_size]
    val_df = df_shuffled[train_size:train_size+val_size]
    test_df = df_shuffled[train_size+val_size:]

    # Print the sizes of the subsets
    print("Train subset size:", len(train_df))
    print("Validation subset size:", len(val_df))
    print("Test subset size:", len(test_df))
    
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    output_path = './dataframes'
    train_df.to_csv(f'{output_path}/train.csv', index=False)
    val_df.to_csv(f'{output_path}/val.csv', index=False)
    test_df.to_csv(f'{output_path}/test.csv', index=False)

def split_60(input_path,output_path):
    df = pd.read_csv(input_path)

    df_input = df.drop(['hr','calculated_hr','DBP','SBP'], axis=1)

    df_mod=pd.DataFrame()
    l=[]
    for x in range(0,210,30):
        dfx = df_input.iloc[:, x:x+60]
        desired_order = [str(i) for i in range(60)]
        dfx.columns = desired_order
        dfx.loc[:,'hr'] = df['hr'] 
        dfx.loc[:,'calculated_hr'] = df['calculated_hr']
        dfx.loc[:,'DBP'] = df['DBP']
        dfx.loc[:,'SBP'] = df['SBP']
        
    #     display(dfx)
        l.append(dfx)

    df_mod = pd.concat(l)
        
    df_mod = df_mod.reset_index(drop=True)

    df_mod.to_csv(output_path, index=False)
    
    
    
input_path = 'dataframes/data.csv'
output_path = 'dataframes/data60.csv'

split_60(input_path,output_path)

def load_data():

    df_train = pd.read_csv('dataframes/train60.csv')
    df_val = pd.read_csv('dataframes/val60.csv')
    df_test = pd.read_csv('dataframes/test60.csv')

    ratio = len(df_train),len(df_val),len(df_test)
    print(ratio)

    df = pd.concat([df_train ,df_val ,df_test ])
    df = df.reset_index(drop=True)
    
    return df ,ratio