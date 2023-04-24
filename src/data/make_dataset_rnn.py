# -*- coding: utf-8 -*-
# import click
import logging
from pathlib import Path
# from dotenv import find_dotenv, load_dotenv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
# import seaborn as sns


# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    datasets = 4

    for i in range(1,datasets+1):
        train_name = "train_FD00" + str(i) + ".txt"
        test_name = "test_FD00" + str(i) + ".txt"
        
        test_target_df = pd.read_csv(input_filepath+'RUL_FD00'+str(i)+'.txt',header=None)
        train_file = input_filepath + train_name
        test_file = input_filepath + test_name

        df_train = pd.read_csv(train_file,sep=' ',header=None)
        df_test = pd.read_csv(test_file,sep=' ',header=None)
        
        headers = ['unit','cycle','os 1', 'os 2', 'os 3'] #os = operational setting
        for j in range(1,24):
            headers.append('sm ' + str(j)) #sm = sensor measurement
        df_train.set_axis(headers,axis=1,inplace = True)
        df_train = df_train.drop('sm 22',axis=1)
        df_train = df_train.drop('sm 23',axis=1)
        df_test.set_axis(headers,axis=1,inplace = True)
        df_test = df_test.drop('sm 22',axis=1)
        df_test = df_test.drop('sm 23',axis=1)

        # create RUL col to replace cycle
        num_units = df_train['unit'].max()
        train_groups = df_train.groupby('unit')
        
        for j in range(1,num_units+1):
            # cycle to RUL for training data
            total_life = train_groups.get_group(j)['cycle'].max()

            df_train.loc[df_train['unit'] == j, 'cycle'] = total_life - df_train.loc[df_train['unit'] == j, 'cycle']
        df_train.rename(columns={'cycle':'RUL'},inplace=True)
        df_train['RUL'][df_train['RUL'] > 130] = 130 # described in Heimes 2008 paper

        target = df_train['RUL']
        # df_train = df_train.drop('too_soon',axis=1)
        # df_test = df_test.drop('too_soon',axis=1)
        # df_train = df_train.drop('RUL',axis=1)


        df_train.to_csv(output_filepath+'processed_rnn_'+train_name, sep= ',')
        # target.to_csv(output_filepath+'process_target_'+train_name,sep=',')


        num_units_test = df_test['unit'].max()
        print('test units: ' + str(num_units_test))
        test_groups = df_test.groupby('unit')
        for j in range(1,num_units_test+1):
            #cycle to RUL for test data
            final_cycle = int(test_target_df.iloc[j-1])
            num_cycles = test_groups.get_group(j)['cycle'].max()

            df_test.loc[df_test['unit']==j,'cycle'] = num_cycles + final_cycle - df_test.loc[df_test['unit']==j,'cycle']
            # df_test.loc[df_test['unit']==j,'cycle'] = num_cycles+final_cycle+1-df_test.loc[df_test['unit']==j,'cycle']
        df_test.rename(columns={'cycle':'RUL'},inplace=True)
        df_test['RUL'][df_test['RUL'] > 130] = 130 # described in GRU for RUL paper

        df_test.to_csv(output_filepath+'processed_rnn_'+test_name,sep=',')



        train_file = np.loadtxt(output_filepath+'processed_rnn_'+train_name,delimiter=",",skiprows=1)
        test_file  = np.loadtxt(output_filepath+'processed_rnn_'+test_name,delimiter=",",skiprows=1)
        #,unit,RUL,os 1,os 2,os 3,sm 1,sm 2,sm 3,sm 4,sm 5,sm 6,sm 7,sm 8,sm 9,sm 10,sm 11,sm 12,sm 13,sm 14,sm 15,sm 16,sm 17,sm 18,sm 19,sm 20,sm 21
        train_file = train_file[:,1:] #remove row numbers
        test_file = test_file[:,1:]

        scaler = MinMaxScaler(feature_range=(-1,1)) #discussed in GRU for RUL paper, range is now 0 => -1, 1 => 130
        scaler.fit(train_file[:,1:]) #ignore unit number 
        train_file[:,1:] = scaler.transform(train_file[:,1:])
        test_file[:,1:] = scaler.transform(test_file[:,1:])

        np.savetxt(output_filepath+'standardized_'+train_name,train_file,delimiter=",")
        np.savetxt(output_filepath+'standardized_'+test_name,test_file,delimiter=",")

        #create windowed train data sets as in paper (lots of storage for this one)
        N_tws = np.array([0,50,20,30,15])
        N_tw = N_tws[i]

        train_starts = np.zeros((num_units,1))
        train_lengths = np.zeros((num_units,1))
        test_starts = np.zeros((num_units_test,1))
        test_lengths = np.zeros((num_units_test,1))
        #mark start locations in numpy mat for each unit and number of entries
        ind = 0
        for n in range(1,num_units+1):
            while(train_file[ind,0] != n):
                ind += 1
                # train_lengths[n-1] += 1
            train_starts[n-1] = ind
            # train_lengths[n-1] += 1
        for n in range(1,num_units):
            train_lengths[n-1] = train_starts[n] - train_starts[n-1]
        train_lengths[num_units - 1] = train_file.shape[0] - train_starts[num_units-1]
        ind = 0

        train_num_windows = train_lengths - N_tw + 1
        #if want to save the windowed data (big file)
        if(True):
            train_windowed = np.zeros((int(np.sum(train_num_windows))*N_tw,train_file.shape[1]))
            row = 0
            for n in range(num_units):
                for j in range(int(train_num_windows[n])):
                    # print(f"n: {n}")
                    # print(f"j: {j}")


                    start = int(train_starts[n]) + j
                    end = start + N_tw

                
                    train_windowed[row:(row+N_tw),:] = train_file[start:end,:]
                    row += N_tw
            #this is very big file but hopefully can copy to GPU once and then whole thing runs fast
            np.savetxt(output_filepath+'windowed_'+train_name,train_windowed,delimiter=",")


     

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # # find .env automagically by walking up directories until it's found, then
    # # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())
    input_path = "data/external/"
    output_path = "data/processed/"

    main(input_path,output_path)
