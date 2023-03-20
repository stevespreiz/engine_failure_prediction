# -*- coding: utf-8 -*-
# import click
import logging
from pathlib import Path
# from dotenv import find_dotenv, load_dotenv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        for i in range(1,24):
            headers.append('sm ' + str(i)) #sm = sensor measurement
        df_train.set_axis(headers,axis=1,inplace = True)
        df_train = df_train.drop('sm 22',axis=1)
        df_train = df_train.drop('sm 23',axis=1)
        df_test.set_axis(headers,axis=1,inplace = True)
        df_test = df_test.drop('sm 22',axis=1)
        df_test = df_test.drop('sm 23',axis=1)

        # plan to create two versions of the training data
        # 1. typical normalized features on just sample average 
        # 2. running summaries like in the reference machine failure project

        # 1. normalized features

        # 2. running summaries
        feature_window = 7 # number of cycles to average over
        
        df_train['too_soon'] = np.where((df_train.cycle < feature_window),1,0)
        df_test['too_soon'] = np.where((df_test.cycle < feature_window),1,0)
        
       
        for j in range(1,4):
            #loop operational settings
            col = 'os '+str(j)
            df_train[col+'_mean'] = np.where((df_train.too_soon == 0), \
                (df_train[col].rolling(min_periods=1, window=feature_window).mean()) , df_train[col])
            df_train[col+'_med'] = np.where((df_train.too_soon == 0), \
                (df_train[col].rolling(min_periods=1, window=feature_window).median()) , df_train[col])
            df_train[col+'_max'] = np.where((df_train.too_soon == 0), \
                (df_train[col].rolling(min_periods=1, window=feature_window).max()) , df_train[col])
            df_train[col+'_min'] = np.where((df_train.too_soon == 0), \
                (df_train[col].rolling(min_periods=1, window=feature_window).min()) , df_train[col])
            df_test[col+'_mean'] = np.where((df_test.too_soon == 0), \
                (df_test[col].rolling(min_periods=1, window=feature_window).mean()) , df_test[col])
            df_test[col+'_med'] = np.where((df_test.too_soon == 0), \
                (df_test[col].rolling(min_periods=1, window=feature_window).median()) , df_test[col])
            df_test[col+'_max'] = np.where((df_test.too_soon == 0), \
                (df_test[col].rolling(min_periods=1, window=feature_window).max()) , df_test[col])
            df_test[col+'_min'] = np.where((df_test.too_soon == 0), \
                (df_test[col].rolling(min_periods=1, window=feature_window).min()) , df_test[col])

        for j in range(1,22):
            #loop sensor outputs
            col = 'sm '+str(j)
            df_train[col+'_mean'] = np.where((df_train.too_soon == 0), \
                (df_train[col].rolling(min_periods=1, window=feature_window).mean()) , df_train[col])
            df_train[col+'_med'] = np.where((df_train.too_soon == 0), \
                (df_train[col].rolling(min_periods=1, window=feature_window).median()) , df_train[col])
            df_train[col+'_max'] = np.where((df_train.too_soon == 0), \
                (df_train[col].rolling(min_periods=1, window=feature_window).max()) , df_train[col])
            df_train[col+'_min'] = np.where((df_train.too_soon == 0), \
                (df_train[col].rolling(min_periods=1, window=feature_window).min()) , df_train[col])
            df_train[col+'_chg'] = np.where((df_train[col+'_mean']==0),0,df_train[col]/df_train[col+'_mean'])
            df_test[col+'_mean'] = np.where((df_test.too_soon == 0), \
                (df_test[col].rolling(min_periods=1, window=feature_window).mean()) , df_test[col])
            df_test[col+'_med'] = np.where((df_test.too_soon == 0), \
                (df_test[col].rolling(min_periods=1, window=feature_window).median()) , df_test[col])
            df_test[col+'_max'] = np.where((df_test.too_soon == 0), \
                (df_test[col].rolling(min_periods=1, window=feature_window).max()) , df_test[col])
            df_test[col+'_min'] = np.where((df_test.too_soon == 0), \
                (df_test[col].rolling(min_periods=1, window=feature_window).min()) , df_test[col])
            df_test[col+'_chg'] = np.where((df_test[col+'_mean']==0),0,df_test[col]/df_test[col+'_mean'])

        # create RUL col to replace cycle
        num_units = df_train['unit'].max()
        train_groups = df_train.groupby('unit')
        
        for j in range(1,num_units+1):
            # cycle to RUL for training data
            total_life = train_groups.get_group(j)['cycle'].max()

            df_train.loc[df_train['unit'] == j, 'cycle'] = total_life - df_train.loc[df_train['unit'] == j, 'cycle']
        df_train.rename(columns={'cycle':'RUL'},inplace=True)

        target = df_train['RUL']
        df_train = df_train.drop('too_soon',axis=1)
        df_test = df_test.drop('too_soon',axis=1)
        # df_train = df_train.drop('RUL',axis=1)


        df_train.to_csv(output_filepath+'processed_features_'+train_name, sep= ',')
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


        # test_target_df.set_axis(["RUL"],axis=1,inplace=True)
        # df_test['RUL'] = test_target_df['RUL']


        df_test.to_csv(output_filepath+'processed_features_'+test_name,sep=',')






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
