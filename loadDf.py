import numpy as np
import pandas as pd
def loadDf(path):
    '''
    load data from disk or online to train, test and all\r
    :param path:  the str of data's location\r
    :return:  return train_df,test_df,df\r
    '''
    train_df=pd.read_csv(path+'train.csv')
    test_df=pd.read_csv(path+'test.csv')
    df=pd.concat([train_df,test_df])
    df.reset_index(inplace=True)
    df.drop('index',axis=1,inplace=True)
    df=df.reindex_axis(train_df.columns,axis=1)
    return train_df,test_df,df
