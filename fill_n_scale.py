import pandas as pd
import numpy as np
from sklearn import preprocessing

def processPclass(df, keep_binary=False, keep_scaled=False):
    '''

    :param df:  input data;
    :param keep_binary:  if we want some binary feature
    :param keep_scaled:  str-->float, also easy to calculate
    :return:  transformed df
    '''


    # create binary features
    # to generate more features for some linear models
    if keep_binary:
        df = pd.concat([df, pd.get_dummies(df['Pclass']).rename(columns=lambda x: 'Pclass_' + str(x))], axis=1)
    if keep_scaled:
        scaler = preprocessing.StandardScaler()
        df['Pclass_scaled'] = scaler.fit_transform(df['Pclass'].values.reshape(-1, 1))
    return df


###Generate feature from 'Sex' variable
def processSex(df,keep_scaled = False):
    '''
    from sex to gender, 1 for male
    make it scaled to do some interaction later
    :param df:
    :return: df
    '''
    df['Gender'] = np.where(df['Sex'] == 'male', 1, 0)
    if keep_scaled:
        scaler = preprocessing.StandardScaler()
        df['Gender_scaled'] = scaler.fit_transform(df['Gender'].values.reshape(-1, 1))

    del df['Sex']
    return df

def processFamily(df,keep_binary=False,keep_scaled=False):
    '''
    Generate feature from 'SibSp' and 'Parch'
    :param df:
    :param keep_binary:
    :param keep_scaled:
    :return:
    '''
    #interaction variables require no zeros ,lift up everything
    # df['SibSp']=df['SibSp']+1
    # df['Parch']=df['Parch']+1

    if keep_binary:
        sibsps=pd.get_dummies(df['SibSp']).rename(columns=lambda x:'SibSp_'+str(x))
        parchs=pd.get_dummies(df['Parch']).rename(columns=lambda x:'Parch_'+str(x))
        df=pd.concat([df,sibsps,parchs],axis=1)
    if keep_scaled:
        scaler=preprocessing.StandardScaler()
        df['SibSp_scaled']=scaler.fit_transform(df['SibSp'].values.reshape(-1, 1))
        df['Parch_scaled']=scaler.fit_transform(df['Parch'].values.reshape(-1, 1))
    return df

def processEmbarked(df,keep_binary=False,keep_scaled=False):
    #replace the missing values with most common port
    df['Embarked'][df['Embarked'].isnull()]=df.Embarked.dropna().mode().values
    #turn into number
    df['Embarked']=pd.factorize(df['Embarked'])[0]
    # Create binary features for each port
    if keep_binary:
        df = pd.concat([df, pd.get_dummies(df['Embarked']).rename(columns=lambda x: 'Embarked_' + str(x))], axis=1)
    if keep_scaled:
        scaler=preprocessing.StandardScaler()
        df['Embarked_scaled']=scaler.fit_transform(df['Embarked'].values.reshape(-1,1))
    return df


