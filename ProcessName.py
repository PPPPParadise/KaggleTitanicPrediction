###Generate features from the 'Name' variable
import numpy as np
import pandas as pd
from sklearn import preprocessing



def processName(df, keep_binary=False, keep_bins=False, keep_scaled=False, delName = False):
    """
    Parameters:
        keep_binary:include 'Title_Mr' 'Title_Mrs'...
        keey_scaled&&keep_bins:include 'Names_scaled' 'Title_id_scaled'
    Note: the string feature 'Name' can be deleted
    """
    # how many different names do they have? this feature 'NNames'
    # We assume that if one has more than one names, often in quotes, he will be more ''important'
    import re
    df['NNames'] = df['Name'].map(lambda x: len(re.split('\\(', x)))

    # what is each person's title?
    df['Title'] = df['Name'].map(lambda x: re.compile(", (.*?)\.").findall(x)[0])
    # group low-occuring,related titles together
    # make it to 4 groups
    df['Title'][df.Title.isin(['Mr', 'Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col', 'Sir'])] = 'Mr'
    df['Title'][df.Title.isin(['Master'])] = 'Master'
    df['Title'][df.Title.isin(['Countess', 'Mme', 'Mrs', 'Lady', 'the Countess', 'Dona'])] = 'Mrs'
    df['Title'][df.Title.isin(['Mlle', 'Ms', 'Miss'])] = 'Miss'
    df['Title'][(df.Title.isin(['Dr'])) & (df['Sex'] == 'male')] = 'Mr'
    df['Title'][(df.Title.isin(['Dr'])) & (df['Sex'] == 'female')] = 'Mrs'
    df['Title'][df.Title.isnull()][df['Sex'] == 'male'] = 'Master'
    df['Title'][df.Title.isnull()][df['Sex'] == 'female'] = 'Miss'
    # build binary features
    if keep_binary:
        df = pd.concat([df, pd.get_dummies(df['Title']).rename(columns=lambda x: 'Title_' + str(x))], axis=1)
    # process_scaled
    if keep_scaled:
        scaler = preprocessing.StandardScaler()
        df['Names_scaled'] = scaler.fit_transform(df['NNames'].values.reshape(-1,1))
    if keep_bins:
        df['Title_id'] = pd.factorize(df['Title'])[0] + 1
    if keep_bins and keep_scaled:
        scaler = preprocessing.StandardScaler()
        df['Title_id_scaled'] = scaler.fit_transform(df['Title_id'].values.reshape(-1,1))
    if delName:
        del df['Name']
    return df



##TODO: we may want to make other aggreate to make the title better
# the describe is here
Title_des = False
if  Title_des:
    from loadDf import loadDf
    x,xx,df = loadDf('')
    a = processName(df)
    b = a[['Survived', 'Title', 'NNames']][(1 - a.Survived.isnull()).astype('bool')]
    print('the Mr die and alive')
    print(np.count_nonzero(b.Survived[b['Title'] == 'Mr'] == 0))
    print(np.count_nonzero(b.Survived[b['Title'] == 'Mr'] == 1))
    print('the Mrs die and alive')
    print(np.count_nonzero(b.Survived[b['Title'] == 'Master'] == 0))
    print(np.count_nonzero(b.Survived[b['Title'] == 'Master'] == 1))
    print('the Miss die and alive')
    print(np.count_nonzero(b.Survived[b['Title'] == 'Miss'] == 0))
    print(np.count_nonzero(b.Survived[b['Title'] == 'Miss'] == 1))
    print('the Master die and alive')
    print(np.count_nonzero(b.Survived[b['Title'] == 'Master'] == 0))
    print(np.count_nonzero(b.Survived[b['Title'] == 'Master'] == 1))

