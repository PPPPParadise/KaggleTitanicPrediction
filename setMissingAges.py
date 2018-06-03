import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing

def setMissingAges(df):


    age_df = df[['Age','Fare_scaled','Fare_bin_id_scaled',	'SibSp_scaled',	'Parch_scaled',	'Embarked_scaled'	,'CabinNumber_scaled',
                 'CabinLetter_scaled','Names_scaled', 'Title_id_scaled','Pclass_scaled','Gender_scaled',	'TicketNumber_scaled',	'TicketPrefix_id_scaled']]
    knownAge=age_df[df.Age.notnull()]
    unknownAge=age_df[df.Age.isnull()]
    y=knownAge.values[:,0]
    X=knownAge.values[:,1:]
    rfr=RandomForestRegressor(n_estimators=2000,n_jobs=-1)
    #train the regressor
    rfr.fit(X,y)
    predictedAges=rfr.predict(unknownAge.values[:,1:])
    scaler = preprocessing.StandardScaler()
    df['Age'][df.Age.isnull()]=predictedAges
    scer = preprocessing.StandardScaler()
    df['New'] = scer.fit_transform(df['Age'].values.reshape(-1, 1))
    df['Age_scaled'] = df['New']
    del df['New']
    df['Age2'] = pd.qcut(df['Age'], 5)
    df['Age22'] = pd.factorize(df['Age2'])[0] + 1
    df['Agebinned'] = scer.fit_transform(df['Age22'].values.reshape(-1, 1))
    return df