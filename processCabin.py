###Generate features from 'Cabin'
# Utility method
import re
import pandas as pd
from sklearn import preprocessing


def getCabinLetter(cabin):
    match = re.compile("([a-zA-Z]+)").search(cabin)
    if match:
        return match.group(0)
    else:
        return 'U'


# Utility method
def getCabinNumber(cabin):
    match = re.compile("([0-9]+)").search(cabin)
    if match:
        return match.group(0)
    else:
        return 0


def processCabin(df, keep_binary=False, keep_scaled=False):
    # Replace missing values with "U0"
    df['Cabin'][df.Cabin.isnull()] = 'U0'
    # create feature for the alphabetical part of the cabin number
    df['CabinLetter'] = df['Cabin'].map(lambda x: getCabinLetter(x))
    # change alphbet to number beacause we need tht important feature to regress the age
    df['CabinLetter'] = pd.factorize(df['CabinLetter'])[0]
    # create binary features for each cabin letters
    if keep_binary:
        cletters = pd.get_dummies(df['CabinLetter']).rename(columns=lambda x: 'CabinLetter_' + str(x))
        df = pd.concat([df, cletters], axis=1)
    if keep_scaled:
        # create feature for the numerical part of the cabin number
        df['CabinNumber'] = df['Cabin'].map(lambda x: getCabinNumber(x)).astype(int) + 1
        # scale the number to process as a continuous feature
        scaler = preprocessing.StandardScaler()
        df['CabinNumber_scaled'] = scaler.fit_transform(df['CabinNumber'].values.reshape(-1, 1))
        df['CabinLetter_scaled'] = scaler.fit_transform(df['CabinLetter'].values.reshape(-1, 1))
        del df['CabinNumber']
    del df['CabinLetter']
    return df

from sklearn.ensemble import AdaBoostClassifier