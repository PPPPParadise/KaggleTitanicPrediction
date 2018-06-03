import re
import pandas as pd
import numpy as np
from sklearn import preprocessing

def getTicketPrefix(ticket):
    match = re.compile("([a-zA-Z\.\/]+)").search(ticket)
    if match:
        return match.group(0)
    else:
        return 'U'


def getTicketNumber(ticket):
    match = re.compile("([0-9]+)").search(ticket)
    if match:
        return match.group(0)
    else:
        return '0'


def processTicket(df, keep_binary=False, keep_bins=False, keep_scaled=False):
    df['TicketPrefix'] = df['Ticket'].map(lambda x: getTicketPrefix(x.upper()))
    df['TicketPrefix'] = df['TicketPrefix'].map(lambda x: re.sub('[\.?\/?]', '', x))
    df['TicketPrefix'] = df['TicketPrefix'].map(lambda x: re.sub('STON', 'SOTON', x))

    df['TicketNumber'] = df['Ticket'].map(lambda x: getTicketNumber(x))
    df['TicketNumberStart'] = df['TicketNumber'].map(lambda x: x[0]).astype(np.int)

    if keep_binary:
        numberstart = pd.get_dummies(df['TicketNumberStart']).rename(columns=lambda x: 'TicketNumberStart_' + str(x))
        df = pd.concat([df, numberstart], axis=1)
    if keep_bins:
        # help the interactive feature process,lift by 1
        df['TicketPrefix_id'] = pd.factorize(df['TicketPrefix'])[0] + 1
    if keep_scaled:
        scaler = preprocessing.StandardScaler()
        df['TicketNumber_scaled'] = scaler.fit_transform(df['TicketNumber'].values.reshape(-1,1))
        df['TicketPrefix_id_scaled'] = scaler.fit_transform(df['TicketPrefix_id'].values.reshape(-1,1))
    del df['Ticket'], df['TicketNumber'], df['TicketPrefix'], df['TicketNumberStart'], df['TicketPrefix_id']
    return df


def processFare(df, keep_binary=False, keep_bins=False, keep_scaled=False):
    # replace missing values with the median of the coressponding class
    df.loc[(df.Fare.isnull()) & (df.Pclass == 1), 'Fare'] = np.median(df[df['Pclass'] == 1]['Fare'].dropna())
    df.loc[(df.Fare.isnull()) & (df.Pclass == 2), 'Fare'] = np.median(df[df['Pclass'] == 2]['Fare'].dropna())
    df.loc[(df.Fare.isnull()) & (df.Pclass == 3), 'Fare'] = np.median(df[df['Pclass'] == 3]['Fare'].dropna())
    # lift zeros values to 1/10 of the minium because we will add interactive features
    df['Fare'][np.where(df['Fare'] == 0)[0]] = df['Fare'][df['Fare'].nonzero()[0]].min() / 10
    if keep_bins:
        df['Fare_bin'] = pd.qcut(df['Fare'], 4)
        df['Fare_bin_id'] = pd.factorize(df['Fare_bin'])[0] + 1
    if keep_binary:
        df = pd.concat([df, pd.get_dummies(df['Fare_bin']).rename(columns=lambda x: 'Fare_bin_' + str(x))], axis=1)
    if keep_scaled:
        scaler = preprocessing.StandardScaler()
        df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1, 1))
    if keep_bins and keep_scaled:
        scaler = preprocessing.StandardScaler()
        df['Fare_bin_id_scaled'] = scaler.fit_transform(df['Fare_bin_id'].values.reshape(-1, 1))
        del df['Fare_bin'], df['Fare_bin_id']
    return df

