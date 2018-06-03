import numpy as np
import pandas as pd

def remove_corr(data):
    df_corr=data.drop(['Survived'],axis=1).corr(method='spearman')
    mask=np.ones(df_corr.columns.size)-np.eye(df_corr.columns.size)
    df_corr=df_corr*mask
    drops=[]
    for col in df_corr.columns.values:
        if np.in1d([col],drops):
            continue
        corr=df_corr.index[abs(df_corr[col])>0.9].values
        drops=np.union1d(drops,corr)
    print("\nDropping",drops.shape[0],"highly correlated features")
    data_nocorr=data.drop(drops,axis=1,inplace=False)
    return data_nocorr
