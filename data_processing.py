import pandas as pd
import numpy as np
from interaction import *
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from loadDf import *
from processCabin import *
from processFare import *
from fill_n_scale import *
from ProcessName import *
from setMissingAges import *
from remove_corr import *
'''import functions and modify features'''

path = 'C:\\Users\\txy9r\\Desktop\\ML\\titanic\\'
train_df,test_df,df  = loadDf(path)
processFare(df,keep_bins=1,keep_scaled=1)
processFamily(df,keep_scaled=1)
processEmbarked(df,keep_scaled=1)
processCabin(df,keep_scaled=1)
processName(df,keep_scaled=1,keep_bins=1,delName=1)
processPclass(df,keep_scaled=1)
processSex(df,keep_scaled=1)
processTicket(df,keep_scaled=1,keep_bins=1)
setMissingAges(df)
# pd.DataFrame.to_csv(df,'scaled.csv')
df1 = df.iloc[:,np.array([2,4,10,11,12,13,14,15,16,19,21,22,24,25,26,27,30])-1]
# here we just keep scaled features
df2 = interact(df1)

pd.DataFrame.to_csv(df2,'dfdf.csv',index=False)
df_uncorr = remove_corr(df2)
pd.DataFrame.to_csv(df_uncorr,'df_uncorr.csv',index=False)