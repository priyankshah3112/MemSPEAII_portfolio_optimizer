import pickle
import os
import pandas as pd
path='month_fundamental_trimmed'
df_dict_oc=dict()
stocknames=os.listdir(path)
for files in stocknames:
    df=pd.read_csv("month_fundamental_trimmed/" + files)
    df.set_index('Date',inplace=True)
    df_dict_oc[files[:-4]]=df['PX_LAST']
with open('df_dict_oc.pkl', 'wb') as f:
    pickle.dump(df_dict_oc, f)
