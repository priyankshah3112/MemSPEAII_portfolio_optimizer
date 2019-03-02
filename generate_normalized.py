import os
import pandas as pd
path='month_fundamental_trimmed'
df_dict_oc=dict()
stocknames=os.listdir(path)
# for files in stocknames:
#     df=pd.read_csv(path+'/'+files)
#     df.to_csv('data/'+files[:-4]+'/'+files,index=False)

# for files in stocknames:
#     df=pd.read_csv('data/'+files[:-4]+'/'+files)
#     df=df[:72]
#     df.to_csv('data/' + files[:-4] + '/' + files[:-4]+'_train.csv', index=False)
#
import pickle
import os
import pandas as pd
path='data'
df_dict_norm=dict()
stocknames=os.listdir(path)
for files in stocknames:
    print(files)
    df=pd.read_csv(path +'/'+ files+'/'+ files+'_normalized.csv')
    df.set_index('Date',inplace=True)
    df.drop(['Open','Close','High','Low','Volume'],axis=1,inplace=True)
    print(df.tail())
    df_dict_norm[files]=df
with open('df_dict_norm.pkl', 'wb') as f:
    pickle.dump(df_dict_norm, f)
# for files in stocknames:
#     print("=============================")
#     print(files)
#     df1=pd.read_csv('data/' + files[:-4] + '/' + files[:-4]+'_train.csv')
#     df1.set_index(['Date'],inplace=True)
#     col = df1.columns.values
#     for c in col:
#         print(c)
#         if df1[c].max() == 0 and df1[c].min() == 0:
#             df1[c] = 0
#         elif df1[c].max() == df1[c].min():
#             df1[c] = 1
#         else:
#             df1[c] = (df1[c] - df1[c].min()) / (df1[c].max() - df1[c].min())
#
#     df1.to_csv('data/' + files[:-4] + '/' + files[:-4] + '_normalized.csv')