import pandas as pd
import os
count=0
for file in os.listdir('month_fundamental_trimmed'):
    df=pd.read_csv("month_fundamental_trimmed/"+file)
    # df['PE']= df['PX_LAST'] / df['IS_EPS']
    # df['PE']=df['PE'].round(4)

    if df.isnull().values.any():
        print(count)
        count+=1
        print(file)
    # df.drop(['T12M_DIL_PE_CONT_OPS','DVD_PAYOUT_RATIO_NORMALIZED_EARN','IS_DILUTED_EPS','TRAIL_12M_EPS','TRAIL_12M_DVD_PER_SH','TRAIL_12M_EBITDA','IS_TOTAL_EXPENDITURES','INVENT_TURN','MOV_AVG_50D','MOV_AVG_20D','MOV_AVG_30D','MOV_AVG_100D','VOLATILITY_20D','VOLATILITY_60D','VOLATILITY_90D','RSI_30D','RSI_14D','PX_OPEN'],axis=1,inplace=True)
    # df.to_csv("month_fundamental_trimmed/"+file,index=False)

# for file in os.listdir('month_fundamental_trimmed'):
#     print(file)
#     df_fundamental=pd.read_csv("month_fundamental_trimmed/"+file)
#     df_fundamental.set_index('Date',inplace=True)
#     df_technical=pd.read_csv("daily_technical_indicators/"+file)
#     df_technical.set_index('Date',inplace=True)
#     # df=pd.merge(df_fundamental, df_technical, left_index=True, right_index=True)
#     df=df_fundamental.join(df_technical,how='left')
#     df.to_csv("month_fundamental_trimmed/"+file)
#
