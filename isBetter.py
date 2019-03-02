import pickle
import numpy

def probability(func, n_stocks, df_dict, stocknames, iter, month,factor_list,total_stocks):
    df = df_dict[stocknames[0]]
    # print('df_tail',df.tail())
    # print('df_shape',df.shape)
    date_60 = df.index.values[0:((iter) * 12 + 12 + month)]
    # print('dates_60',date_60)
    # print('date_len',date_60.size)
    portfolio_weights = dict()
    portfolio_number_of_stocks = dict()
    portfolio_cash = dict()
    index_number_of_stocks = dict()
    equal_weight= 1/n_stocks
    cash = 100000
    IndexValue = list()
    Value = list()  # stores the fund value of each month
    '''assign 0.0 weights to all stocks in the portfolio'''
    for stocks in stocknames:
        portfolio_weights[stocks] = 0.0
        portfolio_number_of_stocks[stocks] = 0.0
        portfolio_cash[stocks] = 0.0
        index_number_of_stocks[stocks] = 0.0
    counter = 0
    with open('df_dict_oc.pkl', 'rb') as f:
        df_dict_oc = pickle.load(f)
    '''start the 60 month iteration for that portfolio'''
    for z, date in enumerate(date_60):
        counter += 1
        factor_model_value = list()
        '''start reading each stock'''
        for files in stocknames:
            df1 = df_dict[files]
            factor_value_array = df1.values[z]
            factor_dict=dict()
            for r,factor in enumerate(factor_list):
                factor_dict[factor]=factor_value_array[r]
            # caculate factor model values for this stock
            answer = func(**factor_dict)
            factor_model_value.append(answer)  # contains fmv for all stocks

        # rank and sort all stocks
        list_values, list_stocks = zip(*sorted(zip(factor_model_value, stocknames), reverse=True))
        array = numpy.array(factor_model_value)
        temp = array.argsort()[::-1]
        ranks_fmv = numpy.empty(len(array), int)
        ranks_fmv[temp] = numpy.arange(len(array))

        # start trading
        top_fractile = list()
        bottom_fractile = list()
        current_month = dict()
        month_end = dict()

        # portfolio cash for each stock
        for j in stocknames:
            df_oc = df_dict_oc[j]
            df_oc_2 = df_oc.values
            current_month[j] = df_oc_2[z]
            month_end[j] = df_oc_2[z+1]
        # print(df_oc.index.values[z])
        # print(df_dict_oc['Britania'].head())
        fund_value = 0
        index_value=0
        index_cash=100000
        fund_value = fund_value + cash
        for i in stocknames:
            index_number_of_stocks[i]=index_cash/current_month[i]

        for i in stocknames:
            fund_value = fund_value + (current_month[i] * portfolio_number_of_stocks[i])
            index_value = index_value + (current_month[i])*index_number_of_stocks[i]
            portfolio_cash[i] = portfolio_number_of_stocks[i] * current_month[i]  # update cash portfolio
        fund_value_start = fund_value

        # store fund value globally
        c_max = fund_value * 0.03
        # update weights
        for i in stocknames:
            portfolio_weights[i] = portfolio_cash[i] / fund_value
        # start selling
        for i in range(total_stocks-1,n_stocks -1 , -1):
            bottom_fractile.append(list_stocks[i])
        # sell the stocks in bottom fractile
        for i in bottom_fractile:
            if portfolio_weights[i] != 0.0:
                cash = cash + (current_month[i] * portfolio_number_of_stocks[i])  # add cash back to cash reserve
                portfolio_number_of_stocks[i] = 0.0
                portfolio_weights[i] = 0.0
                portfolio_cash[i] = 0.0
        # sell stocks weights more than 0 if not in top quartile to bring it till 0
        #rebalancing
        for j,i in enumerate(list_stocks[:n_stocks]):
                if portfolio_weights[list_stocks[j]] > equal_weight:
                    # print("stock number 1")
                    extra_weight = portfolio_weights[i] - equal_weight
                    extra_cash = extra_weight * fund_value
                    cash = cash + (fund_value * extra_weight)
                    portfolio_cash[i] = portfolio_cash[i] - extra_cash
                    portfolio_weights[i] = portfolio_cash[i] / fund_value
                    portfolio_number_of_stocks[i] = portfolio_number_of_stocks[i] - (extra_cash / current_month[i])

        '''start buying'''
        # count the number of stocks in portfolio
        Sn = 0
        count1 = 0
        for i in stocknames:
            if portfolio_weights[i] != 0.0:
                Sn += 1
        for i in range( n_stocks):
            top_fractile.append(list_stocks[i])
            if portfolio_weights[list_stocks[i]] > 0.0:
                count1 = count1 + 1
        # buy stocks in top fractile which are not in the portfolio at all
        n = 0
        for i in top_fractile:
            if Sn < n_stocks and cash > c_max:
                if portfolio_weights[i] == 0.0:
                    cash_ratio = (cash - c_max) / (n_stocks - Sn)
                    if cash_ratio > (equal_weight* fund_value):
                        cash_ratio = (equal_weight * fund_value)
                    portfolio_cash[i] = cash_ratio
                    portfolio_weights[i] = cash_ratio / fund_value
                    portfolio_number_of_stocks[i] = round(cash_ratio / current_month[i])
                    cash -= cash_ratio
                    Sn += 1
        # if some fund is left ,buy the rest of stocks starting from the most attractive stock till 0.25
        for i in top_fractile:
            if portfolio_weights[i] < equal_weight and cash > c_max:
                missing_cash = fund_value * (equal_weight- portfolio_weights[i])
                if (missing_cash > cash):
                    missing_cash = cash
                    portfolio_cash[i] += missing_cash
                    portfolio_weights[i] += missing_cash / fund_value
                    portfolio_number_of_stocks[i] += missing_cash / current_month[i]
                    cash -= missing_cash
        fund_value_end = 0
        fund_value_end = fund_value_end + cash
        index_value_end=0
        portfolio_cash_end = dict()
        portfolio_cash_start = dict()
        portfolio_returns = dict()

        for i in stocknames:
            fund_value_end = fund_value_end + (month_end[i] * portfolio_number_of_stocks[i])
            index_value_end=index_value_end+month_end[i]*index_number_of_stocks[i]
            portfolio_cash_end[i] = month_end[i] * portfolio_number_of_stocks[i]
            portfolio_cash_start[i] = current_month[i] * portfolio_number_of_stocks[i]
        returns_list = []
        for i in stocknames:
            portfolio_returns[i] = (month_end[i] / current_month[i]) - 1
            returns_list.append(portfolio_returns[i])
        month_return = (fund_value_end / fund_value_start) - 1
        index_return = (index_value_end / index_value) - 1
        IndexValue.append(index_return)

        Value.append(month_return)

    prod = 1
    Return_array = numpy.array(Value)
    RETURN = []
    for ret in Return_array:
        RETURN.append(1 + ret)
    for RET in RETURN:
        prod = prod * RET
    Avg_Return = prod - 1
    Avg_Risk = Return_array.std() * 3.46
    denominator = len(date_60)
    numerator=0
    for i in range(len(Value)):
        if (Value[i] > IndexValue[i]):
            numerator += 1
    probability = numerator / denominator
    return probability