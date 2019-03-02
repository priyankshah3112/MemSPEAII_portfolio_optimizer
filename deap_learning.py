from deap import base, creator, gp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing
import pickle
from deap import algorithms
from deap.tools.emo import _randomizedSelect
import operator
import test
import numpy as np
import pandas as pd
from deap import tools
import os
import isBetter
Ratios=dict()
df1=pd.read_excel('FMCG//Britania//Britania_full.xlsx',sheetname=0)
df1 = df1[['Date', 'Price', 'Volume', 'MarketCap', 'PE', 'PB','ProfitMargin','ROA', 'DebtEquity','CurrentRatio', 'InventoryTurnover','DividendPayout','CrudePrice','GoldPrice','Inflation','Forex','GDP','MA','Volatility','RepoRate','FII']]
c=df1.columns.values
# print(c)
Cash=100000
path='FMCG'
stocknames=os.listdir(path)
Cmax=3
n_stocks=4
Cash_max=Cash*Cmax/100.0
df1.set_index('Date',inplace=True)

#print(df1.head())
# for date in df1.index:
#     l=[]
#     for col in df1.columns:
#         l.append(df1.get_value(date,col))
#     Ratios[date]=l
# dates=list(Ratios.keys())
# df_dict=dict()
# for files in stocknames:
#     df=pd.read_excel("FMCG//" + files + "//" + files + "_final.xlsx",sheetname=0)
#     df.set_index('Date',inplace=True)
#     df =df[['Price', 'Volume', 'MarketCap', 'PE', 'PB','ProfitMargin','ROA', 'DebtEquity','CurrentRatio', 'InventoryTurnover','DividendPayout','CrudePrice','GoldPrice','Inflation','Forex','GDP','MA','Volatility','RepoRate','FII']]
#     df_dict[files]=df


# with open('df_dict_norm.pkl', 'rb') as f:
#     df_dict=pickle.load(f)
def pow_2(a):
    return  (a*a)
def pow_3(a):
    return  (a*a*a)
def my_div(a,b):
    if b==0:
        return 1
    else:
        return  a/b
pset = gp.PrimitiveSet("MAIN", 20)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(pow_2, 1)
pset.addPrimitive(pow_3, 1)
pset.addPrimitive(my_div, 2)
pset.renameArguments(ARG0='Price',ARG1='Volume',ARG2='MarketCap',ARG3='PE',ARG4='PB', ARG5='ProfitMargin',ARG6='ROA',ARG7='DebtEquity',ARG8='CurrentRatio',ARG9='InventoryTurnover',ARG10='DividendPayout',ARG11='CrudePrice',ARG12='GoldPrice',ARG13='Inflation',ARG14='Forex',ARG15='GDP',ARG16='MA',ARG17='Volatility',ARG18='RepoRate',ARG19='FII')


creator.create("FitnessMulti", base.Fitness, weights=(1.0,-1.0))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)


toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("prob",isBetter.probability)
toolbox.register("trailing_prob",isBetter.trailing_probability)
def evalStockRanking(tup):
    individual,df_dict=tup
    func = toolbox.compile(expr=individual)
    with open('iter.pkl', 'rb') as f:
        iter = pickle.load(f)
    with open('month.pkl', 'rb') as f:
        month = pickle.load(f)
    # for i in stocknames:
    #     print(df_dict[i])
    avg_ret,avg_risk=test.fitness_trailing_normalized(func=func,Cash=Cash,C_max=Cash_max,N_stocks=4,df_dict=df_dict,stocknames=stocknames,iter=iter,month=month)
    plt.scatter(avg_ret,avg_risk)
    # print("Return"+str(avg_ret))
    # print("Risk"+str(avg_risk))
    return avg_ret,avg_risk
def evalProbability(tup):
    individual,df_dict,iter,month=tup
    func = toolbox.compile(expr=individual)
    # for i in stocknames:
    #     print(df_dict[i])
    probability=toolbox.trailing_prob(func=func,df_dict=df_dict,iter=iter,month=month)

    # print("Return"+str(avg_ret))
    # print("Risk"+str(avg_risk))
    return probability
# def evalStockTesting(individual):
#     func = toolbox.compile(expr=individual)
#     with open('iter.pkl', 'rb') as f:
#         iter = pickle.load(f)
#     with open('month.pkl', 'rb') as f:
#         month=pickle.load(f)
#     avg_ret=monthly_test.fitness_trailing_test(func=func,Cash=Cash,C_max=Cash_max,N_stocks=4,df_dict=df_dict,stocknames=stocknames,iter=iter,month=month)
#
#     # print("Return"+str(avg_ret))
#     # print("Risk"+str(avg_risk))
#     return avg_ret
def loop(tup):
    i,distances_cd,fits,N,K=tup
    j = i + 1
    temp = [0.0] * j
    dist = temp + list(distances_cd[i][j:])
    distances = [i ** 2 for i in dist]
    kth_dist = _randomizedSelect(distances, 0, N - 1, K)
    density = 1.0 / (kth_dist + 2.0)
    fits[i] += density
    return fits[i]
MAX_LIMIT=17
toolbox.register("evaluate_prob", evalProbability)
toolbox.register("evaluate", evalStockRanking)
toolbox.register("select", tools.selSPEA2)
toolbox.register("mate", gp.cxOnePoint)
toolbox.decorate("mate",gp.staticLimit(operator.attrgetter('height'),MAX_LIMIT))
toolbox.register("expr_mut", gp.genGrow, min_=2, max_=3)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.register("loop",loop)
toolbox.decorate("mutate",gp.staticLimit(operator.attrgetter('height'),MAX_LIMIT))


def main(pool):
    with open('df_dict_norm.pkl', 'rb') as f:
        df_dict=pickle.load(f)
    toolbox.register("map", pool.map)
    if os.path.isfile('checkpoint_name.pkl'):
        with open('checkpoint_name.pkl', "rb") as cp_file:
            cp = pickle.load(cp_file)
        pop = cp["population"]
        start_gen = cp["generation"]
        hof = cp["halloffame"]
        logbook = cp["logbook"]

    else:
        # Start a new evolution
        pop = toolbox.population(n=500)
        start_gen = 0
        hof = tools.HallOfFame(maxsize=5)
        logbook = tools.Logbook()
    with open('iter.pkl', 'rb') as f:
        iter_pk = pickle.load(f)
    with open('month.pkl', 'rb') as f:
        month= pickle.load(f)
    #randomize the search population and pickle with fitness
    rand_pop=toolbox.population(n=500)
    fitnesses_rand = toolbox.map(toolbox.evaluate, ((rand,df_dict) for rand in rand_pop))
    for ind, fit in zip(rand_pop, fitnesses_rand):
        ind.fitness.values = fit
    with open('search_pop.pkl', 'wb') as f:
        pickle.dump(rand_pop, f)
    #assign fitness values to last envs pop for new env
    for ind in pop:
        del ind.fitness.values
    fitnesses_pop = toolbox.map(toolbox.evaluate, ((ind,df_dict) for ind in pop))

    for ind, fit in zip(pop, fitnesses_pop):
        ind.fitness.values = fit
    with open('env_final_pop/' + str(iter_pk) + 'pop.pkl', 'wb') as f:
        pickle.dump(pop, f)
    #unpickle the memory
    if (os.path.isfile('memory.pkl')):
        with open('memory.pkl', 'rb') as f:
            memory=pickle.load(f)
        for mem in memory:
            del mem.fitness.values
        fitnesses_mem = toolbox.map(toolbox.evaluate,  ((mem,df_dict) for mem in memory))
        for mem, fit in zip(memory, fitnesses_mem):
            mem.fitness.values = fit

        with open('env_memory/' + str(iter_pk) + str(month)+'memory.pkl', 'wb') as f:
            pickle.dump(memory, f)
        # print(len(memory))
        # print(len(pop))
        pop=pop+memory

        # print("ADDED POP"+str(len(pop)))
    else:
        pop=pop
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    # print("LEN",len(pop))
    # for ind in pop:
    #     print(ind.fitness.values)
    pop,log=algorithms.eaMuPlusLambda(start_gen,pop, toolbox,500,500, 0.8, 0.1,start_gen+50, stats, halloffame=hof)
    pool.close()
    return pop, stats, hof


if __name__ == "__main__":

    if (os.path.isfile('iter.pkl')):
        with open('iter.pkl', 'rb') as f:
            iter_pk = pickle.load(f)
    else:
        iter_pk = 0
    # with open('df_dict.pkl', 'rb') as f:
    #     df_dict=pickle.load(f)
    for iter in range(6, 9):
        print(iter)
        month_pk=0
        df_dict_update=dict()
        df_dict_norm=dict()
        for month in range(0, 12):
            with open('df_dict_full.pkl', 'rb') as f:
                df_dict_full = pickle.load(f)
            with open('df_dict_done.pkl', 'rb') as f:
                df_dict_done = pickle.load(f)
            # new_date=((iter) * 12 + 12 + month)+1
            # df2 = df_dict['Britania']
            # date_60 = df2.index.values[:new_date]
            if not (iter==6 and month==0):
                print ("iter",iter)
                print("month",month)
                for files in stocknames:
                    print("==============="+files+"=================")
                    df2 = df_dict_full[files]
                    df3 = df_dict_done[files]
                    new_date = ((iter) * 12 + 12 + month)-1
                    date = df2.index.values[new_date]
                    print("New date",date)
                    df_new = df2.iloc[[new_date]]
                    df1 = pd.concat([df3, df_new])
                    df1 = df1[['Price', 'Volume', 'MarketCap', 'PE', 'PB', 'ProfitMargin', 'ROA', 'DebtEquity',
                               'CurrentRatio', 'InventoryTurnover', 'DividendPayout', 'CrudePrice', 'GoldPrice',
                               'Inflation',
                               'Forex', 'GDP', 'MA', 'Volatility', 'RepoRate', 'FII']]
                    df_dict_update[files]=df1
                    df_norm = (df1 - df1.min()) / (df1.max() - df1.min())
                    df_dict_norm[files]=df_norm
                with open('df_dict_done.pkl', 'wb') as f:
                    pickle.dump(df_dict_update,f)
                with open('df_dict_norm.pkl', 'wb') as f:
                    pickle.dump(df_dict_norm,f)
                print(df_dict_norm['PNG'].tail())
            print(month)
            with open('iter.pkl', 'wb') as f:
                pickle.dump(iter, f)
            with open('month.pkl', 'wb') as f:
                pickle.dump(month, f)
            pool = multiprocessing.Pool(processes=4)
            population, stats, hof = main(pool)
            with open("HOF//"+str(iter)+"_"+str(month)+"hof.pkl","wb") as f:
                pickle.dump(hof,f)
            with open("POP//"+str(iter)+"_"+str(month)+"pop.pkl","wb") as f:
                pickle.dump(population,f)
