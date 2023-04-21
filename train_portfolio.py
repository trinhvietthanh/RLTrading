import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import datetime

import config
import config_tickers
from meta.preprocessor.yahoodownloader import YahooDownloader
from meta.preprocessor.vndirectdownloader import VNDirectDownloader
from meta.preprocessor.preprocessors import FeatureEngineer, data_split
from meta.env_trade.portfolio.env_portfolio import StockPortfolioEnv
from agents.stablebaselines3.models import DRLAgent
from plot import backtest_stats, backtest_plot, get_daily_return, get_baseline,convert_daily_return_to_pyfolio_ts
from meta.data_processor import DataProcessor
from meta.data_processors.yahoo_finance import YahooFinanceProcessor
import sys
sys.path.append("../FinRL-Library")

#create dir
import os
if not os.path.exists("./" + config.DATA_SAVE_DIR):
    os.makedirs("./" + config.DATA_SAVE_DIR)
if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
    os.makedirs("./" + config.TRAINED_MODEL_DIR)
if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
    os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
if not os.path.exists("./" + config.RESULTS_DIR):
    os.makedirs("./" + config.RESULTS_DIR)
    
TRAIN_START_DATE = '2022-10-02'
TRAIN_END_DATE = '2022-11-01'
TEST_START_DATE = '2022-11-02'
TEST_END_DATE = '2023-03-10'

# df = VNDirectDownloader(start_date = TRAIN_START_DATE,
#                      end_date = TEST_END_DATE,
#                      ticker_list = config.FUEVFVND).fetch_data()
# # df.to_csv("datavip.csv", index=False)
# # df = pd.read_csv("datavip.csv")
# fe = FeatureEngineer(
#                     use_technical_indicator=True,
#                     use_turbulence=False,
#                     user_defined_feature = False)

# df = fe.preprocess_data(df)

# df.to_csv("data_fe.csv", index=False)

df = pd.read_csv("data_fe.csv")

# add covariance matrix as states
df=df.sort_values(['date','tic'],ignore_index=True)
df.index = df.date.factorize()[0]

cov_list = []
return_list = []

# look back is one year
lookback=252
for i in range(lookback,len(df.index.unique())):
  data_lookback = df.loc[i-lookback:i,:]
  price_lookback=data_lookback.pivot_table(index = 'date',columns = 'tic', values = 'close')
  return_lookback = price_lookback.pct_change().dropna()
  return_list.append(return_lookback)

  covs = return_lookback.cov().values 
  cov_list.append(covs)

  
df_cov = pd.DataFrame({'date':df.date.unique()[lookback:],'cov_list':cov_list,'return_list':return_list})
df = df.merge(df_cov, on='date')
df = df.sort_values(['date','tic']).reset_index(drop=True)


train = data_split(df, TRAIN_START_DATE, TRAIN_END_DATE)

stock_dimension = len(train.tic.unique()) 
state_space = stock_dimension 
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

init_state = [config.FUEVFVND_PER[key]/100 for key in config.FUEVFVND]
print("STATE: ", init_state)
env_kwargs = {
    "hmax": 100, 
    "initial_amount": 10000, 
    "transaction_cost_pct": 0.001, 
    "state_space": state_space, 
    "stock_dim": stock_dimension, 
    "tech_indicator_list": config.INDICATORS,
    "init_state":  init_state,
    "action_space": stock_dimension, 
    "reward_scaling": 1e-4
    
}

e_train_gym = StockPortfolioEnv(df = train, **env_kwargs)

env_train, _ = e_train_gym.get_sb_env()


# initialize
agent = DRLAgent(env = env_train)


# #ppo
agent = DRLAgent(env = env_train)
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 64,
}
model_ppo = agent.get_model("ppo",model_kwargs = PPO_PARAMS)

trained_ppo = agent.train_model(model=model_ppo, 
                             tb_log_name='ppo',
                             total_timesteps=80000)
# trained_ppo.save('trained_models/trained_ppo.zip')

# Trader
trade = data_split(df, TEST_START_DATE, TEST_END_DATE)

e_trade_gym = StockPortfolioEnv(df = trade, **env_kwargs)
# trained_ppo = agent.DRL_prediction_load_from_file("ppo", e_trade_gym, "trained_models/trained_ppo.zip")
df_daily_return, df_actions = DRLAgent.DRL_prediction(model=trained_ppo,
                        environment = e_trade_gym)

df_daily_return.to_csv('df_daily_return.csv')
df_actions.to_csv('df_actions.csv')


# #Backtest
# from pyfolio import timeseries
# DRL_strat = convert_daily_return_to_pyfolio_ts(df_daily_return)
# perf_func = timeseries.perf_stats 
# perf_stats_all = perf_func( returns=DRL_strat, 
#                               factor_returns=DRL_strat, 
#                                 positions=None, transactions=None, turnover_denom="AGB")

# print("==============DRL Strategy Stats===========")
# print(perf_stats_all)

# #baseline stats
# print("==============Get Baseline Stats===========")
# baseline_df = get_baseline(
#         ticker="^DJI", 
#         start = df_daily_return.loc[0,'date'],
#         end = df_daily_return.loc[len(df_daily_return)-1,'date'])

# stats = backtest_stats(baseline_df, value_col_name = 'close')

# import pyfolio
# baseline_df = get_baseline(
#         ticker='^DJI', start=df_daily_return.loc[0,'date'], end='2021-11-01'
#     )

# baseline_returns = get_daily_return(baseline_df, value_col_name="close")

# with pyfolio.plotting.plotting_context(font_scale=1.1):
#         pyfolio.create_full_tear_sheet(returns = DRL_strat,
#                                        benchmark_rets=baseline_returns, set_context=False)