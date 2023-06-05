
from agents.stablebaselines3.models import DRLAgent
from meta.env_trade.portfolio.env_portfolio import StockPortfolioEnv
from meta.preprocessor.vndirectdownloader import VNDirectDownloader
from meta.preprocessor.preprocessors import FeatureEngineer, data_split
import pandas as pd
import config
import json

from plot import convert_daily_return_to_pyfolio_ts

def a2cEnv(stock_dimension, initail_amount):
    pass

def ppoEnv(train, initail_amount):
    pass

    # return env_train
methods = {
    "A2C": a2cEnv,
    "PPO": ppoEnv
}
    
def get_env(type):
    return methods[type]

def chose_methods(method):
    pass
  
def ppo_method():
    pass

def get_data_for_train(assets, start, end):
    df = VNDirectDownloader(start_date=start, end_date=end,ticker_list=assets).fetch_data()
    # print(df)

    fe = FeatureEngineer(
                        use_technical_indicator=True,
                        use_turbulence=False,
                        user_defined_feature = False)
    df = fe.preprocess_data(df)
    filename = start + "_" + end + ".csv"
    df.to_csv("data/"+ filename, index=False)
    return filename

def load_data_predict(filedata, assets, end_date):
    filedata = 'data_fe.csv'
    data = pd.read_csv("data/"+ filedata)
    last_date = data['date'][len(data['date'])-1]
    if end_date < last_date:
        return data
    else:
        df = VNDirectDownloader(start_date=last_date, end_date=end_date,ticker_list=assets).fetch_data()
        # print(df)

        fe = FeatureEngineer(
                            use_technical_indicator=True,
                            use_turbulence=False,
                            user_defined_feature = False)
        df = fe.preprocess_data(df)
        data.append(df, ignore_index=True)
        return data

def trainAI(filename, task, method="PPO", initial_amount=1000000):
    data = pd.read_csv("data/" +filename)
    start_date=data['date'][0];
    end_date=data['date'][len(data['date']) -1];
    data = data.sort_values(["date", "tic"], ignore_index=True)
    data.index = data.date.factorize()[0]

    cov_list = []
    return_list = []
    # look back is one year
    lookback = 252
    for i in range(lookback, len(data.index.unique())):
        data_lookback = data.loc[i - lookback: i, :]
        price_loockback = data_lookback.pivot_table(
            index="date", columns="tic", values="close"
        )
        
        return_lookback = price_loockback.pct_change().dropna()
        return_list.append(return_lookback)
        
        covs = return_lookback.cov().values
        cov_list.append(covs)
        
    df_cov = pd.DataFrame(
        {
            "date": data.date.unique()[lookback:],
            "cov_list": cov_list,
            "return_list": return_list
        }
    )
    data = data.merge(df_cov, on="date")
    data = data.sort_values(["date", "tic"]).reset_index(drop=True)
    train = data_split(data, start=start_date, end=end_date)
    
    stock_dimension = len(train.tic.unique()) 
    state_space = stock_dimension 
    env_kwargs = {
         "hmax": 100, 
        "initial_amount": initial_amount, 
        "transaction_cost_pct": 0.001, 
        "state_space": state_space, 
        "stock_dim": stock_dimension, 
        "tech_indicator_list": config.INDICATORS,
        "init_state":  None,
        "action_space": stock_dimension, 
        "reward_scaling": 1e-4,
        "T_plus": 1,
    }
    e_train_gym = StockPortfolioEnv(df = train, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    agent = DRLAgent(env=env_train)
    PPO_PARAMS = {
        "n_steps": 2048,
        "ent_coef": 0.01,
        "learning_rate": 0.00025,
        "batch_size": 64,
    }
    print(">>>>>>>TRAINED>>>>>>>>>", agent)

    model_ppo = agent.get_model("ppo",model_kwargs = PPO_PARAMS)
    print(">>>>>>>TRAINED>>>>>>>>>")

    trained_ppo = agent.train_model(model=model_ppo, 
                                tb_log_name='ppo',
                                total_timesteps=80000)
    print(">>>>>>>ENDTRAINED>>>>>>")
    filename = "trained_" + task + method + ".zip"
    trained_ppo.save('trained_models/'+ filename)
    import requests
    requests.post("http://locahost:3001/task", {"step": "done", "model_trained": filename})
    # trained_ppo = agent.load_model("ppo", "trained_models/trained_ppo.zip")
    return True

def predictData(filedata, assets, end_trader, start_trader, method="ppo"):
    # df = pd.read_csv("data/"+filedata)
    df = load_data_predict(filedata, assets, end_trader)
    TRAIN_START_DATE=df['date'][0];
    TRAIN_END_DATE=df['date'][len(df['date']) -1];
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

    init_state = [config.VN30_PER[key]/100 for key in config.VN30_PER]
    env_kwargs = {
        "hmax": 100, 
        "initial_amount": 10000, 
        "transaction_cost_pct": 0.001, 
        "state_space": state_space, 
        "stock_dim": stock_dimension, 
        "tech_indicator_list": config.INDICATORS,
        "init_state":  init_state,
        "action_space": stock_dimension, 
        "reward_scaling": 1e-4,
        "T_plus": 0,
    }
    e_train_gym = StockPortfolioEnv(df = train, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()


    # initialize
    agent = DRLAgent(env = env_train)
    if method == "a2c":
        trained_a2c = agent.load_model("a2c", "trained_models/trained_a2c.zip")
        trade = data_split(df, start_trader, end_trader)
        e_trade_gym = StockPortfolioEnv(df = trade, **env_kwargs)
        df_daily_return, df_actions = DRLAgent.DRL_prediction(model=trained_a2c,
                            environment = e_trade_gym)
    else:
        trained_ppo = agent.load_model("ppo", "trained_models/trained_ppo.zip")
        trade = data_split(df, start_trader, end_trader)
        e_trade_gym = StockPortfolioEnv(df = trade, **env_kwargs)
        df_daily_return, df_actions = DRLAgent.DRL_prediction(model=trained_ppo,
                            environment = e_trade_gym)
        
    df_daily_return.to_csv('results/1.csv')
    df_actions.to_csv('results/2.csv')
    result_value = df_daily_return.iloc[-1]
    last_action = df_actions.iloc[-1:]
    
    action = last_action.to_json(orient='records')
    from pyfolio import timeseries
    DRL_strat = convert_daily_return_to_pyfolio_ts(df_daily_return)
    perf_func = timeseries.perf_stats 
    perf_stats_all = perf_func( returns=DRL_strat, 
                                factor_returns=DRL_strat, 
                                    positions=None, transactions=None, turnover_denom="AGB")
    jsonStats = perf_stats_all.to_json(orient='index')
    return {
        "value": result_value['daily_return'],
        "action": json.loads(action)[0],
        "stats": json.loads(jsonStats),
        "startAmount": e_trade_gym.initial_amount,
        "endAmount": e_trade_gym.portfolio_value
    }