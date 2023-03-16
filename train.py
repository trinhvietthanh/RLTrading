from __future__ import annotations

from config import ERL_PARAMS
from config import INDICATORS
from config import RLlib_PARAMS
from config import SAC_PARAMS, PPO_PARAMS
from config import TRAIN_END_DATE
from config import TRAIN_START_DATE
from config  import VN_30_TICKER, DOW_30_TICKER
from meta.data_processor import DataProcessor
from meta.env_trade.portfolio import StockPortfolioEnv

import pandas as pd
# construct environment


def train(
    start_date,
    end_date,
    ticker_list,
    data_source,
    time_interval,
    technical_indicator_list,
    drl_lib,
    env,
    model_name,
    if_vix=True,
    **kwargs,
):
    # download data
    dp = DataProcessor(data_source, **kwargs)
    # data = dp.download_data(ticker_list, start_date, end_date, time_interval)
    # data = dp.clean_data(data)
    # data = dp.add_technical_indicator(data, technical_indicator_list)
    # if if_vix:
    #     data = dp.add_vix(data)
    data = pd.read_csv("test.csv")

    # price_array, tech_array, turbulence_array = dp.df_to_array(data, if_vix)
    # env_config = {
    #     "price_array": price_array,
    #     "tech_array": tech_array,
    #     "turbulence_array": turbulence_array,
    #     "if_train": True,
    # }
    
    # add covariance matrix as states
    # data=data.sort_values(['timestamp','tic'],ignore_index=True)
    # data.index = data.timestamp.factorize()[0]

    cov_list = []
    return_list = []
    # data.to_csv("test.csv")
    # look back is one year
    lookback=252
    for i in range(lookback,len(data.index.unique())):
        data_lookback = data.loc[i-lookback:i,:]
        price_lookback=data_lookback.pivot_table(index = 'timestamp',columns = 'tic', values = 'close')
        return_lookback = price_lookback.pct_change().dropna()
        return_list.append(return_lookback)

        covs = return_lookback.cov().values 
        cov_list.append(covs)

    
    data_cov = pd.DataFrame({'timestamp':data.timestamp.unique()[lookback:],'cov_list':cov_list,'return_list':return_list})
    data = data.merge(data_cov, on='timestamp')
    data = data.sort_values(['timestamp','tic']).reset_index(drop=True)
    stock_dimension = len(ticker_list)
    state_space = stock_dimension
    
    print(">>>>>", data["cov_list"])
    
    env_kwargs = {
        "hmax": 100, 
        "initial_amount": 1000000, 
        "transaction_cost_pct": 0.001, 
        "state_space": state_space, 
        "stock_dim": stock_dimension, 
        "tech_indicator_list": technical_indicator_list, 
        "action_space": stock_dimension, 
        "reward_scaling": 1e-4
    }
    # env_instance = env(config=env_config)
    env_instance = env(data ,**env_kwargs)

    # read parameters
    cwd = kwargs.get("cwd", "./" + str(model_name))

    if drl_lib == "elegantrl":
        # from finrl.agents.elegantrl.models import DRLAgent as DRLAgent_erl

        # break_step = kwargs.get("break_step", 1e6)
        # erl_params = kwargs.get("erl_params")
        # agent = DRLAgent_erl(
        #     env=env,
        #     price_array=price_array,
        #     tech_array=tech_array,
        #     turbulence_array=turbulence_array,
        # )
        # model = agent.get_model(model_name, model_kwargs=erl_params)
        # trained_model = agent.train_model(
        #     model=model, cwd=cwd, total_timesteps=break_step
        # )
        pass
    elif drl_lib == "rllib":
        # total_episodes = kwargs.get("total_episodes", 100)
        # rllib_params = kwargs.get("rllib_params")
        # from finrl.agents.rllib.models import DRLAgent as DRLAgent_rllib

        # agent_rllib = DRLAgent_rllib(
        #     env=env,
        #     price_array=price_array,
        #     tech_array=tech_array,
        #     turbulence_array=turbulence_array,
        # )
        # model, model_config = agent_rllib.get_model(model_name)
        # model_config["lr"] = rllib_params["lr"]
        # model_config["train_batch_size"] = rllib_params["train_batch_size"]
        # model_config["gamma"] = rllib_params["gamma"]
        # # ray.shutdown()
        # trained_model = agent_rllib.train_model(
        #     model=model,
        #     model_name=model_name,
        #     model_config=model_config,
        #     total_episodes=total_episodes,
        # )
        # trained_model.save(cwd)
        pass
    elif drl_lib == "stable_baselines3":
        total_timesteps = kwargs.get("total_timesteps", 1e6)
        agent_params = kwargs.get("agent_params")
        from agents.stablebaselines3.models import DRLAgent as DRLAgent_sb3

        agent = DRLAgent_sb3(env=env_instance)
        model = agent.get_model(model_name, model_kwargs=agent_params)
        trained_model = agent.train_model(
            model=model, tb_log_name=model_name, total_timesteps=total_timesteps
        )
        print("Training is finished!")
        trained_model.save(cwd)
        print("Trained model is saved in " + str(cwd))
    else:
        raise ValueError("DRL library input is NOT supported. Please check.")


if __name__ == "__main__":

    env = StockPortfolioEnv

    # demo for elegantrl
    kwargs = (
        {}
    )  # in current meta, with respect yahoofinance, kwargs is {}. For other data sources, such as joinquant, kwargs is not empty
    # train(
    #     start_date=TRAIN_START_DATE,
    #     end_date=TRAIN_END_DATE,
    #     ticker_list=VN_30_TICKER,
    #     data_source="yahoofinance",
    #     time_interval="1D",
    #     technical_indicator_list=INDICATORS,
    #     drl_lib="elegantrl",
    #     env=env,
    #     model_name="ppo",
    #     cwd="./test_ppo",
    #     erl_params=ERL_PARAMS,
    #     break_step=1e5,
    #     kwargs=kwargs,
    # )

    ## if users want to use rllib, or stable-baselines3, users can remove the following comments

    # # demo for rllib
    # import ray
    # ray.shutdown()  # always shutdown previous session if any
    # train(
    #     start_date=TRAIN_START_DATE,
    #     end_date=TRAIN_END_DATE,
    #     ticker_list=DOW_30_TICKER,
    #     data_source="yahoofinance",
    #     time_interval="1D",
    #     technical_indicator_list=INDICATORS,
    #     drl_lib="rllib",
    #     env=env,
    #     model_name="ppo",
    #     cwd="./test_ppo",
    #     rllib_params=RLlib_PARAMS,
    #     total_episodes=30,
    # )
    #
    # # demo for stable-baselines3
    train(
        start_date=TRAIN_START_DATE,
        end_date=TRAIN_END_DATE,
        ticker_list=DOW_30_TICKER,
        data_source="yahoofinance",
        time_interval="1D",
        technical_indicator_list=INDICATORS,
        drl_lib="stable_baselines3",
        env=env,
        model_name="ppo",
        cwd="./test_ppo",
        agent_params=PPO_PARAMS,
        total_timesteps=1e4,
    )
