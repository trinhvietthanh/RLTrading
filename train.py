import os

import config
import config_tickers
from meta.env_trade.portfolio.env_portfolio import StockPortfolioEnv
from meta.preprocessor.vndirectdownloader import VNDirectDownloader

if not os.path.exists("./" + config.DATA_SAVE_DIR):
    os.makedirs("./" + config.DATA_SAVE_DIR)
if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
    os.makedirs("./" + config.TRAINED_MODEL_DIR)
if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
    os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
if not os.path.exists("./" + config.RESULTS_DIR):
    os.makedirs("./" + config.RESULTS_DIR)


from meta.preprocessor.yahoodownloader import YahooDownloader
from meta.preprocessor.preprocessors import (
    FeatureEngineer,
    data_split,
)
import pandas as pd

TRAIN_START_DATE = '2018-01-01'
TRAIN_END_DATE = '2022-12-01'
TEST_START_DATE = '2022-12-02'
TEST_END_DATE = '2023-04-20'

df = VNDirectDownloader(start_date = TRAIN_START_DATE,
                     end_date = TEST_END_DATE,
                     ticker_list = config.VN_30_TICKER).fetch_data()

fe = FeatureEngineer(
    use_technical_indicator=True,
    use_turbulence=False,
    user_defined_feature=False,
)

df = fe.preprocess_data(df)

# add covariance matrix as states
df = df.sort_values(["date", "tic"], ignore_index=True)
df.index = df.date.factorize()[0]

cov_list = []
return_list = []

# look back is one year
lookback = 252
for i in range(lookback, len(df.index.unique())):
    data_lookback = df.loc[i - lookback : i, :]
    price_lookback = data_lookback.pivot_table(
        index="date", columns="tic", values="close"
    )
    return_lookback = price_lookback.pct_change().dropna()
    return_list.append(return_lookback)

    covs = return_lookback.cov().values
    cov_list.append(covs)

df_cov = pd.DataFrame(
    {
        "date": df.date.unique()[lookback:],
        "cov_list": cov_list,
        "return_list": return_list,
    }
)
df = df.merge(df_cov, on="date")
df = df.sort_values(["date", "tic"]).reset_index(drop=True)

train = data_split(df, TRAIN_START_DATE, TRAIN_END_DATE)

import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


stock_dimension = len(train.tic.unique())
state_space = stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
tech_indicator_list = ["macd", "rsi_30", "cci_30", "dx_30"]
feature_dimension = len(tech_indicator_list)
print(f"Feature Dimension: {feature_dimension}")

env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "transaction_cost_pct": 0,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": tech_indicator_list,
    "action_space": stock_dimension,
    "reward_scaling": 1e-1,
}

e_train_gym = StockPortfolioEnv(df=train, **env_kwargs)

env_train, _ = e_train_gym.get_sb_env()
print(type(env_train))

from agents.stablebaselines3.models import DRLAgent

agent = DRLAgent(env=env_train)

# A2C_PARAMS = {"n_steps": 10, "ent_coef": 0.005, "learning_rate": 0.0004}
# model_a2c = agent.get_model(model_name="a2c", model_kwargs=A2C_PARAMS)

# trained_a2c = agent.train_model(
#     model=model_a2c, tb_log_name="a2c", total_timesteps=40000
# )
# trained_a2c.save('trained_models/trained_a2c_vn.zip')
trained_a2c = agent.load_model("ppo", "trained_models/trained_a2c_vn.zip")

agent = DRLAgent(env=env_train)
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.005,
    "learning_rate": 0.001,
    "batch_size": 128,
}
# model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS)

# trained_ppo = agent.train_model(
#     model=model_ppo, tb_log_name="ppo", total_timesteps=40000
# )

# trained_ppo.save('trained_models/trained_ppo_vn.zip')

trained_ppo = agent.load_model("ppo", "trained_models/trained_ppo_vn.zip")
trade = data_split(df, TEST_START_DATE, TEST_END_DATE)
e_trade_gym = StockPortfolioEnv(df=trade, **env_kwargs)


import torch

import plotly.express as px


from pypfopt.efficient_frontier import EfficientFrontier
import pandas as pd
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import objective_functions

unique_tic = trade.tic.unique()
unique_trade_date = trade.date.unique()

import pyfolio

from plot import (
    backtest_stats,
    backtest_plot,
    get_daily_return,
    get_baseline,
    convert_daily_return_to_pyfolio_ts,
)

baseline_df = get_baseline(ticker="^DJI", start="2020-07-01", end="2021-09-01")

baseline_df_stats = backtest_stats(baseline_df, value_col_name="close")
baseline_returns = get_daily_return(baseline_df, value_col_name="close")

dji_cumpod = (baseline_returns + 1).cumprod() - 1

from pyfolio import timeseries

df_daily_return_a2c, df_actions_a2c = DRLAgent.DRL_prediction(
    model=trained_a2c, environment=e_trade_gym
)
df_daily_return_ppo, df_actions_ppo = DRLAgent.DRL_prediction(
    model=trained_ppo, environment=e_trade_gym
)

df_actions_ppo.to_csv("df_action_ppo.csv")
df_actions_a2c.to_csv("df_action_a2c.csv")
df_daily_return_a2c.to_csv("daily_return_a2c.csv")
df_daily_return_ppo.to_csv("daily_return_ppo.csv")

time_ind = pd.Series(df_daily_return_a2c.date)
a2c_cumpod = (df_daily_return_a2c.daily_return + 1).cumprod() - 1
ppo_cumpod = (df_daily_return_ppo.daily_return + 1).cumprod() - 1
DRL_strat_a2c = convert_daily_return_to_pyfolio_ts(df_daily_return_a2c)
DRL_strat_ppo = convert_daily_return_to_pyfolio_ts(df_daily_return_ppo)

perf_func = timeseries.perf_stats
perf_stats_all_a2c = perf_func(
    returns=DRL_strat_a2c,
    factor_returns=DRL_strat_a2c,
    positions=None,
    transactions=None,
    turnover_denom="AGB",
)
perf_stats_all_ppo = perf_func(
    returns=DRL_strat_ppo,
    factor_returns=DRL_strat_ppo,
    positions=None,
    transactions=None,
    turnover_denom="AGB",
)


def extract_weights(drl_actions_list):
    a2c_weight_df = {"date": [], "weights": []}
    for i in range(len(drl_actions_list)):
        date = drl_actions_list.index[i]
        tic_list = list(drl_actions_list.columns)
        weights_list = (
            drl_actions_list.reset_index()[list(drl_actions_list.columns)]
            .iloc[i]
            .values
        )
        weight_dict = {"tic": [], "weight": []}
        for j in range(len(tic_list)):
            weight_dict["tic"] += [tic_list[j]]
            weight_dict["weight"] += [weights_list[j]]

        a2c_weight_df["date"] += [date]
        a2c_weight_df["weights"] += [pd.DataFrame(weight_dict)]

    a2c_weights = pd.DataFrame(a2c_weight_df)
    return a2c_weights


a2c_weights = extract_weights(df_actions_a2c)
ppo_weights = extract_weights(df_actions_ppo)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor


def prepare_data(trainData):
    train_date = sorted(set(trainData.date.values))
    X = []
    for i in range(0, len(train_date) - 1):
        d = train_date[i]
        d_next = train_date[i + 1]
        y = (
            train.loc[train["date"] == d_next]
            .return_list.iloc[0]
            .loc[d_next]
            .reset_index()
        )
        y.columns = ["tic", "return"]
        x = train.loc[train["date"] == d][["tic", "macd", "rsi_30", "cci_30", "dx_30"]]
        train_piece = pd.merge(x, y, on="tic")
        train_piece["date"] = [d] * len(train_piece)
        X += [train_piece]
    trainDataML = pd.concat(X)
    X = trainDataML[tech_indicator_list].values
    Y = trainDataML[["return"]].values

    return X, Y


train_X, train_Y = prepare_data(train)
rf_model = RandomForestRegressor(
    max_depth=35, min_samples_split=10, random_state=0
).fit(train_X, train_Y.reshape(-1))
dt_model = DecisionTreeRegressor(
    random_state=0, max_depth=35, min_samples_split=10
).fit(train_X, train_Y.reshape(-1))
svm_model = SVR(epsilon=0.14).fit(train_X, train_Y.reshape(-1))
lr_model = LinearRegression().fit(train_X, train_Y)


def output_predict(model, reference_model=False):
    meta_coefficient = {"date": [], "weights": []}

    portfolio = pd.DataFrame(index=range(1), columns=unique_trade_date)
    initial_capital = 1000000
    portfolio.loc[0, unique_trade_date[0]] = initial_capital

    for i in range(len(unique_trade_date) - 1):

        current_date = unique_trade_date[i]
        next_date = unique_trade_date[i + 1]
        df_current = df[df.date == current_date].reset_index(drop=True)
        tics = df_current["tic"].values
        features = df_current[tech_indicator_list].values
        df_next = df[df.date == next_date].reset_index(drop=True)
        if not reference_model:
            predicted_y = model.predict(features)
            mu = predicted_y
            Sigma = risk_models.sample_cov(df_current.return_list[0], returns_data=True)
        else:
            mu = df_next.return_list[0].loc[next_date].values
            Sigma = risk_models.sample_cov(df_next.return_list[0], returns_data=True)
        predicted_y_df = pd.DataFrame(
            {
                "tic": tics.reshape(
                    -1,
                ),
                "predicted_y": mu.reshape(
                    -1,
                ),
            }
        )
        min_weight, max_weight = 0, 1
        ef = EfficientFrontier(mu, Sigma)
        weights = ef.nonconvex_objective(
            objective_functions.sharpe_ratio,
            objective_args=(ef.expected_returns, ef.cov_matrix),
            weights_sum_to_one=True,
            constraints=[
                {
                    "type": "ineq",
                    "fun": lambda w: w - min_weight,
                },  # greater than min_weight
                {
                    "type": "ineq",
                    "fun": lambda w: max_weight - w,
                },  # less than max_weight
            ],
        )

        weight_df = {"tic": [], "weight": []}
        meta_coefficient["date"] += [current_date]
        # it = 0
        for item in weights:
            weight_df["tic"] += [item]
            weight_df["weight"] += [weights[item]]

        weight_df = pd.DataFrame(weight_df).merge(predicted_y_df, on=["tic"])
        meta_coefficient["weights"] += [weight_df]
        cap = portfolio.iloc[0, i]
        # current cash invested for each stock
        current_cash = [element * cap for element in list(weights.values())]
        # current held shares
        current_shares = list(np.array(current_cash) / np.array(df_current.close))
        # next time period price
        next_price = np.array(df_next.close)
        portfolio.iloc[0, i + 1] = np.dot(current_shares, next_price)

    portfolio = portfolio.T
    portfolio.columns = ["account_value"]
    portfolio = portfolio.reset_index()
    portfolio.columns = ["date", "account_value"]
    stats = backtest_stats(portfolio, value_col_name="account_value")
    portfolio_cumprod = (portfolio.account_value.pct_change() + 1).cumprod() - 1

    return portfolio, stats, portfolio_cumprod, pd.DataFrame(meta_coefficient)


lr_portfolio, lr_stats, lr_cumprod, lr_weights = output_predict(lr_model)
dt_portfolio, dt_stats, dt_cumprod, dt_weights = output_predict(dt_model)
svm_portfolio, svm_stats, svm_cumprod, svm_weights = output_predict(svm_model)
rf_portfolio, rf_stats, rf_cumprod, rf_weights = output_predict(rf_model)
(
    reference_portfolio,
    reference_stats,
    reference_cumprod,
    reference_weights,
) = output_predict(None, True)


print("LR PORTFOLIO: ", lr_stats)

def calculate_gradient(
    model, interpolated_input, actions, feature_idx, stock_idx, h=1e-1
):
    forward_input = interpolated_input
    forward_input[feature_idx + stock_dimension][stock_idx] += h
    forward_Q = model.policy.evaluate_actions(
        torch.FloatTensor(forward_input).reshape(
            -1, stock_dimension * (stock_dimension + feature_dimension)
        ),
        torch.FloatTensor(actions).reshape(-1, stock_dimension),
    )
    interpolated_Q = model.policy.evaluate_actions(
        torch.FloatTensor(interpolated_input).reshape(
            -1, stock_dimension * (stock_dimension + feature_dimension)
        ),
        torch.FloatTensor(actions).reshape(-1, stock_dimension),
    )
    forward_Q = forward_Q[0].detach().cpu().numpy()[0]
    interpolated_Q = interpolated_Q[0].detach().cpu().numpy()[0]
    return (forward_Q - interpolated_Q) / h


import copy

meta_Q = {"date": [], "feature": [], "Saliency Map": [], "algo": []}

for algo in {"A2C", "PPO"}:
    if algo == "A2C":
        prec_step = 1e-2
    else:
        prec_step = 1e-1

    model = eval("trained_" + algo.lower())
    df_actions = eval("df_actions_" + algo.lower())
    
    for i in range(0, len(unique_trade_date) - 1, 2):
        date = unique_trade_date[i]
        covs = trade[trade["date"] == date].cov_list.iloc[0]
        features = trade[trade["date"] == date][tech_indicator_list].values  # N x K
        actions = df_actions.loc[date].values

        for feature_idx in range(len(tech_indicator_list)):

            int_grad_per_feature = 0
            for stock_idx in range(features.shape[0]):  # N

                int_grad_per_stock = 0
                avg_interpolated_grad = 0
                for alpha in range(1, 51):
                    scale = 1 / 50
                    baseline_features = copy.deepcopy(features)
                    baseline_noise = np.random.normal(0, 1, stock_dimension)
                    baseline_features[:, feature_idx] = [0] * stock_dimension
                    interpolated_features = baseline_features + scale * alpha * (
                        features - baseline_features
                    )  # N x K
                    interpolated_input = np.append(
                        covs, interpolated_features.T, axis=0
                    )
                    interpolated_gradient = calculate_gradient(
                        model,
                        interpolated_input,
                        actions,
                        feature_idx,
                        stock_idx,
                        h=prec_step,
                    )[0]

                    avg_interpolated_grad += interpolated_gradient * scale
                int_grad_per_stock = (
                    features[stock_idx][feature_idx] - 0
                ) * avg_interpolated_grad
                int_grad_per_feature += int_grad_per_stock

            meta_Q["date"] += [date]
            meta_Q["algo"] += [algo]
            meta_Q["feature"] += [tech_indicator_list[feature_idx]]
            meta_Q["Saliency Map"] += [int_grad_per_feature]

meta_Q = pd.DataFrame(meta_Q)

import statsmodels.api as sm

meta_score_coef = {"date": [], "coef": [], "algo": []}

for algo in ["LR", "RF", "Reference Model", "SVM", "DT", "A2C", "PPO"]:
    if algo == "LR":
        weights = lr_weights
    elif algo == "RF":
        weights = rf_weights
    elif algo == "DT":
        weights = dt_weights
    elif algo == "SVM":
        weights = svm_weights
    elif algo == "A2C":
        weights = a2c_weights
    elif algo == "PPO":
        weights = ppo_weights
    else:
        weights = reference_weights

    for i in range(0, len(unique_trade_date) - 1, 2):
        date = unique_trade_date[i]
        next_date = unique_trade_date[i + 1]
        df_temp = df[df.date == date].reset_index(drop=True)
        df_temp_next = df[df.date == next_date].reset_index(drop=True)
        weight_piece = weights[weights.date == date].iloc[0]["weights"]
        piece_return = pd.DataFrame(
            df_temp_next.return_list.iloc[0].loc[next_date]
        ).reset_index()
        piece_return.columns = ["tic", "return"]
        X = df_temp[["macd", "rsi_30", "cci_30", "dx_30", "tic"]]
        X_next = df_temp_next[["macd", "rsi_30", "cci_30", "dx_30", "tic"]]
        piece = weight_piece.merge(X, on="tic").merge(piece_return, on="tic")
        piece["Y"] = piece["return"] * piece["weight"]
        X = piece[["macd", "rsi_30", "cci_30", "dx_30"]]
        X = sm.add_constant(X)
        Y = piece[["Y"]]
        model = sm.OLS(Y, X)
        results = model.fit()
        meta_score_coef["coef"] += [(X * results.params).sum(axis=0)]
        meta_score_coef["date"] += [date]
        meta_score_coef["algo"] += [algo]

meta_score_coef = pd.DataFrame(meta_score_coef)

performance_score = {"date": [], "algo": [], "score": []}

for i in range(0, len(unique_trade_date)):
    date_ = unique_trade_date[i]
    if len(meta_score_coef[(meta_score_coef["date"] == date_)]) == 0:
        continue
    lr_coef = (
        meta_score_coef[
            (meta_score_coef["date"] == date_) & (meta_score_coef["algo"] == "LR")
        ]["coef"]
        .values[0][["macd", "rsi_30", "cci_30", "dx_30"]]
        .values
    )
    rf_coef = (
        meta_score_coef[
            (meta_score_coef["date"] == date_) & (meta_score_coef["algo"] == "RF")
        ]["coef"]
        .values[0][["macd", "rsi_30", "cci_30", "dx_30"]]
        .values
    )
    reference_coef = (
        meta_score_coef[
            (meta_score_coef["date"] == date_)
            & (meta_score_coef["algo"] == "Reference Model")
        ]["coef"]
        .values[0][["macd", "rsi_30", "cci_30", "dx_30"]]
        .values
    )
    dt_coef = (
        meta_score_coef[
            (meta_score_coef["date"] == date_) & (meta_score_coef["algo"] == "DT")
        ]["coef"]
        .values[0][["macd", "rsi_30", "cci_30", "dx_30"]]
        .values
    )
    svm_coef = (
        meta_score_coef[
            (meta_score_coef["date"] == date_) & (meta_score_coef["algo"] == "SVM")
        ]["coef"]
        .values[0][["macd", "rsi_30", "cci_30", "dx_30"]]
        .values
    )

    saliency_coef_a2c = meta_Q[(meta_Q["date"] == date_) & (meta_Q["algo"] == "A2C")][
        "Saliency Map"
    ].values
    saliency_coef_ppo = meta_Q[(meta_Q["date"] == date_) & (meta_Q["algo"] == "PPO")][
        "Saliency Map"
    ].values

    lr_score = np.corrcoef(lr_coef, reference_coef)[0][1]
    rf_score = np.corrcoef(rf_coef, reference_coef)[0][1]
    dt_score = np.corrcoef(dt_coef, reference_coef)[0][1]
    svm_score = np.corrcoef(svm_coef, reference_coef)[0][1]
    saliency_score_a2c = np.corrcoef(saliency_coef_a2c, reference_coef)[0][1]
    saliency_score_ppo = np.corrcoef(saliency_coef_ppo, reference_coef)[0][1]

    for algo in ["LR", "A2C", "PPO", "RF", "DT", "SVM"]:
        performance_score["date"] += [date_]
        performance_score["algo"] += [algo]
        if algo == "LR":
            score = lr_score
        elif algo == "RF":
            score = rf_score
        elif algo == "DT":
            score = dt_score
        elif algo == "A2C":
            score = saliency_score_a2c
        elif algo == "SVM":
            score = svm_score
        else:
            score = saliency_score_ppo
        performance_score["score"] += [score]

performance_score = pd.DataFrame(performance_score)

multi_performance_score = {"date": [], "algo": [], "score": []}
window = 20
for i in range(0, len(unique_trade_date) - window):
    date_ = unique_trade_date[i]
    if len(meta_score_coef[(meta_score_coef["date"] == date_)]) == 0:
        continue
    lr_coef = (
        meta_score_coef[
            (meta_score_coef["date"] == date_) & (meta_score_coef["algo"] == "LR")
        ]["coef"]
        .values[0][["macd", "rsi_30", "cci_30", "dx_30"]]
        .values
    )
    rf_coef = (
        meta_score_coef[
            (meta_score_coef["date"] == date_) & (meta_score_coef["algo"] == "RF")
        ]["coef"]
        .values[0][["macd", "rsi_30", "cci_30", "dx_30"]]
        .values
    )
    reference_coef = (
        meta_score_coef[
            (meta_score_coef["date"] == date_)
            & (meta_score_coef["algo"] == "Reference Model")
        ]["coef"]
        .values[0][["macd", "rsi_30", "cci_30", "dx_30"]]
        .values
    )
    for w in range(1, window):
        date_f = unique_trade_date[i + w]
        prx_coef = (
            meta_score_coef[
                (meta_score_coef["date"] == date_f)
                & (meta_score_coef["algo"] == "Reference Model")
            ]["coef"]
            .values[0][["macd", "rsi_30", "cci_30", "dx_30"]]
            .values
        )
        reference_coef += prx_coef
    reference_coef = reference_coef / window
    dt_coef = (
        meta_score_coef[
            (meta_score_coef["date"] == date_) & (meta_score_coef["algo"] == "DT")
        ]["coef"]
        .values[0][["macd", "rsi_30", "cci_30", "dx_30"]]
        .values
    )
    svm_coef = (
        meta_score_coef[
            (meta_score_coef["date"] == date_) & (meta_score_coef["algo"] == "SVM")
        ]["coef"]
        .values[0][["macd", "rsi_30", "cci_30", "dx_30"]]
        .values
    )
    saliency_coef_a2c = meta_Q[(meta_Q["date"] == date_) & (meta_Q["algo"] == "A2C")][
        "Saliency Map"
    ].values
    saliency_coef_ppo = meta_Q[(meta_Q["date"] == date_) & (meta_Q["algo"] == "PPO")][
        "Saliency Map"
    ].values
    lr_score = np.corrcoef(lr_coef, reference_coef)[0][1]
    rf_score = np.corrcoef(rf_coef, reference_coef)[0][1]
    dt_score = np.corrcoef(dt_coef, reference_coef)[0][1]
    svm_score = np.corrcoef(svm_coef, reference_coef)[0][1]
    saliency_score_a2c = np.corrcoef(saliency_coef_a2c, reference_coef)[0][1]
    saliency_score_ppo = np.corrcoef(saliency_coef_ppo, reference_coef)[0][1]

    for algo in ["LR", "A2C", "RF", "PPO", "DT", "SVM"]:
        multi_performance_score["date"] += [date_]
        multi_performance_score["algo"] += [algo]
        if algo == "LR":
            score = lr_score
        elif algo == "RF":
            score = rf_score
        elif algo == "DT":
            score = dt_score
        elif algo == "A2C":
            score = saliency_score_a2c
        elif algo == "SVM":
            score = svm_score
        else:
            score = saliency_score_ppo
        multi_performance_score["score"] += [score]

multi_performance_score = pd.DataFrame(multi_performance_score)

from datetime import datetime as dt

import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go

trace1_portfolio = go.Scatter(x=time_ind, y=a2c_cumpod, mode="lines", name="A2C")
trace2_portfolio = go.Scatter(x=time_ind, y=ppo_cumpod, mode="lines", name="PPO")
trace3_portfolio = go.Scatter(x=time_ind, y=dji_cumpod, mode="lines", name="DJIA")
trace4_portfolio = go.Scatter(x=time_ind, y=lr_cumprod, mode="lines", name="LR")
trace5_portfolio = go.Scatter(x=time_ind, y=rf_cumprod, mode="lines", name="RF")
trace6_portfolio = go.Scatter(x=time_ind, y=dt_cumprod, mode="lines", name="DT")
trace7_portfolio = go.Scatter(x=time_ind, y=svm_cumprod, mode="lines", name="SVM")

fig = go.Figure()
fig.add_trace(trace1_portfolio)
fig.add_trace(trace2_portfolio)

fig.add_trace(trace3_portfolio)

fig.add_trace(trace4_portfolio)
fig.add_trace(trace5_portfolio)
fig.add_trace(trace6_portfolio)
fig.add_trace(trace7_portfolio)

fig.update_layout(
    legend=dict(
        x=0,
        y=1,
        traceorder="normal",
        font=dict(family="sans-serif", size=15, color="black"),
        bgcolor="White",
        bordercolor="white",
        borderwidth=2,
    ),
)
fig.update_layout(
    title={
        # 'text': "Cumulative Return using FinRL",
        "y": 0.85,
        "x": 0.5,
        "xanchor": "center",
        "yanchor": "top",
    }
)

fig.update_layout(
    paper_bgcolor="rgba(1,1,0,0)",
    plot_bgcolor="rgba(1, 1, 0, 0)",
    xaxis_title="Date",
    yaxis=dict(titlefont=dict(size=30), title="Cumulative Return"),
    font=dict(
        size=40,
    ),
)
fig.update_layout(font_size=20)
fig.update_traces(line=dict(width=2))

fig.update_xaxes(
    showline=True,
    linecolor="black",
    showgrid=True,
    gridwidth=1,
    gridcolor="LightSteelBlue",
    mirror=True,
)
fig.update_yaxes(
    showline=True,
    linecolor="black",
    showgrid=True,
    gridwidth=1,
    gridcolor="LightSteelBlue",
    mirror=True,
)
fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="LightSteelBlue")

fig.show()

meta_score = {
    "Annual return": [],
    "Annual volatility": [],
    "Max drawdown": [],
    "Sharpe ratio": [],
    "Algorithm": [],
    "Calmar ratio": [],
}
for name in ["LR", "A2C", "RF", "Reference Model", "PPO", "SVM", "DT", "DJI"]:
    if name == "DT":
        annualreturn = dt_stats["Annual return"]
        annualvol = dt_stats["Annual volatility"]
        sharpeRatio = dt_stats["Sharpe ratio"]
        maxdradown = dt_stats["Max drawdown"]
        calmarratio = dt_stats["Calmar ratio"]
    elif name == "LR":
        annualreturn = lr_stats["Annual return"]
        annualvol = lr_stats["Annual volatility"]
        sharpeRatio = lr_stats["Sharpe ratio"]
        maxdradown = lr_stats["Max drawdown"]
        calmarratio = lr_stats["Calmar ratio"]
    elif name == "SVM":
        annualreturn = svm_stats["Annual return"]
        annualvol = svm_stats["Annual volatility"]
        sharpeRatio = svm_stats["Sharpe ratio"]
        maxdradown = svm_stats["Max drawdown"]
        calmarratio = svm_stats["Calmar ratio"]
    elif name == "RF":
        annualreturn = rf_stats["Annual return"]
        annualvol = rf_stats["Annual volatility"]
        sharpeRatio = rf_stats["Sharpe ratio"]
        maxdradown = rf_stats["Max drawdown"]
        calmarratio = rf_stats["Calmar ratio"]
    elif name == "Reference Model":
        annualreturn = reference_stats["Annual return"]
        annualvol = reference_stats["Annual volatility"]
        sharpeRatio = reference_stats["Sharpe ratio"]
        maxdradown = reference_stats["Max drawdown"]
        calmarratio = reference_stats["Calmar ratio"]
    elif name == "PPO":
        annualreturn = perf_stats_all_ppo["Annual return"]
        annualvol = perf_stats_all_ppo["Annual volatility"]
        sharpeRatio = perf_stats_all_ppo["Sharpe ratio"]
        maxdradown = perf_stats_all_ppo["Max drawdown"]
        calmarratio = perf_stats_all_ppo["Calmar ratio"]
    elif name == "DJI":
        annualreturn = baseline_df_stats["Annual return"]
        annualvol = baseline_df_stats["Annual volatility"]
        sharpeRatio = baseline_df_stats["Sharpe ratio"]
        maxdradown = baseline_df_stats["Max drawdown"]
        calmarratio = baseline_df_stats["Calmar ratio"]
    else:
        annualreturn = perf_stats_all_a2c["Annual return"]
        annualvol = perf_stats_all_a2c["Annual volatility"]
        sharpeRatio = perf_stats_all_a2c["Sharpe ratio"]
        maxdradown = perf_stats_all_a2c["Max drawdown"]
        calmarratio = perf_stats_all_a2c["Calmar ratio"]
    meta_score["Algorithm"] += [name]
    meta_score["Annual return"] += [annualreturn]
    meta_score["Annual volatility"] += [annualvol]
    meta_score["Max drawdown"] += [maxdradown]
    meta_score["Sharpe ratio"] += [sharpeRatio]
    meta_score["Calmar ratio"] += [calmarratio]

meta_score = pd.DataFrame(meta_score).sort_values("Sharpe ratio")


postiveRatio = pd.DataFrame(
    performance_score.groupby("algo").apply(lambda x: np.mean(x["score"]))
)

postiveRatio = postiveRatio.reset_index()
postiveRatio.columns = ["algo", "avg_correlation_coefficient"]
postiveRatio["Sharpe Ratio"] = [0] * 6

# postiveRatio.plot.bar(x = 'algo', y = 'avg_correlation_coefficient')

postiveRatiom = pd.DataFrame(
    multi_performance_score.groupby("algo").apply(lambda x: np.mean(x["score"]))
)
postiveRatiom = postiveRatiom.reset_index()
postiveRatiom.columns = ["algo", "avg_correlation_coefficient"]
postiveRatiom["Sharpe Ratio"] = [0] * 6

# postiveRatiom.plot.bar(x = 'algo', y = 'avg_correlation_coefficient')


for algo in ["A2C", "PPO", "LR", "DT", "RF", "SVM"]:
    postiveRatio.loc[postiveRatio["algo"] == algo, "Sharpe Ratio"] = meta_score.loc[
        meta_score["Algorithm"] == algo, "Sharpe ratio"
    ].values[0]
    postiveRatiom.loc[postiveRatio["algo"] == algo, "Sharpe Ratio"] = meta_score.loc[
        meta_score["Algorithm"] == algo, "Sharpe ratio"
    ].values[0]

postiveRatio.sort_values("Sharpe Ratio", inplace=True)

postiveRatiom.sort_values("Sharpe Ratio", inplace=True)

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Scatter(
        x=postiveRatiom["algo"],
        y=postiveRatiom["Sharpe Ratio"],
        name="Sharpe Ratio",
        marker_size=15,
        line_width=5,
    ),
    secondary_y=True,
)

fig.add_trace(
    go.Bar(
        x=postiveRatiom["algo"],
        y=postiveRatiom["avg_correlation_coefficient"],
        name="Multi-Step Average Correlation Coefficient          ",
        width=0.38,
    ),
    secondary_y=False,
)
fig.add_trace(
    go.Bar(
        x=postiveRatio["algo"],
        y=postiveRatio["avg_correlation_coefficient"],
        name="Single-Step Average Correlation Coefficient           ",
        width=0.38,
    ),
    secondary_y=False,
)

fig.update_layout(
    paper_bgcolor="rgba(1,1,0,0)",
    plot_bgcolor="rgba(1, 1, 0, 0)",
)
fig.update_layout(legend=dict(yanchor="top", y=1.5, xanchor="right", x=0.95))
fig.update_layout(font_size=15)

# Set x-axis title
fig.update_xaxes(title_text="Model")
fig.update_xaxes(
    showline=True,
    linecolor="black",
    showgrid=True,
    gridwidth=1,
    gridcolor="LightSteelBlue",
    mirror=True,
)
fig.update_yaxes(
    showline=True,
    linecolor="black",
    showgrid=True,
    secondary_y=False,
    gridwidth=1,
    gridcolor="LightSteelBlue",
    mirror=True,
)
fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="LightSteelBlue")
# Set y-axes titles
fig.update_yaxes(
    title_text="Average Correlation Coefficient",
    secondary_y=False,
    range=[-0.1, 0.1],
)
fig.update_yaxes(title_text="Sharpe Ratio", secondary_y=True, range=[-0.5, 2.5])

fig.show()

import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(rows=2, cols=3)

trace0 = go.Histogram(
    x=performance_score[performance_score["algo"] == "A2C"]["score"].values,
    nbinsx=25,
    name="A2C",
    histnorm="probability",
)
trace1 = go.Histogram(
    x=performance_score[performance_score["algo"] == "PPO"]["score"].values,
    nbinsx=25,
    name="PPO",
    histnorm="probability",
)
trace2 = go.Histogram(
    x=performance_score[performance_score["algo"] == "DT"]["score"].values,
    nbinsx=25,
    name="DT",
    histnorm="probability",
)
trace3 = go.Histogram(
    x=performance_score[performance_score["algo"] == "LR"]["score"].values,
    nbinsx=25,
    name="LR",
    histnorm="probability",
)
trace4 = go.Histogram(
    x=performance_score[performance_score["algo"] == "SVM"]["score"].values,
    nbinsx=25,
    name="SVM",
    histnorm="probability",
)
trace5 = go.Histogram(
    x=performance_score[performance_score["algo"] == "RF"]["score"].values,
    nbinsx=25,
    name="RF",
    histnorm="probability",
)

fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig.append_trace(trace2, 1, 3)
fig.append_trace(trace3, 2, 1)
fig.append_trace(trace4, 2, 2)
fig.append_trace(trace5, 2, 3)
# Update xaxis properties
fig.update_xaxes(title_text="Correlation coefficient", row=2, col=2)
fig.update_yaxes(title_text="Frequency", row=1, col=1)
fig.update_yaxes(title_text="Frequency", row=2, col=1)

fig.update_layout(
    paper_bgcolor="rgba(1,1,0,0)",
    plot_bgcolor="rgba(1, 1, 0, 0)",
    font=dict(
        size=18,
    ),
)
fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=1))

fig.update_xaxes(
    showline=True,
    linecolor="black",
    showgrid=True,
    gridwidth=1,
    gridcolor="LightSteelBlue",
    mirror=True,
)
fig.update_yaxes(
    showline=True,
    linecolor="black",
    showgrid=True,
    gridwidth=1,
    gridcolor="LightSteelBlue",
    mirror=True,
)
fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="LightSteelBlue")
fig.savefig("squares.png")

fig.show()

import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(rows=2, cols=3)

trace0 = go.Histogram(
    x=multi_performance_score[multi_performance_score["algo"] == "A2C"]["score"].values,
    nbinsx=25,
    name="A2C",
    histnorm="probability",
)
trace1 = go.Histogram(
    x=multi_performance_score[multi_performance_score["algo"] == "PPO"]["score"].values,
    nbinsx=25,
    name="PPO",
    histnorm="probability",
)
trace2 = go.Histogram(
    x=multi_performance_score[multi_performance_score["algo"] == "DT"]["score"].values,
    nbinsx=25,
    name="DT",
    histnorm="probability",
)
trace3 = go.Histogram(
    x=multi_performance_score[multi_performance_score["algo"] == "LR"]["score"].values,
    nbinsx=25,
    name="LR",
    histnorm="probability",
)
trace4 = go.Histogram(
    x=multi_performance_score[multi_performance_score["algo"] == "SVM"]["score"].values,
    nbinsx=25,
    name="SVM",
    histnorm="probability",
)
trace5 = go.Histogram(
    x=multi_performance_score[multi_performance_score["algo"] == "RF"]["score"].values,
    nbinsx=25,
    name="RF",
    histnorm="probability",
)

fig.update_layout(yaxis1=dict(range=[0, 0.2]))
fig.update_layout(yaxis2=dict(range=[0, 0.2]))
fig.update_layout(yaxis3=dict(range=[0, 0.4]))
fig.update_layout(yaxis4=dict(range=[0, 0.4]))
fig.update_layout(yaxis5=dict(range=[0, 0.4]))
fig.update_layout(yaxis6=dict(range=[0, 0.4]))

fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig.append_trace(trace2, 1, 3)
fig.append_trace(trace3, 2, 1)
fig.append_trace(trace4, 2, 2)
fig.append_trace(trace5, 2, 3)
# Update xaxis properties
fig.update_xaxes(title_text="Correlation coefficient", row=2, col=2)
fig.update_yaxes(title_text="Frequency", row=1, col=1)
fig.update_yaxes(title_text="Frequency", row=2, col=1)

fig.update_layout(
    paper_bgcolor="rgba(1,1,0,0)",
    plot_bgcolor="rgba(1, 1, 0, 0)",
    font=dict(
        size=18,
    ),
)
fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=1))
fig.update_xaxes(
    showline=True,
    linecolor="black",
    showgrid=True,
    gridwidth=1,
    gridcolor="LightSteelBlue",
    mirror=True,
)
fig.update_yaxes(
    showline=True,
    linecolor="black",
    showgrid=True,
    gridwidth=1,
    gridcolor="LightSteelBlue",
    mirror=True,
)
fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="LightSteelBlue")
fig.savefig("re.png")
fig.show()

pass