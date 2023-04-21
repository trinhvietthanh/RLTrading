import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from pypfopt import risk_models, objective_functions
from pypfopt import EfficientFrontier
from plot import (
    backtest_stats
)

import statsmodels.api as sm


tech_indicator_list = ["macd", "rsi_30", "cci_30", "dx_30"]

def prepare_data(trainData):
    train_date = sorted(set(trainData.data.value))
    X = []
    for i in range(0, len(train_date) - 1):
        d = train_date[i]
        d_next = train_date[i + 1]
        y = (
            trainData.loc[trainData["date"] == d_next].return_list.iloc[0].loc[d_next].reset_index()
        )
        y.columns = ["tic", "return"]
        x = trainData.loc[trainData['date'] == d]["tic", "macd", "rsi_30", "cci_30", "dx_30"]
        # Merge x and y by tic 
        train_price = pd.merge(x, y, on="tic")
        train_price["date"] = [d] * len(train_price)
        X += [train_price]
    trainDataML = pd.concat(X)
    X = trainDataML[tech_indicator_list].values
    Y = trainDataML[["return"]].values
    
    return X, Y

def output_predict(model, unique_trade_date, reference_model=False):
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



def list_method(train, unique_trade_date, initial_capital=1000000):
    train_X, train_Y = prepare_data(train)
    lr_model = LinearRegression().fit(train_X, train_Y)
    
    lr_portfolio, lr_stats, lr_cumprod, lr_weights = output_predict(lr_model)

    meta_score_coef = {"date": [], "coef": [], "algo": []}

    for algo in ["LR", "RF"]:
        if algo == "LR":
            weights = lr_weights
        
        for i in range(len(unique_trade_date) - 1):
            date = unique_trade_date[i]
            next_date = unique_trade_date[i + 1]
            df_temp = df[df.date == date].reset_index(drop=True)
            df_temp_next = df[df.date == next_date].reset_index(drop=True)
            weight_piece = weights[weights.date == date].iloc[0]["weights"]
            piece_return = pd.DateFrame(
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
        lr_score = np.corrcoef(lr_coef, reference_coef)[0][1]
        
        for algo in ["LR"]:
            performance_score['date'] += [date_]
            performance_score['algo'] += [algo]
            
            if algo == "LR":
                score = lr_score
            performance_score["score"] += [score]
    performance_score = pd.DataFrame(performance_score)
    
    multi_performance_score = {"date": [], "algo": [], "score": []}
    
    window = 20
    
