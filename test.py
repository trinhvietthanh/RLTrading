from meta.preprocessor.preprocessors import data_split
import mlmethod
import pandas as pd
df = pd.read_csv("data_fe.csv")
TRAIN_START_DATE = '2018-01-01'
TRAIN_END_DATE = '2022-12-01'
tech_indicator_list = ["macd", "rsi_30", "cci_30", "dx_30"]

train = data_split(df, TRAIN_START_DATE, TRAIN_END_DATE)

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

print(train_X)