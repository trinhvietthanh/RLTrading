from vnstock import stock_historical_data

def get_chart_day(symbol):
    data_day = stock_historical_data(symbol, "2023-01-02", "2023-05-04")
    
    return data_day.to_json(orient='records')