from vnstock import *

def overview_stock(symbol):
    ticker = ticker_overview(symbol)
    print(">>>>", ticker)

    price_latest = stock_historical_data(symbol, "2023-05-22", "2023-05-23")
    data = {
      "name": ticker["shortName"][0],
      "establishedYear": ticker["establishedYear"][0],
      "website": ticker["website"][0],
      "exchange": ticker["exchange"][0],
      "open": price_latest["Open"][0],
      "high": price_latest["High"][0],
      "low": price_latest["Low"][0],
      "close": price_latest["Close"][0],
      "volume": price_latest["Volume"][0],
      "industry": ticker["industry"][0],
    }
    
    return data