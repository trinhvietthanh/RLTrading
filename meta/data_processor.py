from __future__ import annotations

import numpy as np
import pandas as pd

from meta.data_processors.yahoo_finance import (YahooFinanceProcessor as YahooFinance)

class DataProcessor:
    def __init__(self, data_source, **kwargs) -> None:
        if data_source == "yahoofinance":
            self.processor = YahooFinance()
        else:
            raise NotImplementedError("Data source input is NOT supported yet.")
        
    
    def download_data(
        self, ticker_list, start_date, end_date, time_interval
    ) -> pd.DataFrame:
        df = self.processor.download_data(
            ticker_list=ticker_list,
            start_date=start_date,
            end_date=end_date,
            time_interval=time_interval
        )
        
        return df
    
    def clean_data(self, df) -> pd.DataFrame:
        df = self.processor.clean_data(df) 
        
        return df

    def add_technical_indicator(self, df, tech_indicator_list) -> pd.DataFrame:
        
        self.tech_indicator_list = tech_indicator_list
        df = self.processor.add_technical_indicator(df, tech_indicator_list)

        return df

    def add_vix(self, df) -> pd.DataFrame:
        df = self.processor.add_vix(df)

        return df

    def add_turbulence(self, df) -> pd.DataFrame:
        df = self.processor.add_vix(df)
        
        return df
    
    def df_to_array(self, df, if_vix) -> np.array:
        price_array, tech_array, turbulence_array = self.processor.df_to_array(
            df, self.tech_indicator_list, if_vix
        )
        tech_nan_positions = np.isnan(tech_array)
        tech_array[tech_nan_positions] = 0
        tech_inf_positions = np.isinf(tech_array)
        tech_array[tech_inf_positions] = 0
        
        return price_array, tech_array, turbulence_array