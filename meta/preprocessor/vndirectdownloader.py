from __future__ import annotations

import pandas as pd
from tqdm import tqdm
import requests
import logging
from utils import utils
from bs4 import BeautifulSoup
from datetime import datetime
import config

class VNDirectDownloader:

    def __init__(self, start_date: str, end_date: str, ticker_list: list):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list
        
    def fetch_data(self, proxy=None) -> pd.DataFrame:
        data_df = pd.DataFrame()
        num_failures = 0
        for tic in tqdm(self.ticker_list, total=len(self.ticker_list)):
            temp_df = self.download_one_new(tic)
            temp_df["tic"] = tic
            
            if len(temp_df) > 0:
                data_df = data_df.append(temp_df)
            
            else:
                num_failures += 1
            
        if num_failures == len(self.ticker_list):
            raise ValueError("no data is fetched.")
        
        data_df = data_df.reset_index()
        try:
            # convert the columns nams to standardized names
            data_df.columns = [
                "date",
                "open",
                "high",
                "low",
                "close",
                # "adjcp",
                "volume",
                "tic"
            ]
        except NotImplementedError:
            print("Features are not supported!!!")
            
        # create day of the week column (monday=0)
        data_df["date"] =  pd.to_datetime(data_df["date"], format='%Y-%m-%d')
        data_df["day"] = data_df["date"].dt.day_of_week
        # convert date to standard string format, easy to filter
        data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
        
        # drop missing data
        data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)
        data_df = data_df.sort_values(by=["date", "tic"]).reset_index(drop=True)
        
        return data_df


    def select_equal_rows_stock(self, df):
        df_check = df.tic.value_counts()
        df_check = pd.DataFrame(df_check).reset_index()
        df_check.columns = ["tic", "counts"]
        mean_df = df_check.count.mean()
        equal_list = list(df.tic.value_counts() >= mean_df)
        
        names = df.tic.value_counts().index
        select_stocks_list = list(names[equal_list])
        df = df[df.tic.isin(select_stocks_list)]
        return df

    def download(self):

        stock_datas = []
        
        if not isinstance(self.ticker_list, list):
            symbols = [self.symbols]
        else:
            symbols = self.symbols
        
        for symbol in symbols:
            stock_datas.append(self.download_one_new(symbol))
        
        data = pd.concat(stock_datas, axis=1)
        return data

    def download_one_new(self, symbol):
        # start_date = utils.convert_text_dateformat(self.start_date, origin_type = '%d/%m/%Y', new_type = '%Y-%m-%d')
        # end_date = utils.convert_text_dateformat(self.end_date, origin_type = '%d/%m/%Y', new_type = '%Y-%m-%d')
        start_date = self.start_date
        end_date = self.end_date
        API_VNDIRECT = 'https://finfo-api.vndirect.com.vn/v4/stock_prices/'
        query = 'code:' + symbol + '~date:gte:' + start_date + '~date:lte:' + end_date
        delta = datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')
        params = {
            "sort": "date",
            "size": delta.days + 1,
            "page": 1,
            "q": query
        }
        res = requests.get(API_VNDIRECT, params=params, headers=config.HEADERS)
        data = res.json()['data']

        data = pd.DataFrame(data)
        stock_data = data[['date', 'adClose', 'close', 'pctChange', 'average', 'nmVolume',
                        'nmValue', 'ptVolume', 'ptValue', 'open', 'high', 'low']].copy()
        stock_data.columns = ['date', 'adjust', 'close', 'change_perc', 'avg',
                        'volume_match', 'value_match', 'volume_reconcile', 'value_reconcile',
                        'open', 'high', 'low']
        stock_data = stock_data.set_index('date').apply(pd.to_numeric, errors='coerce')
        stock_data.date = list(map(utils.convert_date, stock_data.index))
        stock_data.index.name = 'date'
        stock_data = stock_data.sort_index()
        stock_data.fillna(0, inplace=True)

        stock_data['volume'] = stock_data.volume_match + stock_data.volume_reconcile
        stock_data['tic'] = symbol
        stock_data.drop(["adjust",'change_perc', 'avg',
                        'volume_match', 'value_match', 'volume_reconcile', 'value_reconcile'], inplace=True, axis=1)
        logging.info('data {} from {} to {} have already cloned!' \
                        .format(symbol,
                                self.start_date, self.end_date))

        return stock_data

    def download_one(self, symbol):
            stock_data = pd.DataFrame(columns=['date', 'change_perc1', 'change_perc2',
                                            'open', 'high', 'low', 'close',
                                            'avg', 'volume_match', 'volume_reconcile'])
            last_page = self.get_last_page(symbol)
            # logging.info('Last page {}'.format(last_page))
            for i in range(last_page):
                stock_slice_batch = self.download_batch(i+1, symbol)
                stock_data = pd.concat([stock_data, stock_slice_batch], axis=0)
            stock_data = stock_data.set_index('date').apply(pd.to_numeric, errors='coerce')
            stock_data.index = list(map(utils.convert_date, stock_data.index))
            stock_data.index.name = 'date'
            stock_data = stock_data.sort_index()
            stock_data.fillna(0, inplace=True)
            stock_data['volume'] = int(stock_data.volume_match + stock_data.volume_reconcile)
            stock_data.dropna()
            # Create multiple columns
            iterables = [stock_data.columns.tolist(), [symbol]]
            mulindex = pd.MultiIndex.from_product(iterables, names=['Attributes', 'Symbols'])
            stock_data.columns = mulindex

            logging.info('data {} from {} to {} have already cloned!' \
                        .format(symbol,
                                utils.convert_text_dateformat(self.start, origin_type = '%d/%m/%Y', new_type = '%Y-%m-%d'),
                                utils.convert_text_dateformat(self.end, origin_type='%d/%m/%Y', new_type='%Y-%m-%d')))

            return stock_data

    def download_batch(self, id_batch, symbol):
        form_data = {"model.downloadType": "",
                        "pagingInfo.indexPage": str(id_batch),
                        "searchMarketStatisticsView.symbol": symbol,
                        "strFromDate": self.start,
                        "strToDate": self.end}
        r = requests.post(config.URL_VND, form_data, headers=config.HEADERS, verify=False)
        soup = BeautifulSoup(r.content, 'html.parser')
        data_node = soup.find(class_='list_tktt lichsugia')

        dates = []
        change_percents1 = []
        change_percents2 = []
        opens = []
        highs = []
        lows = []
        closes = []
        avgs = []
        adjusts = []
        volume_matchs = []
        volume_reconciles = []
        # logging.info(data_node)
        for i, value in enumerate(data_node.select('div')):
            if i < 10: continue
            value = utils.clean_text(value.text)
            if i % 10 == 0:
                dates.append(value)
            elif i % 10 == 1:
                values = value.split()
                change_percents1.append(values[0])
                change_percents2.append(values[1])
            elif i % 10 == 2:
                opens.append(value)
            elif i % 10 == 3:
                highs.append(value)
            elif i % 10 == 4:
                lows.append(value)
            elif i % 10 == 5:
                closes.append(value)
            elif i % 10 == 6:
                avgs.append(value)
            elif i % 10 == 7:
                adjusts.append(value)
            elif i % 10 == 8:
                volume_matchs.append(value)
            elif i % 10 == 9:
                volume_reconciles.append(value)

        stock_slice_batch = pd.DataFrame(
            {'date': dates, 'change_perc1': change_percents1, 'change_perc2': change_percents2,
                'open': opens, 'high': highs, 'low': lows, 'close': closes,
                'avg': avgs, 'volume_match': volume_matchs, 'volume_reconcile': volume_reconciles})

        return stock_slice_batch

    def get_last_page(self, symbol):
        form_data = {"searchMarketStatisticsView.symbol":symbol,
                    "strFromDate":self.start,
                    "strToDate":self.end}

        r = requests.post(config.URL_VND, form_data, headers=config.HEADERS, verify=False)
        soup = BeautifulSoup(r.content, 'html.parser')
        # last_page = utils.extract_number(str(soup.find_all('div', {'class': 'paging'})[-1].select('a')[-1].attrs))
        text_div = soup.find_all('div', {'class': 'paging'})[-1].get_text()
        try:
            last_page = int(text_div.split()[1].split('/')[1])
        except:
            last_page = int(text_div)
        return last_page


