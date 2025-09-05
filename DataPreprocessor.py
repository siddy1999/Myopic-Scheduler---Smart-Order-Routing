import pandas as pd
import yfinance as yf
from datetime import timedelta
class DataPreprocessor :
    def __init__(self,dataFile):
        self.data = pd.read_csv(dataFile)
        self.processedData = pd.DataFrame()

    def integrateYFinance(self,ticker):
        """
        Input : Dataset, Ticker(str)
        Processing : Appends the relevant data from Yfinance and stores it in processedData
        Output : Updates self.processedData
        
        """
        df = self.data
        df['ts_event'] = pd.to_datetime(df['ts_event'], utc=True) # time standard
        df = df.sort_values('ts_event').reset_index(drop=True) # date sort
        df['Timestamp'] = pd.to_datetime(df['ts_event'])
        df['Date'] = df['Timestamp'].dt.date
        # Calculate the earliest and latest dates in 'df'
        earliest_date = df['Date'].min()
        latest_date = df['Date'].max()
        # Adjust 'start_date' and 'end_date' based on 'df'
        start_date = earliest_date - timedelta(days=30)  # Earliest date minus 30 days
        end_date = latest_date + timedelta(days=1)       # Latest date plus 1 day
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        # code to download daily volume and close price
        data = yf.download(ticker, start=start_date_str, end=end_date_str, interval='1d')
        data = data.reset_index()
        # Handle MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(level=1)
        data['Return'] = data['Adj Close'].pct_change()
        data['Volatility'] = data['Return'].rolling(window=5).std()  # 5-day rolling volatility
        data['ADV'] = data['Volume'].rolling(window=10).mean() # 10-day rolling average daily volume
        data = data.dropna()
        # Ensure 'Date' column in 'data' matches the format in 'df'
        data['Date'] = data['Date'].dt.date
        merged_df = pd.merge(df, data, on='Date', how='left')
        self.processedData = merged_df
        pass
    def CompletePreProcess(self,aggPeriod = "4s"):

        merged_df= self.processedData[['ts_event','bid_fill','ask_fill','Signed Volume','best_bid','best_ask','Volatility','ADV']]
        market_open_time = pd.to_datetime('13:30:00').time()  # Market opens at 13:30:00
        market_close_time = pd.to_datetime('20:00:00').time()  # Market closes at 19:59:59

        # Set 'ts_event' as the index
        merged_df.set_index('ts_event', inplace=True)
        # Filter data between market open and close times
        df_market = merged_df.between_time(market_open_time, market_close_time)

        # Aggregation
        df_minute = df_market.resample(aggPeriod).agg({
            'bid_fill': 'sum',
            'ask_fill': 'sum',
            'Signed Volume': 'sum',
            'best_bid': 'max',
            'best_ask': 'min',
            'Volatility': 'max',
            'ADV': 'max',
        })
        df_minute = df_minute.dropna(subset=['best_bid']) # downsample will fill NaNs to out-of-market hours
        df_minute = df_minute.reset_index()
        df_minute['mid_price'] = (df_minute['best_bid'] + df_minute['best_ask']) / 2
        self.processedData = df_minute
        pass
    def run(self,ticker,aggPeriod):
        self.integrateYFinance(ticker)
        self.CompletePreProcess(aggPeriod)
        print(f"Processed Data : {self.processedData}")
        return self.processedData




    