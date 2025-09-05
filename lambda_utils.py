import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import statsmodels.api as sm

class LambdaEstimation:
    def __init__(self,dataset):
        self.df = pd.read_csv(dataset)  # Previous months data to predict the current value of lambda
        self.data = pd.DataFrame()       # Preprocessed data 
        self.lam = 0

        self.model = LinearRegression()

        self.X = []
        self.Y = []

        self.r2Value = 0

        self.volume = 38846600
        self.volatility = 0.00929

    def preProcessing(self):
        self.df['ts_event'] = pd.to_datetime(self.df['ts_event'], utc=True) # time standard
        self.df = self.df.sort_values('ts_event').reset_index(drop=True) # date sort
        market_open_time = pd.to_datetime('13:30:00').time()  # Market opens at 13:30:00
        market_close_time = pd.to_datetime('20:00:00').time()  # Market closes at 19:59:59

        # Set 'ts_event' as the index
        self.df.set_index('ts_event', inplace=True)
        # Filter data between market open and close times
        df_market = self.df.between_time(market_open_time, market_close_time)
        df_minute = df_market.resample('1s').agg({
            'bid_fill': 'sum',
            'ask_fill': 'sum',
            'Signed Volume': 'sum',
            'best_bid': 'max',
            'best_ask': 'min',
        })
        df_minute = df_minute.reset_index()
        df_minute['mid_price'] = (df_minute['best_bid'] + df_minute['best_ask']) / 2

        self.data = df_minute
    

    def estimationModel(self):
        self.data['delta_p'] = self.data['mid_price'].diff()
        self.data['delta_Q'] = self.data['Signed Volume']
        self.data=self.data.dropna(subset=['delta_p', 'delta_Q'])
        self.X = self.data[['delta_Q']].values
        self.Y = self.data['delta_p'].values

        
        self.model.fit(self.X, self.Y) 
        self.lam = self.model.coef_[0]        # Assigning the coefficient of the model to self.lam
        self.r2Value = self.model.score(self.X, self.Y)


    def plot(self):
        plt.scatter(self.X, self.Y, label='Data Points')
        plt.plot(self.X, self.model.predict(self.X), color='red', label='Fitted Line')
        plt.xlabel('Delta Q')
        plt.ylabel('Delta P')
        plt.legend()
        plt.show()

    def printSummary(self):
        X = self.data['delta_Q']
        y = self.data['delta_p']

        # Fit the linear regression model using statsmodels
        model = sm.OLS(y, X).fit()

        # Print the summary of the regression, which includes t-tests and F-test
        print(model.summary())

    def getLam(self):
        self.preProcessing()
        self.estimationModel()
        return f"Lambda : {self.lam}"






        


