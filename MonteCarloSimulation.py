import numpy as np
import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
from scipy.stats import norm

plt.style.use('ggplot')

ticker = 'CIB'

data = pd.DataFrame()
data[ticker] = yf.download(ticker, start='2014-01-01')['Adj Close']

log_returns = np.log(1+data.pct_change())
u = log_returns.mean()
var = log_returns.var()
drift = u - (0.5*var)
stdev = log_returns.std()

days = 100
trials = 1000

Z = norm.ppf(np.random.rand(days,trials))
dailys_returns = np.exp(drift.values+stdev.values*Z)
prices_path = np.zeros_like(dailys_returns)
prices_path[0] = data.iloc[-1]

for t in range(1,days):
    prices_path[t] = prices_path[t-1]*dailys_returns[t]
    
mean_prices = np.mean(prices_path, axis=1)
df_mean_prices = pd.DataFrame(mean_prices, columns=['Mean Price'])
df_mean_prices.index.name = 'Day'
df_mean_prices.reset_index(inplace=True)

#print(df_mean_prices.head())
#np.savetxt('prices_road.csv',prices_path[t])
#np.savetxt('prices_road_all.csv',prices_path)

plt.figure(figsize=(15,6))
plt.plot(pd.DataFrame(prices_path))
plt.xlabel('Number Days')
plt.ylabel('Price of '+ticker)

sns.displot(pd.DataFrame(prices_path).iloc[-1])
plt.xlabel('Price to '+str(days)+' days')
plt.ylabel('Frecuency')

plt.figure(figsize=(10, 6))
plt.plot(df_mean_prices['Day'], df_mean_prices['Mean Price'], label='Mean Prices Path')
plt.title('Simulated Average Price Trajectories')
plt.xlabel('Days')
plt.ylabel('Mean Price')
plt.legend()
plt.show()

