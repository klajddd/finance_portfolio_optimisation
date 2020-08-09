# OPTIMISE A PORTFOLIO USING THE EfficientFrontier CLASS OF THE pypfopt PYTHON MODULE
# The real time prices of the portfolio are retrieved from Yahoo.

# Import libraries
from pandas_datareader import data as web
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

total_trading_days = 252
optimise_portfolio_by_USD_amount = 15000

plt.style.use('fivethirtyeight')

# Stock symbols
assets = ['FB', 'AMZN', 'AAPL', 'NFLX', 'GOOG']

# Portfolio weights (total weights = 1)
weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

# Porftolio starting and ending dates
stock_start_date = '2013-01-01'
today = datetime.today().strftime('%Y-%m-%d')

# Store the Adjusted Closed Price in a dataframe
adjusted_closed_price_df = pd.DataFrame()

# Retrieve the adjusted closed prices from Yahoo
for stock in assets:
    adjusted_closed_price_df[stock] = web.DataReader(
        stock, data_source='yahoo', start=stock_start_date, end=today)['Adj Close']

title = 'Portfolio Adjusted Close Price History'

# Plot the graph, set its title, labels and legend
for c in adjusted_closed_price_df.columns.values:
    plt.plot(adjusted_closed_price_df[c], label=c)

plt.title(title)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Adjusted Price (USD)', fontsize=18)
plt.legend(adjusted_closed_price_df.columns.values, loc='upper right')
plt.show()

# Calculate the daily simple return, (new stock price - old stock price) / old stock price
returns = adjusted_closed_price_df.pct_change()
print(returns)

# Annualized Covariance Matrix:
# Represents the directional relationship between 2 asset prices

# Variance:
# How much a set of observations differ from one-another (represented by the diagonal values of the matrix)
# Volatility = sqrt(variance)
cov_matrix_annual = returns.cov() * total_trading_days
print(cov_matrix_annual)


# Portfolio variance = weights transposed * covariance matrix * weights
port_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))
print(port_variance)

# Portfolio volatility ( = standard deviation)
port_volatility = np.sqrt(port_variance)
print(port_volatility)

# Snnual portfolio return
portfolioSimpleAnnualReturn = np.sum(
    returns.mean() * weights) * total_trading_days
print(portfolioSimpleAnnualReturn)

# Expected annual return, volatility (risk) / standard deviation, and variance
percent_var = str(round(port_variance, 2) * 100) + '%'
percent_vols = str(round(port_volatility, 2) * 100) + '%'
percent_ret = str(round(portfolioSimpleAnnualReturn, 2) * 100) + '%'

print('Expected annual return: ' + percent_ret)
print('Annual volatility / risk : ' + percent_vols)
print('Annual variance: ' + percent_var)


# Optimise the portfolio using EfficientFrontier
# mu = mean
mu = expected_returns.mean_historical_return(adjusted_closed_price_df)

# Sample covariance matrix
s = risk_models.sample_cov(adjusted_closed_price_df)

# Optimise for maximum Sharpe ratio:
# Measures the performance of an investment compared to a virtually risk free investment such as bonds or treasury bills
ef = EfficientFrontier(mu, s)

# Maximising the sharpe ratio and getting the raw weights
weights = ef.max_sharpe()

# Sets to zero any weight with absolute values below some cutoff value then rounds the rest of the values (introduces some rounding error)
# Enables removal of stocks that would not contribute into optimising the portfolio
cleaned_weights = ef.clean_weights()
print(cleaned_weights)
ef.portfolio_performance(verbose=True)

# Discrete allocation for each share of stock
latest_prices = get_latest_prices(adjusted_closed_price_df)
weights = cleaned_weights

# 15,000 amount of money we are putting into the portfolio to optimise it
da = DiscreteAllocation(weights, latest_prices,
                        total_portfolio_value=optimise_portfolio_by_USD_amount)
allocation, leftover = da.lp_portfolio()
print('Discrete Allocation: ', allocation)
print('Funds remaining: ${:.2f}'.format(leftover))
