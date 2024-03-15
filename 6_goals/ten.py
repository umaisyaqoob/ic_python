import numpy as np
import sys
import pandas as pd
import requests
import matplotlib.pyplot as plt
from PyPortfolioOpt.pypfopt import BlackLittermanModel, risk_models, expected_returns


# Ab aap BlackLittermanModel, risk_models, expected_returns ka istemal kar sakte hain


print("OK",sys.path[0])


# # Configuration and API Key
# api_key = 'c9864e890304fb88b00e5227e8423b1d'
# tickers = [
#     "MMM", "DD", "MRK", "AA", "XOM", "MSFT", "AXP", "GE", "PFE", "T",
#     "BAC", "HD", "TRV", "BA", "INTC", "UNH", "CAT", "IBM", "CVX",
#     "JNJ", "VZ", "CSCO", "JPM", "WMT", "KO", "MCD", "DIS", "GLD"
# ]

# rebalance_period = 10  # Rebalance every 10 days
# lookback_period = 252  # Lookback period for historical returns, e.g., 252 trading days for 1 year

# # Define the date range for historical data
# start_date = (datetime.today() - timedelta(days=12 * 365)).strftime('%Y-%m-%d')  # 3 years of data
# end_date = datetime.today().strftime('%Y-%m-%d')

# # Fetch historical price data function
# def fetch_data(ticker):
#     url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?from={start_date}&to={end_date}&apikey={api_key}"
#     response = requests.get(url)
#     data = response.json()
    
#     if 'historical' not in data:
#         print(f"No historical data found for {ticker}")
#         return pd.Series()
    
#     df = pd.DataFrame(data['historical']).set_index('date')['close'].sort_index()
#     return df

# # Fetch and prepare price data
# price_data = pd.DataFrame({ticker: fetch_data(ticker) for ticker in tickers})

# # Calculate historical returns for the lookback period to set as views
# historical_returns = expected_returns.mean_historical_return(price_data[-lookback_period:], frequency=252)

# # Initialize DataFrame for portfolio weights and values
# portfolio_weights = pd.DataFrame(index=price_data.index, columns=tickers, data=0.0)
# portfolio_values_bl = pd.Series(index=price_data.index, dtype=float)

# # Set initial portfolio value
# initial_portfolio_value = 1000
# portfolio_values_bl.iloc[0] = initial_portfolio_value

# # Rebalancing and portfolio value computation
# for i in range(0, len(price_data) - rebalance_period, rebalance_period):
#     current_window = price_data.iloc[max(0, i - lookback_period):i + rebalance_period]
#     mu = expected_returns.mean_historical_return(current_window, frequency=252)
#     S = risk_models.CovarianceShrinkage(current_window).ledoit_wolf()

#     # Create views and link matrix
#     Q = historical_returns.values.reshape(-1, 1)  # Views (Q) as numpy array
#     P = np.eye(len(Q))  # Identity matrix for P since each view applies to one asset

#     # Assume a uniform confidence level of 75% for all views
#     confidence_levels = np.full(len(Q), 0.75)

# # Pass the confidence levels to the Black-Litterman model
#     bl = BlackLittermanModel(S, pi=mu, Q=Q, P=P, omega="idzorek", view_confidences=confidence_levels)

#     # Calculate Black-Litterman weights and normalize
#    # Normalize weights and check constraints
#     bl_weights = pd.Series(bl.bl_weights(), index=tickers)
#     if any(bl_weights < 0):
#         bl_weights[bl_weights < 0] = 0  # Set any negative weight to 0
#     bl_weights /= bl_weights.sum()  # Re-normalize after adjusting for negatives

# # Ensure the sum of weights is 1
#     assert np.isclose(bl_weights.sum(), 1), "Weights do not sum to 1"
# # Normalize weights

#     # Assign weights to the portfolio
#     portfolio_weights.iloc[i:i + rebalance_period] = bl_weights.T

#     # Calculate portfolio values
#     for j in range(i, min(i + rebalance_period, len(price_data))):
#         if j > 0:
#             returns_bl = np.dot(portfolio_weights.iloc[j - 1], price_data.pct_change(fill_method=None).iloc[j])
#             portfolio_values_bl.iloc[j] = portfolio_values_bl.iloc[j - 1] * (1 + returns_bl)

# # Plotting the portfolio values
# plt.figure(figsize=(12, 8))
# portfolio_values_bl.plot(title='Portfolio Value Over Time: Black-Litterman Portfolio')
# plt.xlabel('Date')
# plt.ylabel('Portfolio Value ($)')
# plt.show()

# # Plotting the portfolio weights
# plt.figure(figsize=(12, 8))
# for ticker in tickers:
#     plt.plot(portfolio_weights.index, portfolio_weights[ticker], label=ticker)
# plt.title('Portfolio Allocation Over Time')
# plt.xlabel('Date')
# plt.ylabel('Weight')
# plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
# plt.show()

# # Initialize DataFrame for equal-weighted portfolio weights and values
# equal_weights = np.full(len(tickers), 1 / len(tickers))
# portfolio_values_eq = pd.Series(index=price_data.index, dtype=float)
# portfolio_values_eq.iloc[0] = initial_portfolio_value  # Set initial value

# # Calculate portfolio values for equal-weighted portfolio
# for i in range(len(price_data)):
#     if i % rebalance_period == 0 or i == 0:
#         # Rebalance: Assign equal weights at the start and at each rebalance period
#         eq_weights = pd.Series(equal_weights, index=tickers)
#     if i > 0:
#         # Calculate returns for the equal-weighted portfolio
#         returns_eq = np.dot(eq_weights, price_data.pct_change(fill_method=None).iloc[i])
#         portfolio_values_eq.iloc[i] = portfolio_values_eq.iloc[i - 1] * (1 + returns_eq)

# # Plotting both portfolio values for comparison
# plt.figure(figsize=(12, 8))
# portfolio_values_bl.plot(label='Black-Litterman Portfolio')
# portfolio_values_eq.plot(label='Equal-Weighted Portfolio')
# plt.title('Portfolio Value Over Time: Black-Litterman vs. Equal-Weighted')
# plt.xlabel('Date')
# plt.ylabel('Portfolio Value ($)')
# plt.legend()
# plt.show()


# fetch_data(ticker)