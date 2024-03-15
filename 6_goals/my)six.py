import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import schedule
import time
import alpaca_trade_api as tradeapi
import pytz

# Assuming API key is correctly set
api_key = 'c9864e890304fb88b00e5227e8423b1d'

# Function to calculate CAGR
def calculate_cagr(start_value, end_value, num_years):
    return (end_value / start_value) ** (1 / num_years) - 1

import requests
from datetime import datetime, timedelta

# Function to fetch stock price from Financial Modeling Prep API
def get_stock_price_eodhd(ticker, api_key, days_back=3008):
    """
    Fetch historical stock price data for a given ticker from Financial Modeling Prep API,
    starting from a specific number of days back until today.

    Parameters:
    - ticker: Stock ticker symbol as a string.
    - api_key: Your Financial Modeling Prep API key as a string.
    - days_back: Integer specifying how many days back from today the data should start.

    Returns:
    - Tuple containing the ticker and its historical data as a list of dictionaries.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)  # Calculate start date based on days_back
    base_url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}"
    params = {
        'from': start_date.strftime('%Y-%m-%d'),
        'to': end_date.strftime('%Y-%m-%d'),
        'apikey': api_key
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json().get('historical', [])
        # Adapt the structure if needed, depending on how FMP formats their response
        return ticker, data
    else:
        print(f"Failed to fetch data for {ticker}")
        return ticker, []

# Parallel data fetching (placeholder for actual function)
# Use ThreadPoolExecutor to fetch data in parallel
def fetch_data_concurrently(tickers, api_key):
    results = {}
    with ThreadPoolExecutor(max_workers=len(tickers)) as executor:
        future_to_ticker = {executor.submit(get_stock_price_eodhd, ticker, api_key): ticker for ticker in tickers}
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                ticker, data = future.result()
                results[ticker] = data
            except Exception as exc:
                print(f'{ticker} generated an exception: {exc}')
    return results

# Process fetched data into DataFrame
def process_fetched_data(stock_data):
    prices = {}
    for ticker, data in stock_data.items():
        if data:
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            prices[ticker] = df['adjClose']
    prices_df = pd.DataFrame(prices)
    return prices_df

# Objective function: Negative Sharpe Ratio
def neg_sharpe_ratio(weights, returns, Rf=0.02/252):
    port_return = np.dot(weights, returns.mean()) * 252
    port_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    sharpe_ratio = (port_return - Rf) / port_vol
    return -sharpe_ratio

# Placeholder for fetched stock data
stock_data = {}  # Assuming this is filled with the fetched stock data

# List of tickers

# List of tickers
tickers = [
    "MMM",  # 3M
    "AXP",  # American Express
    "AMGN", # Amgen
    "AAPL", # Apple
    "BA",   # Boeing
    "CAT",  # Caterpillar Inc.
    "CVX",  # Chevron Corp.
    "CSCO", # Cisco Systems
    "KO",   # The Coca-Cola Company
    "DOW",  # Dow
    "GS",   # Goldman Sachs
    "HD",   # Home Depot
    "HON",  # Honeywell
    "INTC", # Intel Corp.
    "IBM",  # International Business Machines
    "JNJ",  # Johnson & Johnson Corporation
    "JPM",  # JPMorgan Chase & Co.
    "MCD",  # McDonald's Corporation
    "MRK",  # Merck & Co. Inc.
    "MSFT", # Microsoft
    "NKE",  # Nike
    "PG",   # Procter & Gamble
    "CRM",  # Salesforce
    "TRV",  # The Travelers Companies
    "UNH",  # UnitedHealth Group
    "VZ",   # Verizon
    "V",    # Visa
    "WBA",  # Walgreens Boots Alliance
    "WMT",  # Wal-Mart Stores Inc.
    "DIS",   # Walt Disney Company
    "GLD",
]
stock_data = fetch_data_concurrently(tickers, api_key)

# Process the fetched data
prices_df = process_fetched_data(stock_data)

# Calculate daily returns
returns = prices_df.pct_change().dropna()

# Define the negative Sharpe ratio (objective function to minimize)
def neg_sharpe_ratio(weights, returns, risk_free_rate=0.052):
    port_return = np.dot(weights, returns.mean()) * 252  # Annual return
    port_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))  # Annual volatility
    sharpe_ratio = (port_return - risk_free_rate) / port_vol
    return -sharpe_ratio

def perform_rolling_optimization(returns, window=252, step=10, max_weight_per_stock=1):
    num_assets = len(returns.columns)
    optimized_weights = pd.DataFrame(index=returns.index, columns=returns.columns)
    for start in range(0, len(returns) - window + 1, step):
        end = start + window
        rolling_returns = returns.iloc[start:end]
        initial_guess = np.array([1. / num_assets] * num_assets)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, max_weight_per_stock) for asset in range(num_assets))
        result = minimize(neg_sharpe_ratio, initial_guess, args=(rolling_returns,),
                          method='SLSQP', bounds=bounds, constraints=constraints)
        if result.success:
            optimized_weights.iloc[end - 1] = result.x
        else:
            optimized_weights.iloc[end - 1] = np.nan
    optimized_weights.ffill(inplace=True)
    return optimized_weights


# Perform rolling optimization
optimized_weights = perform_rolling_optimization(returns)

# Calculate daily returns for the rollingly optimized portfolio
optimized_portfolio_daily_returns = (optimized_weights.shift(1) * returns).sum(axis=1)

# Initial investment and equal weights for the equally weighted portfolio
initial_investment = 1000 * len(returns.columns)
equal_weights = np.array([1. / len(returns.columns)] * len(returns.columns))

# Calculate daily returns for the equally weighted portfolio
equal_weighted_daily_returns = returns.dot(equal_weights)

# Calculate cumulative returns
optimized_cumulative_returns = (1 + optimized_portfolio_daily_returns).cumprod() * initial_investment
equal_weighted_cumulative_returns = (1 + equal_weighted_daily_returns).cumprod() * initial_investment



def calculate_cagr(start_value, end_value, num_years):
    return (end_value / start_value) ** (1 / num_years) - 1

# First, filter the DataFrame to start from 2019
start_date = '2016-12-01'
end_date = '2023-12-31'

# Filtering the dataframes to the specified date range
# Ensure the index is sorted
optimized_cumulative_returns.sort_index(inplace=True)

# Handling FutureWarning by ensuring no silent downcasting
optimized_weights = optimized_weights.apply(pd.to_numeric, errors='ignore', downcast=None)
optimized_weights.ffill(inplace=True)

# Now, ensure the slicing dates are within the index range
if pd.to_datetime(start_date) in optimized_cumulative_returns.index and pd.to_datetime(end_date) in optimized_cumulative_returns.index:
    optimized_cumulative_returns_filtered = optimized_cumulative_returns.loc[start_date:end_date]
else:
    print("Date range is out of bounds. Please adjust your start_date and end_date.")

# Continue with the subsequent operations...

optimized_cumulative_returns_filtered = optimized_cumulative_returns.loc[start_date:end_date]
equal_weighted_cumulative_returns_filtered = equal_weighted_cumulative_returns.loc[start_date:end_date]

# Normalizing to base 100
optimized_base100 = 100 * optimized_cumulative_returns_filtered / optimized_cumulative_returns_filtered.iloc[0]
equal_weighted_base100 = 100 * equal_weighted_cumulative_returns_filtered / equal_weighted_cumulative_returns_filtered.iloc[0]

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(optimized_base100, label='Rollingly Optimized Portfolio', color='blue')
plt.plot(equal_weighted_base100, label='Equally Weighted Portfolio', color='green')
plt.title('Portfolio Performance (2019-2023) Normalized to Base 100')
plt.xlabel('Date')
plt.ylabel('Portfolio Value (Base 100)')
plt.legend()
plt.grid(True)
plt.show()


# Assumed annual risk-free rate for the calculations
risk_free_rate = 0.02  # 2%

# Calculate annual returns for both portfolios
annual_return_optimized = optimized_portfolio_daily_returns.mean() * 252
annual_return_equal_weighted = equal_weighted_daily_returns.mean() * 252

# Calculate annual volatility for both portfolios
annual_volatility_optimized = optimized_portfolio_daily_returns.std() * np.sqrt(252)
annual_volatility_equal_weighted = equal_weighted_daily_returns.std() * np.sqrt(252)

# Calculate Sharpe Ratios for both portfolios
sharpe_ratio_optimized = (annual_return_optimized - risk_free_rate) / annual_volatility_optimized
sharpe_ratio_equal_weighted = (annual_return_equal_weighted - risk_free_rate) / annual_volatility_equal_weighted

# Calculate Alpha of the rollingly optimized portfolio (using CAPM formula)
# Note: For simplicity, assuming the equally weighted portfolio as the "market"
# and not adjusting for beta, as it would require a more complex calculation.
alpha_optimized = annual_return_optimized - (risk_free_rate + (annual_return_equal_weighted - risk_free_rate))

# Print the calculated metrics
print(f"Annual Return (Optimized): {annual_return_optimized:.4f}")
print(f"Annual Return (Equal Weighted): {annual_return_equal_weighted:.4f}")
print(f"Annual Volatility (Optimized): {annual_volatility_optimized:.4f}")
print(f"Annual Volatility (Equal Weighted): {annual_volatility_equal_weighted:.4f}")
print(f"Sharpe Ratio (Optimized): {sharpe_ratio_optimized:.4f}")
print(f"Sharpe Ratio (Equal Weighted): {sharpe_ratio_equal_weighted:.4f}")
print(f"Alpha (Optimized compared to Equal Weighted): {alpha_optimized:.4f}")

import matplotlib.pyplot as plt
import numpy as np

# Function to calculate drawdown
def calculate_drawdown(cumulative_returns):
    # Calculate the cumulative max of the portfolio's cumulative returns to get the running peak
    running_max = np.maximum.accumulate(cumulative_returns)
    # Calculate the drawdown by comparing the current value to the running max
    drawdown = (cumulative_returns - running_max) / running_max
    return drawdown

# Calculate cumulative returns for the Max Sharpe Ratio portfolio
cumulative_returns_max_sharpe = (1 + optimized_portfolio_daily_returns).cumprod()

# Calculate the drawdown
drawdown_max_sharpe = calculate_drawdown(cumulative_returns_max_sharpe)

# Plotting the drawdown
plt.figure(figsize=(14, 7))
drawdown_max_sharpe.plot()
plt.title('Drawdown of the Rollingly Optimized Portfolio (Max Sharpe Ratio)')
plt.xlabel('Date')
plt.ylabel('Drawdown')
plt.grid(True)
plt.show()

# Objective function to maximize return
def neg_portfolio_return(weights, returns):
    return -np.dot(weights, returns.mean()) * 252  # Negative for minimization
    
def perform_rolling_optimization_max_return(returns, window=252, step=10, max_weight_per_stock=1):
    num_assets = len(returns.columns)
    optimized_weights = pd.DataFrame(index=returns.index, columns=returns.columns)
    for start in range(0, len(returns) - window + 1, step):
        end = start + window
        rolling_returns = returns.iloc[start:end]
        initial_guess = np.array([1. / num_assets] * num_assets)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, max_weight_per_stock) for asset in range(num_assets))
        result = minimize(neg_portfolio_return, initial_guess, args=(rolling_returns,),
                          method='SLSQP', bounds=bounds, constraints=constraints)
        if result.success:
            optimized_weights.iloc[end - 1] = result.x
        else:
            optimized_weights.iloc[end - 1] = np.nan
    optimized_weights.ffill(inplace=True)
    return optimized_weights
# Perform rolling optimization focused on maximizing return
optimized_weights_max_return = perform_rolling_optimization_max_return(returns)

# Normalize the weights to ensure they sum to 1 across each row in case of minor discrepancies.
optimized_weights_normalized = optimized_weights.div(optimized_weights.sum(axis=1), axis='rows')

# Plotting
plt.figure(figsize=(14, 7))

# Create a stacked area plot to show the weights of each asset in the portfolio over time.
# The weights are normalized to sum to 1 at each point in time.
plt.stackplot(optimized_weights_normalized.index, optimized_weights_normalized.T, labels=optimized_weights_normalized.columns)

# Enhancing the plot
plt.title('Portfolio Allocation Over Time')
plt.xlabel('Date')
plt.ylabel('Weight in Portfolio')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Move the legend outside of the plot
plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # Show one tick per year for clarity
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format the ticks to show only the year
plt.xticks(rotation=45)  # Rotate the x-axis ticks for better readability
plt.tight_layout()  # Adjust the layout to make room for the legend

plt.show()

def min_volatility(weights, returns):
    port_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return port_vol

def perform_rolling_optimization_min_vol(returns, window=20, step=10, max_weight_per_stock=1):
    num_assets = len(returns.columns)
    optimized_weights = pd.DataFrame(index=returns.index, columns=returns.columns)
    for start in range(0, len(returns) - window + 1, step):
        end = start + window
        rolling_returns = returns.iloc[start:end]
        initial_guess = np.array([1. / num_assets] * num_assets)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, max_weight_per_stock) for asset in range(num_assets))
        result = minimize(min_volatility, initial_guess, args=(rolling_returns,),
                          method='SLSQP', bounds=bounds, constraints=constraints)
        if result.success:
            optimized_weights.iloc[end - 1] = result.x
        else:
            optimized_weights.iloc[end - 1] = np.nan
    optimized_weights.ffill(inplace=True)
    return optimized_weights

optimized_weights_min_vol = perform_rolling_optimization_min_vol(returns)
optimized_portfolio_daily_returns_min_vol = (optimized_weights_min_vol.shift(1) * returns).sum(axis=1)
optimized_cumulative_returns_min_vol = (1 + optimized_portfolio_daily_returns_min_vol).cumprod() * initial_investment

# Normalize to base 100 for comparison
optimized_base100_min_vol = 100 * optimized_cumulative_returns_min_vol / optimized_cumulative_returns_min_vol.iloc[0]

optimized_portfolio_daily_returns_max_return = (optimized_weights_max_return.shift(1) * returns).sum(axis=1)
optimized_cumulative_returns_max_return = (1 + optimized_portfolio_daily_returns_max_return).cumprod() * initial_investment

# Normalize to base 100 for comparison
optimized_base100_max_sharpe = 100 * optimized_cumulative_returns / optimized_cumulative_returns.iloc[0]
optimized_base100_max_return = 100 * optimized_cumulative_returns_max_return / optimized_cumulative_returns_max_return.iloc[0]
equal_weighted_base100 = 100 * equal_weighted_cumulative_returns / equal_weighted_cumulative_returns.iloc[0]



plt.figure(figsize=(14, 7))
plt.plot(optimized_base100_max_sharpe, label='Rollingly Optimized Portfolio (Max Sharpe)', color='blue')
plt.plot(optimized_base100_max_return, label='Rollingly Optimized Portfolio (Max Return)', color='red')
plt.plot(optimized_base100_min_vol, label='Rollingly Optimized Portfolio (Min Volatility)', color='purple')
plt.plot(equal_weighted_base100, label='Equally Weighted Portfolio', color='green')
plt.title('Comparison of Portfolio Performances with DJIA (Normalized to Base 100)')
plt.xlabel('Date')
plt.ylabel('Portfolio Value (Base 100)')
plt.legend()
plt.grid(True)
plt.show()

strategies_daily_returns = {
     "Max Sharpe": optimized_portfolio_daily_returns,
     "Max Return": optimized_portfolio_daily_returns_max_return,
     "Min Volatility": optimized_portfolio_daily_returns_min_vol,
     "Equal Weighted": equal_weighted_daily_returns,
}
import matplotlib.pyplot as plt
annual_returns = {strategy: daily_returns.mean() * 252 for strategy, daily_returns in strategies_daily_returns.items()}

# Convert annual returns dictionary to items for plotting
strategies, annual_returns_values = zip(*annual_returns.items())

plt.figure(figsize=(10, 6))
plt.bar(strategies, annual_returns_values, color=['blue', 'red', 'purple', 'green'])
plt.title('Annual Returns of Different Portfolio Strategies')
plt.xlabel('Strategy')
plt.ylabel('Annual Return')
plt.xticks(rotation=45)  # Rotate strategy names for better visibility if needed
plt.tight_layout()  # Adjust layout
plt.show()

annual_returns = {strategy: daily_returns.mean() * 252 for strategy, daily_returns in strategies_daily_returns.items()}
# Normalize the weights to sum to 1 across each row
optimized_weights_min_vol_normalized = optimized_weights_min_vol.div(optimized_weights_min_vol.sum(axis=1), axis='index')
plt.figure(figsize=(14, 7))

# Generate the plot
plt.stackplot(optimized_weights_min_vol_normalized.index, optimized_weights_min_vol_normalized.T, labels=optimized_weights_min_vol_normalized.columns)

# Enhancing the plot
plt.title('Minimum Volatility Portfolio Allocation Over Time')
plt.xlabel('Date')
plt.ylabel('Allocation Percentage')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Adjust the legend outside of the plot area
plt.margins(0, 0)  # Optional: Reduces the plot margins

# Improve readability of the x-axis dates
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=45)

plt.tight_layout()  # Adjust the layout to make room for the legend
plt.show()

def annualized_return(daily_returns):
    return np.mean(daily_returns) * 252

def annualized_volatility(daily_returns):
    return np.std(daily_returns) * np.sqrt(252)

def sharpe_ratio(daily_returns, risk_free_rate=0.02):
    return (annualized_return(daily_returns) - risk_free_rate) / annualized_volatility(daily_returns)

def max_drawdown(cumulative_returns):
    drawdowns = 1 - cumulative_returns / np.maximum.accumulate(cumulative_returns)
    return np.max(drawdowns)

# Calculate daily returns for DJIA for metrics calculation

# Pack strategies' daily returns for easy iteration
strategies_daily_returns = {
    "Max Sharpe": optimized_portfolio_daily_returns,
    "Max Return": optimized_portfolio_daily_returns_max_return,
    "Equal Weighted": equal_weighted_daily_returns,
}

# Print performance metrics for each strategy
for strategy_name, daily_returns in strategies_daily_returns.items():
    print(f"Metrics for {strategy_name} Strategy:")
    print(f"Annualized Return: {annualized_return(daily_returns):.4f}")
    print(f"Annualized Volatility: {annualized_volatility(daily_returns):.4f}")
    print(f"Sharpe Ratio: {sharpe_ratio(daily_returns):.4f}")
    
    # For Max Drawdown, calculate cumulative returns from daily returns
    cumulative_returns = (1 + daily_returns).cumprod()
    print(f"Maximum Drawdown: {max_drawdown(cumulative_returns):.4f}\n")

# Ensure the DataFrame is sorted by date if not already
optimized_weights_max_return = optimized_weights_max_return.sort_index()

# Plotting
plt.figure(figsize=(14, 7))
plt.stackplot(optimized_weights_max_return.index, optimized_weights_max_return.T, labels=optimized_weights_max_return.columns)
plt.title('Portfolio Allocation Over Time (Max Return Optimization)')
plt.xlabel('Date')
plt.ylabel('Allocation Percentage')
plt.legend(loc='upper left', bbox_to_anchor=(1, 0.5))  # Adjust legend placement to your liking

# Adding text to the plot
plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # One tick per year for clarity
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Display ticks in 'Year' format
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to make room for the legend if necessary
plt.show()

# Fetch SPY data for the same period
def fetch_spy_data(api_key):
    ticker = "SPY"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 3)  # Adjust according to your portfolio's period
    base_url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}"
    params = {
        'from': start_date.strftime('%Y-%m-%d'),
        'to': end_date.strftime('%Y-%m-%d'),
        'apikey': api_key
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json().get('historical', [])
        return pd.DataFrame(data)[['date', 'adjClose']]
    else:
        return pd.DataFrame(columns=['date', 'adjClose'])

# Process SPY data
spy_data = fetch_spy_data('c9864e890304fb88b00e5227e8423b1d')
spy_data['date'] = pd.to_datetime(spy_data['date'])
spy_data.set_index('date', inplace=True)
spy_returns = spy_data['adjClose'].pct_change().dropna()
spy_cumulative_returns = (1 + spy_returns).cumprod()

# Normalize to base 100 for comparison
spy_base100 = 100 * spy_cumulative_returns / spy_cumulative_returns.iloc[0]

# Plotting all optimized portfolios + SPY
plt.figure(figsize=(14, 7))
plt.plot(optimized_base100_max_sharpe, label='Max Sharpe', color='blue')
plt.plot(optimized_base100_max_return, label='Max Return', color='red')
plt.plot(optimized_base100_min_vol, label='Min Volatility', color='purple')
plt.plot(equal_weighted_base100, label='Equally Weighted', color='green')
plt.plot(spy_base100, label='S&P 500', color='black', linestyle='--')  # SPY as a dashed line for comparison
plt.title('Portfolio Performances vs. S&P 500 (Normalized to Base 100)')
plt.xlabel('Date')
plt.ylabel('Portfolio Value (Base 100)')
plt.legend()
plt.grid(True)
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

def min_volatility_for_target_return(returns, target_return):
    num_assets = len(returns.columns)
    args = (returns,)
    
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Sum of weights = 1
        {'type': 'eq', 'fun': lambda x: target_return_portfolio(x, returns) - target_return}  # Portfolio return = target return
    ]
    
    bounds = tuple((0, 1) for asset in range(num_assets))
    initial_guess = [1./num_assets] * num_assets
    
    result = minimize(min_volatility, initial_guess, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    print("Muhammad Umais TARGET RETURN", result.x)
    return result.x

def target_return_portfolio(weights, returns):
    return np.dot(weights, returns.mean()) * 252

target_return = 0.10  # Example target return

min_volatility_for_target_return(returns, target_return)