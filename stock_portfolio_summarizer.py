import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import yfinance as yf
import warnings
import time
from matplotlib.backends.backend_pdf import PdfPages
import io
import sys

print('start')
# Suppress potential warnings from yfinance or pandas that aren't critical for this script
warnings.filterwarnings('ignore')

# --- Beta Calculation Function ---
def calculate_beta(stock_ticker, benchmark_ticker='^GSPC', period='5y', interval='1wk'):
    """
    Calculates the beta for a given stock against a benchmark index.

    Args:
        stock_ticker (str): Ticker symbol of the stock.
        benchmark_ticker (str): Ticker symbol of the benchmark index (default: S&P 500).
        period (str): Period for historical data (e.g., '1y', '5y', 'max').
        interval (str): Data interval (e.g., '1d', '1wk', '1mo').

    Returns:
        float: Calculated beta, or None if data is insufficient or calculation fails.
    """
    try:
        stock_data = yf.download(stock_ticker, period=period, interval=interval, progress=False)
        benchmark_data = yf.download(benchmark_ticker, period=period, interval=interval, progress=False)

        if stock_data.empty or benchmark_data.empty:
            print(f"      Warning: Insufficient historical data from yfinance for beta calculation for {stock_ticker}.")
            return None

        stock_price_col = 'Adj Close' if 'Adj Close' in stock_data.columns else 'Close'
        if stock_price_col not in stock_data.columns:
            print(f"      Error: Neither 'Adj Close' nor 'Close' found for {stock_ticker}.")
            return None
        stock_prices = stock_data[stock_price_col].squeeze()

        benchmark_price_col = 'Adj Close' if 'Adj Close' in benchmark_data.columns else 'Close'
        if benchmark_price_col not in benchmark_data.columns:
            print(f"      Error: Neither 'Adj Close' nor 'Close' found for benchmark {benchmark_ticker}.")
            return None
        benchmark_prices = benchmark_data[benchmark_price_col].squeeze()

        merged_prices = pd.DataFrame({'Stock': stock_prices, 'Benchmark': benchmark_prices}).dropna()

        if len(merged_prices) < 2:
            print(f"      Warning: Not enough overlapping valid price data for beta calculation for {stock_ticker} after merging. Found {len(merged_prices)} points.")
            return None

        stock_returns = merged_prices['Stock'].pct_change().dropna()
        benchmark_returns = merged_prices['Benchmark'].pct_change().dropna()

        common_index = stock_returns.index.intersection(benchmark_returns.index)

        if common_index.empty:
            print(f"      Warning: No common dates for return data after intersection for beta calculation for {stock_ticker}.")
            return None

        stock_returns_aligned = stock_returns.loc[common_index]
        benchmark_returns_aligned = benchmark_returns.loc[common_index]

        stock_returns_arr = stock_returns_aligned.values.astype(float)
        benchmark_returns_arr = benchmark_returns_aligned.values.astype(float)

        finite_mask = np.isfinite(stock_returns_arr) & np.isfinite(benchmark_returns_arr)
        stock_returns_final = stock_returns_arr[finite_mask]
        benchmark_returns_final = benchmark_returns_arr[finite_mask]

        if len(stock_returns_final) < 2 or len(benchmark_returns_final) < 2:
            print(f"      Warning: Not enough aligned and finite return data points ({len(stock_returns_final)} for stock, {len(benchmark_returns_final)} for benchmark) for beta calculation for {stock_ticker}. Covariance/Variance requires at least 2 points. Returning None.")
            return None

        if np.all(np.isclose(stock_returns_final, stock_returns_final[0])):
            print(f"      Warning: Stock returns for {stock_ticker} are constant or nearly constant (all values approximately {stock_returns_final[0]:.6f}). Cannot calculate meaningful beta.")
            return None
        if np.all(np.isclose(benchmark_returns_final, benchmark_returns_final[0])):
            print(f"      Warning: Benchmark returns for {benchmark_ticker} are constant or nearly constant (all values approximately {benchmark_returns_final[0]:.6f}). Cannot calculate meaningful beta.")
            return None

        try:
            covariance_matrix = np.cov(np.atleast_2d(stock_returns_final), np.atleast_2d(benchmark_returns_final))
            covariance = covariance_matrix[0, 1]
            variance_benchmark = covariance_matrix[1, 1]
        except ValueError as ve:
            print(f"      Error within np.cov for {stock_ticker}: ValueError - {ve}")
            print(f"      This suggests a problem with the processed return data, possibly it became degenerate (e.g., only one element or all identical values despite prior checks).")
            return None
        except Exception as e:
            print(f"      Unexpected error during np.cov for {stock_ticker}: {type(e).__name__} - {e}")
            return None

        if variance_benchmark == 0 or np.isclose(variance_benchmark, 0) or np.isnan(variance_benchmark):
            print(f"      Warning: Variance of benchmark returns is zero, NaN, or very close to zero for {stock_ticker}. Cannot calculate beta.")
            return None
        if np.isnan(covariance):
            print(f"      Warning: Covariance of returns is NaN for {ticker_symbol}. Cannot calculate beta.")
            return None

        calculated_beta = covariance / variance_benchmark
        return calculated_beta

    except Exception as e:
        print(f"      Error calculating beta for {stock_ticker}: {type(e).__name__} - {e}")
        return None

# --- 1. Your Portfolio Input ---
# Define the stocks in your portfolio and the number of shares you own.
# You can customize this dictionary with your actual holdings.
portfolio_input = {
    'AAPL': {'SharesOwned': 10},
    'MSFT': {'SharesOwned': 25},
    'JPM': {'SharesOwned': 20},
    'GOOGL': {'SharesOwned': 10},
    'PINS': {'SharesOwned': 10},
    'CRM': {'SharesOwned': 10},
    'PFE': {'SharesOwned': 10},
    'AMD': {'SharesOwned': 10}
}

# --- 2. Fetch Data from Yahoo Finance ---
portfolio_data_list = []
failed_tickers = []
benchmark_ticker = '^GSPC' # S&P 500 (standard benchmark)

print("Fetching stock data from Yahoo Finance...")
for ticker_symbol, details in portfolio_input.items():
    print(f"  Fetching data for {ticker_symbol}...")
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info

        current_price = info.get('regularMarketPrice', info.get('currentPrice'))
        market_cap = info.get('marketCap')
        long_name = info.get('longName', ticker_symbol)
        market = info.get('market', 'N/A') # Get market info

        # Get dividend yield (using trailingAnnualDividendYield as it's more common for current yield)
        dividend_yield = info.get('trailingAnnualDividendYield')

        beta = info.get('beta')
        if beta is None:
            calculated_beta = calculate_beta(ticker_symbol, benchmark_ticker=benchmark_ticker, period='5y', interval='1wk')
            beta = calculated_beta
            if beta is not None:
                print(f"    Calculated Beta for {ticker_symbol}: {beta:.4f}")
            else:
                print(f"    Could not calculate beta for {ticker_symbol}.")

        # Check if any essential data is still missing after fetching and calculation attempts
        if current_price is None or market_cap is None or beta is None:
            missing_fields = []
            if current_price is None: missing_fields.append('Current Price')
            if market_cap is None: missing_fields.append('Market Cap')
            if beta is None: missing_fields.append('Beta')

            print(f"    Warning: Missing essential data for {ticker_symbol}: {', '.join(missing_fields)}. Skipping.")
            failed_tickers.append(ticker_symbol)
            time.sleep(1) # Pause to avoid overwhelming API
            continue # Skip to the next ticker in the loop

        portfolio_data_list.append({
            'Ticker': ticker_symbol,
            'LongName': long_name,
            'SharesOwned': details['SharesOwned'],
            'CurrentPrice': current_price,
            'MarketCap': market_cap,
            'Market': market, # Store market
            'DividendYield': dividend_yield, # Store dividend yield
            'Beta': beta
        })
        time.sleep(1) # Pause to avoid overwhelming API
    except Exception as e:
        print(f"    Error fetching data for {ticker_symbol}: {type(e).__name__} - {e}. Skipping.")
        failed_tickers.append(ticker_symbol)
        time.sleep(1) # Pause to avoid overwhelming API

if not portfolio_data_list:
    print("No valid stock data fetched. Please check your ticker symbols, yfinance version, or internet connection.")
    exit() # Exit if no data was successfully fetched to prevent errors

portfolio_df = pd.DataFrame(portfolio_data_list)

# Calculate total value for each holding and market cap in billions
portfolio_df['TotalValue'] = portfolio_df['SharesOwned'] * portfolio_df['CurrentPrice']
portfolio_df['MarketCap_Billions'] = portfolio_df['MarketCap'] / 1_000_000_000

# Function to determine capitalization size
def get_capital_size(market_cap_billions):
    if market_cap_billions >= 200:
        return 'Mega-Cap'
    elif market_cap_billions >= 10:
        return 'Large-Cap'
    elif market_cap_billions >= 2:
        return 'Mid-Cap'
    else:
        return 'Small-Cap'

portfolio_df['Capitalization'] = portfolio_df['MarketCap_Billions'].apply(get_capital_size)

# --- 3. Calculate Portfolio Metrics ---
total_portfolio_value = portfolio_df['TotalValue'].sum()

# Calculate Portfolio Beta (Weighted Average Beta)
# Portfolio Beta = SUM( (Weight of Stock i) * (Beta of Stock i) )
portfolio_df['Weight'] = portfolio_df['TotalValue'] / total_portfolio_value
portfolio_beta = (portfolio_df['Weight'] * portfolio_df['Beta']).sum()

# Calculate Portfolio Dividend Yield (Weighted Average)
# Portfolio Dividend Yield = SUM( (Weight of Stock i) * (Dividend Yield of Stock i) )
# Handle cases where DividendYield might be None/NaN
portfolio_df['AdjustedDividendYield'] = portfolio_df['DividendYield'].fillna(0) # Treat None/NaN as 0 for calculation
portfolio_dividend_yield = (portfolio_df['Weight'] * portfolio_df['AdjustedDividendYield']).sum()


# --- 4. Display Portfolio Summary & Capture Text Output ---
# Redirect stdout to capture print statements
old_stdout = sys.stdout
redirected_output = io.StringIO()
sys.stdout = redirected_output

print("\n--- Portfolio Summary ---")
print(f"Total Portfolio Value: ${total_portfolio_value:,.2f}")
print(f"Portfolio Beta (Weighted Average): {portfolio_beta:.4f}")
print(f"Portfolio Dividend Yield (Weighted Average): {portfolio_dividend_yield:.4f}") # Display portfolio yield
print("\nIndividual Holdings:")
# Display updated DataFrame, removing 'Industry' and adding 'Market' and 'DividendYield'
print(portfolio_df[['Ticker', 'LongName', 'SharesOwned', 'CurrentPrice', 'TotalValue', 'Weight', 'Beta', 'Capitalization', 'Market', 'DividendYield']].to_string(index=False))
print("\n--- Additional Information ---")
print("\nAnalysis Complete.")

# Restore stdout
sys.stdout = old_stdout
captured_text_output = redirected_output.getvalue()
print(captured_text_output) # Print to console as usual

# --- 5. Generate Basic Visualizations & Save to PDF ---
plt.style.use('ggplot') # Use a nice ggplot style for charts

# Create a PDF file to save all plots
# Define the output PDF file name
pdf_output_filename = 'Portfolio_Analysis_Report.pdf'

with PdfPages(pdf_output_filename) as pdf:
    # Page 1: Text Summary
    fig_text = plt.figure(figsize=(8.5, 11)) # A4 size paper
    ax_text = fig_text.add_subplot(111)
    ax_text.text(0.05, 0.95, captured_text_output, transform=ax_text.transAxes,
                 fontsize=6, verticalalignment='top', family='monospace')
    ax_text.axis('off') # Hide axes for text display
    ax_text.set_title('Portfolio Summary and Details', fontsize=14, pad=20)
    pdf.savefig(fig_text)
    plt.close(fig_text) # Close the figure to free memory

    # Plot 1: Portfolio Allocation by Value
    fig1 = plt.figure(figsize=(10, 7))
    sns.barplot(x='Ticker', y='TotalValue', data=portfolio_df, palette='viridis')
    plt.title('Portfolio Allocation by Value')
    plt.xlabel('Ticker')
    plt.ylabel('Total Value ($)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    pdf.savefig(fig1) # Save the current figure to the PDF
    plt.close(fig1) # Close the figure to free memory

    # NEW Plot: Dividend Yield of the Portfolio
    fig2 = plt.figure(figsize=(10, 7))
    sns.barplot(x='Ticker', y='DividendYield', data=portfolio_df.dropna(subset=['DividendYield']), palette='cividis')
    plt.axhline(portfolio_dividend_yield, color='red', linestyle='--', label=f'Portfolio Avg Yield: {portfolio_dividend_yield:.2%}')
    plt.title('Individual Stock Dividend Yields and Portfolio Average')
    plt.xlabel('Ticker')
    plt.ylabel('Dividend Yield')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    pdf.savefig(fig2)
    plt.close(fig2)

    # NEW Plot: Portfolio Allocation by Market (Pie Chart)
    market_allocation = portfolio_df.groupby('Market')['TotalValue'].sum()
    if not market_allocation.empty:
        fig3 = plt.figure(figsize=(10, 10))
        plt.pie(market_allocation, labels=market_allocation.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
        plt.title('Portfolio Allocation by Market')
        plt.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.tight_layout()
        pdf.savefig(fig3)
        plt.close(fig3)
    else:
        print("\nWarning: No market data available for pie chart, skipping PDF inclusion.")

    # Plot 3: Beta Distribution
    fig4 = plt.figure(figsize=(8, 6))
    sns.histplot(portfolio_df['Beta'].dropna(), bins=5, kde=True, color='skyblue')
    plt.title('Distribution of Stock Betas in Portfolio')
    plt.xlabel('Beta')
    plt.ylabel('Number of Stocks')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    pdf.savefig(fig4)
    plt.close(fig4)

    # NEW Plot: 10 Year Projection of Portfolio Value (assuming 10% growth)
    projection_years = 10
    growth_rate = 0.10
    projected_values = [total_portfolio_value]
    years = [0]

    for i in range(1, projection_years + 1):
        projected_values.append(projected_values[-1] * (1 + growth_rate))
        years.append(i)

    fig5 = plt.figure(figsize=(10, 7))
    plt.plot(years, projected_values, marker='o', linestyle='-', color='green')
    plt.title(f'10-Year Portfolio Value Projection ({growth_rate*100:.0f}% Annual Growth)')
    plt.xlabel('Years from Now')
    plt.ylabel('Projected Portfolio Value ($)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ticklabel_format(style='plain', axis='y') # Prevent scientific notation on y-axis
    plt.tight_layout()
    pdf.savefig(fig5)
    plt.close(fig5)

print(f"\nAll plots and summary text saved to {pdf_output_filename}")
print("\nAnalysis Complete.")
