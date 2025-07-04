# Portfolio Analysis and Projection Tool with yfinance + Google Gemini

This Python script allows you to analyze your stock portfolio, calculate key metrics like Beta and Dividend Yield, and visualize allocations and future projections. It fetches real-time stock data using the ```yfinance``` library and generates a comprehensive PDF report.  Makes for a good boilerplate for MBA/finance projects.  Have fun and always consult a real financial expert for real world financial decisions.

- We've enhanced the stock portfolio analysis script by integrating the Google Gemini API. This powerful addition allows the script to leverage Google's advanced large language models (LLMs) to provide an intelligent, concise summary of your portfolio's health.

- Instead of just presenting raw data, the Gemini API processes key portfolio metrics and holdings information to generate a narrative overview. This AI-powered summary offers quick, qualitative insights into aspects like diversification, risk profile (based on beta), and potential income from dividends, giving you a more holistic understanding of your investments.

- You'll see this summary seamlessly integrated into the text output and the final PDF report, providing an extra layer of valuable analysis.

## Features
- **Portfolio Summary**: Displays total portfolio value, weighted average Beta, and weighted average Dividend Yield.
- **Individual Holdings Detail**: Provides a breakdown of each stock's shares owned, current price, total value, portfolio weight, Beta, market capitalization, and dividend yield.
- **Portfolio Allocation by Value Plot**: A bar chart showing the value distribution across different stock tickers.
- **Individual Stock Dividend Yields Plot**: A bar chart comparing individual stock dividend yields against the portfolio's average weighted dividend yield.
- **Portfolio Allocation by Market Pie Chart**: Visualizes the portfolio's distribution across different global markets (e.g., 'us_market', 'sg_market').
- **Stock Beta Distribution Plot**: A histogram showing the distribution of individual stock Betas within your portfolio.
- **10-Year Portfolio Value Projection Plot**: Projects your portfolio's value over the next decade, assuming a 10% annual growth rate.

*PDF Report Generation: All summary text and plots are compiled into a single, multi-page PDF document for easy sharing and review.

## Prerequisites
Before running the script, ensure you have the following Python libraries installed. You can install them using pip:

## How to Use
- **Save the Code**: Save the provided Python code as a .py file (e.g., portfolio_analyzer.py).

- **Customize Your Portfolio**: Open the stock_portfolio_summarizer.py file and modify the portfolio_input dictionary with your actual stock holdings (ticker symbol and number of shares owned):

- Replace 'AAPL', 'MSFT', 'JPM' with your desired stock tickers and update SharesOwned accordingly or parameterize the .py file to accept portfolio data.

- **Run the Script**: Open your terminal or command prompt, navigate to the directory where you saved the file, and run the script:

## Output

Upon successful execution, the script will:

- Print a detailed portfolio summary and individual holdings information to your console.
- Generate a PDF file named Portfolio_Analysis_Report.pdf in the same directory as the script. This PDF will contain:
-- A page with the text summary printed to the console.
-- Individual pages for each generated plot (Portfolio Allocation by Value, Dividend Yields, Market Allocation, Beta Distribution, and 10-Year Projection).

## Important Notes
- **Internet Connection**: The script requires an active internet connection to fetch real-time stock data from Yahoo Finance.
- **API Rate Limits**: Frequent or rapid requests to Yahoo Finance might lead to temporary blocking. The script includes small time.sleep(1) pauses to mitigate this, but very large portfolios might still encounter issues.
- **Data Availability**: Some less common tickers might not have complete data (e.g., dividend yield or beta) available on Yahoo Finance, which could result in warnings or None values for those specific metrics.
- **Growth Projection**: The 10-year projection assumes a constant 10% annual growth rate. This is a simplified model and does not account for market volatility, changing dividends, or other real-world factors.
- **Beta Calculation**: If Yahoo Finance does not provide a Beta value directly, the script attempts to calculate it based on 5 years of weekly historical data against the S&P 500 (^GSPC). This calculation might fail if sufficient historical data is not available.
