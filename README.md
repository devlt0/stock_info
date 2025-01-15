# stock_info
experiments with python, yahoo finance (yfinance), and technical indicators (pandas_ta &amp; TA-lib)


Useful sqlite db browser
https://sqlitebrowser.org/



Currently includes most stocks listed in NYSE & NASDAQ in /shards/<exchange_name>/<ticker_symbol>.db <br>
Does not yet include delisted stocks as that information is not readily available via yahoo finance api
<br>
<br>
~5800 tickers combined between NYSE & NASDAQ exchanges with over 30 indicators applied (see https://github.com/devlt0/stock_info/blob/main/yfinance_downloader.py) currently <15gb
