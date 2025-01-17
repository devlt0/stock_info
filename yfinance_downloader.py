from datetime import datetime, timedelta
from os import listdir, getcwd
from time import sleep
from traceback import print_exc
import sqlite3

import pandas as pd
import pandas_ta as ta
import yfinance as yf

chip_tickers = ["ARM", "TSM"]
bigtech_tickers = ["AAPL", "NVDA", "MSFT", "AMZN", "META", "GOOG"]
dow30_tickers = [
    "AAPL", "NVDA", "MSFT", "AMZN", "WMT", "JPM", "V", "UNH", "PG", "HD",
    "JNJ", "CRM", "KO", "CVX", "MRK", "CSCO", "AXP", "MCD", "IBM", "DIS",
    "GS", "CAT", "VZ", "HON", "AMGN", "BA", "NKE", "SHW", "MMM", "TRV"
    ]

nasdaq100_tickers = [
    "AAPL", "NVDA", "MSFT", "GOOG", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "COST",
    "NFLX", "ASML", "TMUS", "CSCO", "PEP", "AZN", "LIN", "AMD", "ADBE", "ISRG",
    "INTU", "PLTR", "TXN", "QCOM", "BKNG", "HON", "CMCSA", "AMGN", "PDD", "AMAT",
    "ARM", "PANW", "ADP", "GILD", "APP", "ADI", "VRTX", "SBUX", "MRVL", "MU",
    "LRCX", "INTC", "MELI", "PYPL", "KLAC", "CRWD", "CDNS", "ABNB", "MDLZ", "MAR",
    "REGN", "SNPS", "CTAS", "FTNT", "CEG", "DASH", "WDAY", "ORLY", "ADSK", "TEAM",
    "MSTR", "CSX", "TTD", "ROP", "CPRT", "CHTR", "PCAR", "NXPI", "MNST", "PAYX",
    "ROST", "AEP", "DDOG", "FANG", "LULU", "AXON", "KDP", "FAST", "BKR", "VRSK",
    "XEL", "EA", "CTSH", "EXC", "ODFL", "KHC", "CCEP", "GEHC", "IDXX", "TTWO",
    "MCHP", "DXCM", "ANSS", "CSGP", "ZS", "ON", "WBD", "GFS", "CDW", "BIIB", "MDB"
    ]

sap500_tickers = [
    "AAPL", "NVDA", "MSFT", "GOOG", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "BRK-B",
    "WMT", "LLY", "JPM", "V", "MA", "XOM", "ORCL", "UNH", "COST", "PG", "HD", "NFLX",
    "JNJ", "BAC", "CRM", "ABBV", "KO", "CVX", "TMUS", "MRK", "CSCO", "WFC", "ACN", "NOW",
    "PEP", "BX", "AXP", "MCD", "IBM", "MS", "DIS", "LIN", "TMO", "ABT", "AMD", "ADBE",
    "GS", "PM", "ISRG", "GE", "INTU", "CAT", "PLTR", "TXN", "QCOM", "VZ", "DHR", "BKNG",
    "T", "BLK", "SPGI", "RTX", "PFE", "NEE", "HON", "CMCSA", "PGR", "AMGN", "LOW", "ANET",
    "UNP", "SYK", "TJX", "SCHW", "C", "AMAT", "BA", "BSX", "KKR", "ETN", "COP", "UBER",
    "PANW", "ADP", "FI", "LMT", "GILD", "DE", "BMY", "NKE", "CB", "UPS", "ADI", "MMC",
    "VRTX", "SBUX", "MDT", "PLD", "MU", "LRCX", "EQIX", "GEV", "SO", "MO", "INTC", "MCO",
    "AMT", "PYPL", "ICE", "ELV", "SHW", "KLAC", "CRWD", "APH", "CME", "DUK", "TT", "CDNS",
    "CMG", "ABNB", "PH", "WM", "DELL", "MDLZ", "WELL", "AON", "MAR", "MSI", "CI", "REGN",
    "PNC", "HCA", "SNPS", "ITW", "USB", "CL", "CTAS", "ZTS", "FTNT", "MCK", "GD", "TDG",
    "EMR", "MMM", "CEG", "EOG", "ORLY", "NOC", "COF", "FDX", "ECL", "WMB", "BDX", "SPG",
    "APD", "ADSK", "RSG", "AJG", "CSX", "RCL", "TGT", "CARR", "KMI", "HLT", "DLR", "OKE",
    "GM", "TFC", "AFL", "MET", "CVS", "BK", "ROP", "SRE", "CPRT", "FCX", "TRV", "CHTR",
    "PCAR", "SLB", "AZO", "NSC", "NXPI", "PSA", "JCI", "AMP", "GWW", "MNST", "ALL", "PAYX",
    "ROST", "AEP", "FICO", "FANG", "CMI", "PSX", "MSCI", "VST", "O", "PWR", "LULU", "OXY",
    "URI", "AIG", "AXON", "D", "DHI", "MPC", "NDAQ", "HWM", "KR", "KMB", "EW", "KDP", "DFS",
    "FIS", "COR", "PCG", "TEL", "PRU", "NEM", "PEG", "AME", "FAST", "KVUE", "HES", "GLW",
    "BKR", "STZ", "LHX", "GRMN", "CBRE", "CCI", "F", "CTVA", "TRGP", "VRSK", "VLO", "DAL",
    "XEL", "EA", "A", "CTSH", "EXC", "ODFL", "SYY", "YUM", "IT", "LVS", "KHC", "OTIS", "LEN",
    "IR", "GEHC", "IQV", "GIS", "ACGL", "HSY", "VMC", "IDXX", "RMD", "WAB", "EXR", "CCL",
    "ETR", "TTWO", "ROK", "UAL", "DD", "HIG", "RJF", "EFX", "MLM", "WTW", "AVB", "MTB", "ED",
    "EIX", "DECK", "IRM", "MCHP", "VICI", "HPQ", "CNC", "HUM", "DXCM", "LYV", "WEC", "EBAY",
    "ANSS", "CSGP", "BRO", "MPWR", "STT", "CAH", "GPN", "FITB", "TSCO", "DOW", "XYL", "HPE",
    "EQR", "K", "SW", "KEYS", "GDDY", "PPG", "EQT", "NUE", "EL", "ON", "BR", "FTV", "WBD",
    "MTD", "DOV", "CHD", "TPL", "SYF", "VLTO", "TROW", "NVR", "DTE", "VTR", "TYL", "AWK",
    "ADM", "LYB", "PPL", "EXPE", "HAL", "AEE", "WST", "HBAN", "NTAP", "CPAY", "CDW", "FE",
    "HUBB", "CINF", "ROL", "PHM", "WRB", "BIIB", "FOXA", "PTC", "WAT", "DRI", "SBAC", "LII",
    "ATO", "TDY", "IFF", "ERIE", "FOX", "DVN", "RF", "ES", "ZBH", "CNP", "WDC", "TER", "MKC",
    "CBOE", "WY", "TSN", "NTRS", "STE", "ULTA", "LUV", "CLX", "PKG", "ZBRA", "CMS", "VRSN",
    "INVH", "CFG", "LDOS", "LH", "ESS", "FSLR", "CTRA", "IP", "MAA", "L", "COO", "BBY",
    "PODD", "NRG", "STX", "FDS", "BF-B", "SMCI", "SNA", "PFG", "STLD", "TRMB", "HRL", "JBHT",
    "NI", "ARE", "KEY", "GEN", "DGX", "OMC", "DG", "MOH", "PNR", "J", "BALL", "BLDR", "HOLX",
    "JBL", "GPC", "NWS", "UDR", "DLTR", "MRNA", "IEX", "KIM", "MAS", "NWSA", "EG", "ALGN",
    "EXPD", "TPR", "LNT", "AVY", "BAX", "VTRS", "CF", "FFIV", "DPZ", "DOC", "AKAM", "RL",
    "APTV", "TXT", "SWKS", "EVRG", "RVTY", "AMCR", "REG", "INCY", "EPAM", "CAG", "BXP",
    "POOL", "JKHY", "KMX", "CPT", "CPB", "JNPR", "SWK", "DVA", "HST", "CHRW", "NDSN", "UHS",
    "TAP", "SJM", "DAY", "PAYC", "TECH", "SOLV", "ALLE", "NCLH", "AIZ", "BG", "BEN", "EMN",
    "IPG", "MGM", "ALB", "AOS", "PNW", "FRT", "LKQ", "LW", "CRL", "WYNN", "GL", "ENPH",
    "GNRC", "AES", "HSIC", "APA", "MKTX", "TFX", "MTCH", "WBA", "IVZ", "MOS", "HAS", "CE",
    "MHK", "HII", "CZR", "PARA", "BWA", "QRVO", "FMC", "AMTM"
]

all_tickers = []
all_tickers.extend(sap500_tickers)
all_tickers.extend(nasdaq100_tickers)
all_tickers.extend(dow30_tickers)

combined_dow30_nasdaq100_sap500_tickers = list( set(  all_tickers ) )

def download_and_store_stock_data(tickers, start_date="1900-01-01", end_date="2025-01-01", db_name="sap500_nasdaq100_dow30_data.db", _use_adj_close=False):
    close_col = 'Adj Close' if _use_adj_close else 'Close'
    # in theory if using adj close should be using adj high and adj low, likely adj volume too
    # however- doing adj close = x * close and using x = adj close / close as modifier against high/low is inaccurate
    #  won't account for things like dividend distrubtion      (split, dividend, rights issue)
    # ToDo figure out how to get adj high/low/vol or accurate way to generate based of automated pull of docs


    columns = [
        'ticker', 'date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume', 'sma', 'ema', 'hma', 'tma', 'rma',
        'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'rsi', 'BBL_14_2', 'BBM_14_2', 'BBU_14_2', 'BBB_14_2',
        'BBP_14_2', 'ADX_14', 'DMP_14', 'DMN_14', 'AROOND_14', 'AROONU_14', 'AROONOSC_14', 'force_index', 'OBV',
        'accumulation_distribution', 'mfi', 'vwap', 'KVO_34_55_13', 'KVOs_34_55_13', 'cmf', 'EFI_13', 'chaikin_oscillator',
        'price_volume_trend', 'pvol', 'FWMA_10', 'atr', 'STOCHRSIk_14_14_3_3', 'STOCHRSId_14_14_3_3', 'TRIX_30_9', 'TRIXs_30_9',
        'PSARl_0_02_0_2', 'PSARs_0_02_0_2', 'PSARaf_0_02_0_2', 'PSARr_0_02_0_2', 'cci', 'donchian_upper', 'donchian_middle',
        'donchian_lower', 'SUPERT_7_3', 'SUPERTd_7_3', 'SUPERTl_7_3', 'SUPERTs_7_3', 'ISA_9', 'ISB_26', 'ITS_9',
        'IKS_26', 'rvi', 'NATR_14', 'candle_Doji', 'candle_Inside'
    ]
    #'sharpe_ratio', 'sortino_ratio', 'max_drawdown'#, 'pure_profit_score'
    ticker_iter = 1
    num_tickers = len(tickers)

    for ticker in tickers:

        try:

            print(f"Downloading data for {ticker}... {ticker_iter} out of {num_tickers}")
            ticker_iter += 1


            data = yf.download(ticker, start=start_date, end=end_date)
            #data['ticker'] = ticker
            #data["Date"] = data.index
            data = data.xs(ticker, level=1, axis=1)


            data['sma'] = ta.sma(data[close_col], length=14)
            data['ema'] = ta.ema(data[close_col], length=14)
            data['hma'] = ta.hma(data[close_col], length=14)
            data['tma'] = ta.trima(data[close_col], length=14)
            data['rma'] = ta.rma(data[close_col], length=14)
            data.ta.macd(close=data[close_col], fast=12, slow=26, signal=9, append=True)
            #data['macd'], data['macd_signal'], data['macd_hist'] = ta.trend.macd(data['Close'], fast=12, slow=26, signal=9, append=True)
            data['rsi'] = ta.rsi(data[close_col], length=14)
            #data['bollinger_upper'], data['bollinger_middle'], data['bollinger_lower'] = ta.bbands(data['Close'], length=14)
            data.ta.bbands(close=data[close_col], length=14, append=True)
            #data['adx'] = ta.adx(data['High'], data['Low'], data['Close'], length=14)
            data.ta.adx(append=True)
            #data['aroon_up'], data['aroon_down'] = ta.aroon(data['High'], data['Low'], length=14)
            data.ta.aroon(high=data['High'], low=data['Low'], length=14, append=True)
            data['force_index'] = ta.efi(data[close_col], data['Volume'], length=14)
            data.ta.obv(append=True)
            #data['obv'] = ta.obv(data['Close'], data['Volume'])
            #data['obv'] = ta.volume.obv(data['Close'], data['Volume'])
            data['accumulation_distribution'] = ta.ad(data['High'], data['Low'], data[close_col], data['Volume'])
            ## ToDo fix the error on int64 vs numpy arr float64 with mfi
            data['mfi'] = ta.mfi(data['High'], data['Low'], data[close_col], data['Volume'], length=14)
            #data['mfi'] = data.get('mfi', pd.Series(dtype='float64'))
            #data['mfi'] = pd.Series(ta.mfi(data['High'], data['Low'], data[close_col], data['Volume'], length=14), dtype='float64')
            data['vwap'] = ta.vwap(data['High'], data['Low'], data[close_col], data['Volume'])
            #data['klinger'] = ta.kvo(data['High'], data['Low'], data['Close'], data['Volume'])
            data.ta.kvo(append=True)
            data['cmf'] = ta.cmf(data['High'], data['Low'], data[close_col], data['Volume'], length=14)
            ##?accumulation swing index ?##data['asi'] = ta.asi(data['High'], data['Low'], data['Close'], data['Volume'])
            #data['efi'] = ta.efi(data['High'], data['Low'], data['Close'], data['Volume'])
            data.ta.efi(append=True)
            data['chaikin_oscillator'] = ta.cmf(data['High'], data['Low'], data[close_col], data['Volume'])
            data['price_volume_trend'] = ta.pvt(data[close_col], data['Volume'])
            data['pvol'] = ta.pvol(data['High'], data['Low'], data[close_col])
            #data['fibonacci_retracement'] = ta.fwma(data['High'], data['Low'])
            data.ta.fwma(append=True)
            data['atr'] = ta.atr(data['High'], data['Low'], data[close_col], length=14)
            #data['stochastic_rsi'] = ta.stochrsi(data['Close'], length=14)
            data.ta.stochrsi(append=True)
            #data['trix'] = ta.trix(data['Close'], length=14)
            data.ta.trix(append=True)
            #data['parabolic_sar'] = ta.psar(data['High'], data['Low'], data['Close'], af=0.02, max_af=0.2)
            data.ta.psar(append=True)
            data['cci'] = ta.cci(data['High'], data['Low'], data[close_col], length=14)
            #data['donchian_upper'], data['donchian_middle'], data['donchian_lower'] = ta.donchian(data['High'], data['Low'], length=14)
            data.ta.donchian(append=True)
            #data['supertrend'] = ta.supertrend(data['High'], data['Low'], data['Close'], length=14, multiplier=3)
            data.ta.supertrend(append=True)
            #data['ichimoku_cloud'] = ta.ichimoku(data['High'], data['Low'], data['Close'])
            data.ta.ichimoku(lookahead=False, append=True)
            #data['dmi_plus'], data['dmi_minus'], data['adx'] = ta.dmi(data['High'], data['Low'], data['Close'], length=14)
            data.ta.dm(append=True)
            data['rvi'] = ta.rvi(data[close_col], length=14)
            #data['normalized_average_true_range'] = ta.natr(data['Close'], length=14)
            data.ta.natr(append=True)

            # Candlestick patterns # ToDo fix and add the candlestick metrics
            # any other candle stick patterns require ta lib, getting py pkg not a problem, getting c library on host eh
            data['doji'] = data.ta.cdl_pattern(name=["doji"])
            data['inside'] = data.ta.cdl_pattern(name=["inside"])
            #data['3blackcrows'] = data.ta.cdl_pattern(name=["3blackcrows"])
            '''
            data['doji'] = ta.cdl_doji(data)
            data['engulfing'] = ta.cdl_engulfing(data)
            data['hammer'] = ta.cdl_hammer(data)
            data['morning_star'] = ta.cdl_morningstar(data)
            data['evening_star'] = ta.cdl_eveningstar(data)
            data['piercing'] = ta.cdl_piercing(data)
            data['dark_cloud_cover'] = ta.cdl_darkcloudcover(data)
            data['marubozu'] = ta.cdl_marubozu(data)
            data['shooting_star'] = ta.cdl_shootingstar(data)
            data['spinning_top'] = ta.cdl_spinningtop(data)
            data['dragonfly_doji'] = ta.cdl_dragonflydoji(data)
            data['gravestone_doji'] = ta.cdl_gravestonedoji(data)
            data['inside'] = ta.cdl_inside(data)
            data['harami'] = ta.cdl_harami(data)
            data['harami_cross'] = ta.cdl_haramicross(data)
            data['inverted_hammer'] = ta.cdl_invertedhammer(data)
            data['abandoned_baby'] = ta.cdl_abandonedbaby(data)
            data['three_white_soldiers'] = ta.cdl_threewhitesoldiers(data)
            data['three_black_crows'] = ta.cdl_threeblackcrows(data)
            '''
            #data = data.ta.cdl_pattern(name=["doji", "inside", "ha", "cdl_z"])  # Heikin-Ashi: ha     Z Score: cdl_z

            ## Performance Metrics (Sharpe Ratio, Sortino Ratio, etc.)
            #data['sharpe_ratio'] = ta.sharpe_ratio(data['Close'])
            #data['sortino_ratio'] = ta.sortino_ratio(data['Close'])
            #data['max_drawdown'] = ta.max_drawdown(data['Close'])

            #print(data.columns)
            data.insert(0, 'Date', data.index)
            data.insert(0, 'Ticker', ticker) # was ticker x_x

            # push dataframe to sql db / table
            db_name = f'{ticker}.db'
            conn = sqlite3.connect(db_name)
            cursor = conn.cursor()

            create_table_sql = f'''
            CREATE TABLE IF NOT EXISTS "{ticker}" (
                {", ".join([f"'{col}' TEXT" for col in data.columns])}
            )
            '''

            cursor.execute(create_table_sql)

            data.to_sql(ticker, conn, if_exists='replace', index=False)
            conn.commit()
            conn.close()

        except Exception as e:
            print(e)
            print_exc()
        sleep(1.9)
#ToDo merge functions add option for getting minutes vs days

def download_and_store_stock_data_minutes(tickers, start_date="2025-01-01", end_date="2025-01-01", db_name="", get_minutes=True):

    # use start date as current date - 30 d
    # end date as current date
    # iterate 7 days at a time b/c yahoo
    # limit to 1.8/1.9s per query to avoid rate limit
    # use while loop
    yahoo_lookback_max_limit = 29 #still causes issues b/c long run time and now not in loop x_X #days # tried 30 fails,  with today can go back 29 additional
    yahoo_efficient_lkbk_lmt = 28 # reduce 5 query per ticker to 4



    #db_name = db_name.replace(".db", f"_{start_date_obj.strftime('%Y-%m-%d')}_to_{end_date_obj.strftime('%Y-%m-%d')}_minute.db")



    columns = [
        'Ticker', 'Datetime', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume', 'sma', 'ema', 'hma', 'tma', 'rma',
        'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'rsi', 'BBL_14_2', 'BBM_14_2', 'BBU_14_2', 'BBB_14_2',
        'BBP_14_2', 'ADX_14', 'DMP_14', 'DMN_14', 'AROOND_14', 'AROONU_14', 'AROONOSC_14', 'force_index', 'OBV',
        #'accumulation_distribution', 'mfi', 'vwap', 'KVO_34_55_13', 'KVOs_34_55_13', 'cmf', 'EFI_13', 'chaikin_oscillator',
        'accumulation_distribution', 'KVO_34_55_13', 'KVOs_34_55_13', 'cmf', 'EFI_13', 'chaikin_oscillator',
        'price_volume_trend', 'pvol', 'FWMA_10', 'atr', 'STOCHRSIk_14_14_3_3', 'STOCHRSId_14_14_3_3', 'TRIX_30_9', 'TRIXs_30_9',
        'PSARl_0_02_0_2', 'PSARs_0_02_0_2', 'PSARaf_0_02_0_2', 'PSARr_0_02_0_2', 'cci', 'donchian_upper', 'donchian_middle',
        'donchian_lower', 'SUPERT_7_3', 'SUPERTd_7_3', 'SUPERTl_7_3', 'SUPERTs_7_3', 'ISA_9', 'ISB_26', 'ITS_9',
        'IKS_26', 'rvi', 'NATR_14', 'sharpe_ratio', 'sortino_ratio', 'max_drawdown'#, 'pure_profit_score'
    ]

    ticker_iter = 1
    num_tickers = len(tickers)
    rate_limit_delay = 1.9 # could go 1.8 for exactly 2000 req in one hour but on chance get blocked, add buffer of 0.1s
    for ticker in tickers:
        print(f"Downloading data for {ticker}... {ticker_iter} out of {num_tickers}")
        ticker_iter += 1

        end_date_obj   = datetime.now()
        start_date_obj = end_date_obj - timedelta(days=yahoo_efficient_lkbk_lmt)
        current_start_obj = start_date_obj
        first_pass = True
        data = pd.DataFrame()
        while current_start_obj < end_date_obj:

            current_end_obj = current_start_obj + timedelta(days=7)
            cur_data = yf.download(ticker, start=current_start_obj.strftime('%Y-%m-%d'), end=current_end_obj.strftime('%Y-%m-%d'), interval='1m')
            #sleep(rate_limit_delay)
            #data['ticker'] = ticker
            #data["Date"] = data.index
            cur_data = cur_data.xs(ticker, level=1, axis=1)

            data = pd.concat([data, cur_data]) if data is not None else cur_data #, ignore_index=True)
            data = data.drop_duplicates()
            current_start_obj += timedelta(days=7)
            sleep(rate_limit_delay)

        data = data.drop_duplicates()

        try:
            data['sma'] = ta.sma(data['Close'], length=14)
            data['ema'] = ta.ema(data['Close'], length=14)
            data['hma'] = ta.hma(data['Close'], length=14)
            data['tma'] = ta.trima(data['Close'], length=14)
            data['rma'] = ta.rma(data['Close'], length=14)
            data.ta.macd(close=data['Close'], fast=12, slow=26, signal=9, append=True)
            #data['macd'], data['macd_signal'], data['macd_hist'] = ta.trend.macd(data['Close'], fast=12, slow=26, signal=9, append=True)
            data['rsi'] = ta.rsi(data['Close'], length=14)
            #data['bollinger_upper'], data['bollinger_middle'], data['bollinger_lower'] = ta.bbands(data['Close'], length=14)
            data.ta.bbands(close=data['Close'], length=14, append=True)
            #data['adx'] = ta.adx(data['High'], data['Low'], data['Close'], length=14)
            data.ta.adx(append=True)
            #data['aroon_up'], data['aroon_down'] = ta.aroon(data['High'], data['Low'], length=14)
            data.ta.aroon(high=data['High'], low=data['Low'], length=14, append=True)
            data['force_index'] = ta.efi(data['Close'], data['Volume'], length=14)
            data.ta.obv(append=True)
            #data['obv'] = ta.obv(data['Close'], data['Volume'])
            #data['obv'] = ta.volume.obv(data['Close'], data['Volume'])
            data['accumulation_distribution'] = ta.ad(data['High'], data['Low'], data['Close'], data['Volume'])
            ##data['mfi'] = (ta.mfi(data['High'], data['Low'], data['Close'], data['Volume'], length=14))
            ##data['vwap'] = ta.vwap(data['High'], data['Low'], data['Close'], data['Volume'])
            #data['klinger'] = ta.kvo(data['High'], data['Low'], data['Close'], data['Volume'])
            data.ta.kvo(append=True)
            data['cmf'] = ta.cmf(data['High'], data['Low'], data['Close'], data['Volume'], length=14)
            ##?accumulation swing index ?##data['asi'] = ta.asi(data['High'], data['Low'], data['Close'], data['Volume'])
            #data['efi'] = ta.efi(data['High'], data['Low'], data['Close'], data['Volume'])
            data.ta.efi(append=True)
            data['chaikin_oscillator'] = ta.cmf(data['High'], data['Low'], data['Close'], data['Volume'])
            data['price_volume_trend'] = ta.pvt(data['Close'], data['Volume'])
            data['pvol'] = ta.pvol(data['High'], data['Low'], data['Close'])
            #data['fibonacci_retracement'] = ta.fwma(data['High'], data['Low'])
            data.ta.fwma(append=True)
            data['atr'] = ta.atr(data['High'], data['Low'], data['Close'], length=14)
            #data['stochastic_rsi'] = ta.stochrsi(data['Close'], length=14)
            data.ta.stochrsi(append=True)
            #data['trix'] = ta.trix(data['Close'], length=14)
            data.ta.trix(append=True)
            #data['parabolic_sar'] = ta.psar(data['High'], data['Low'], data['Close'], af=0.02, max_af=0.2)
            data.ta.psar(append=True)
            data['cci'] = ta.cci(data['High'], data['Low'], data['Close'], length=14)
            #data['donchian_upper'], data['donchian_middle'], data['donchian_lower'] = ta.donchian(data['High'], data['Low'], length=14)
            data.ta.donchian(append=True)
            #data['supertrend'] = ta.supertrend(data['High'], data['Low'], data['Close'], length=14, multiplier=3)
            data.ta.supertrend(append=True)
            #data['ichimoku_cloud'] = ta.ichimoku(data['High'], data['Low'], data['Close'])
            data.ta.ichimoku(lookahead=False, append=True)
            #data['dmi_plus'], data['dmi_minus'], data['adx'] = ta.dmi(data['High'], data['Low'], data['Close'], length=14)
            data.ta.dm(append=True)
            data['rvi'] = ta.rvi(data['Close'], length=14)
            #data['normalized_average_true_range'] = ta.natr(data['Close'], length=14)
            data.ta.natr(append=True)

            # Candlestick patterns # ToDo fix and add the candlestick metrics
            data['doji'] = data.ta.cdl_pattern(name=["doji"])
            data['inside'] = data.ta.cdl_pattern(name=["inside"])
            '''
            data['doji'] = ta.cdl_doji(data)
            data['engulfing'] = ta.cdl_engulfing(data)
            data['hammer'] = ta.cdl_hammer(data)
            data['morning_star'] = ta.cdl_morningstar(data)
            data['evening_star'] = ta.cdl_eveningstar(data)
            data['piercing'] = ta.cdl_piercing(data)
            data['dark_cloud_cover'] = ta.cdl_darkcloudcover(data)
            data['marubozu'] = ta.cdl_marubozu(data)
            data['shooting_star'] = ta.cdl_shootingstar(data)
            data['spinning_top'] = ta.cdl_spinningtop(data)
            data['dragonfly_doji'] = ta.cdl_dragonflydoji(data)
            data['gravestone_doji'] = ta.cdl_gravestonedoji(data)
            data['inside'] = ta.cdl_inside(data)
            data['harami'] = ta.cdl_harami(data)
            data['harami_cross'] = ta.cdl_haramicross(data)
            data['inverted_hammer'] = ta.cdl_invertedhammer(data)
            data['abandoned_baby'] = ta.cdl_abandonedbaby(data)
            data['three_white_soldiers'] = ta.cdl_threewhitesoldiers(data)
            data['three_black_crows'] = ta.cdl_threeblackcrows(data)

            '''


            data.insert(0, 'Datetime', data.index)
            data.insert(0, 'Ticker', ticker)


            db_name = f'{ticker}.db'
            conn = sqlite3.connect(db_name)
            cursor = conn.cursor()


            create_table_sql = f'''
            CREATE TABLE IF NOT EXISTS "{ticker}" (
                {", ".join([f"'{col}' TEXT" for col in data.columns])},
                UNIQUE ("Ticker", "Datetime")
            )
            '''
            if first_pass:
                cursor.execute(create_table_sql)
                first_pass = False
                sleep(0.333)


            data.to_sql(ticker, conn, if_exists='append', index=False)
            sleep(0.333)
            conn.commit()
            sleep(0.333)
            conn.close()
        except Exception as e:
            print(e)
            #current_start_obj = end_date_obj
            #break





def compare_list_to_directory(list_of_names, directory_path=getcwd()):
    directory_contents = listdir(directory_path)
    formated_dir_contents = [cur_db.replace(".db", "").strip() for cur_db in directory_contents]

    missing_items = set(list_of_names) - set(formated_dir_contents)
    extra_items = set(formated_dir_contents) - set(list_of_names)

    return missing_items, extra_items



def add_uniq_to_directory(directory_path=getcwd()):
    directory_contents = listdir(directory_path)
    formatted_dir_contents = [cur_db.strip() for cur_db in directory_contents]
    dbs_to_edit = []
    for cur_file in formatted_dir_contents:
        if cur_file.endswith(".db"):
            dbs_to_edit.append(cur_file)

    #dbs_to_edit = [cur_db if cur_db.endswith(".db") for cur_db in formatted_dir_contents]


    delete_replace_rename_query = \
        '''
        CREATE TABLE "temp_{ticker}" AS
        SELECT DISTINCT column1, column2
        FROM "{ticker}";

        DROP TABLE "{ticker}";

        RENAME TABLE "temp_{ticker}" TO "{ticker}";
        '''

    sql_delay = 0.333
    num_dbs = len(dbs_to_edit)
    db_iter = 0
    for cur_db_to_edit in dbs_to_edit:

        db_iter += 1
        print(f"Working on {cur_db_to_edit} --- {db_iter} / {num_dbs}")

        ticker = cur_db_to_edit.replace(".db", "")
        create_temp_tbl_sql = \
        f'''
        CREATE TABLE "temp_{ticker}" AS
        SELECT DISTINCT "Ticker", "Datetime", *
        FROM "{ticker}";
        '''

        drop_orig_tbl_sql = f'DROP TABLE "{ticker}";'

        replace_orig_tbl_sql = f'RENAME TABLE "temp_{ticker}" TO "{ticker}";'

        constraint_sql = \
        f'''

        ALTER TABLE "{ticker}"
        ADD CONSTRAINT unique_ticker_datetime UNIQUE ("Ticker", "Datetime");
        '''
        conn = None
        try:
            conn = sqlite3.connect(cur_db_to_edit)
            cursor = conn.cursor()

            cursor.execute(create_temp_tbl_sql)
            sleep(sql_delay)
            cursor.execute(drop_orig_tbl_sql)
            sleep(sql_delay)
            cursor.execute(replace_orig_tbl_sql)
            sleep(sql_delay)
            cursor.execute(constraint_sql)

            conn.commit()
        except Exception as e:
            print(e)
        finally:
            if conn is not None:
                conn.close()



    missing_items = set(list_of_names) - set(formated_dir_contents)
    extra_items = set(formated_dir_contents) - set(list_of_names)

    return missing_items, extra_items

if __name__ == '__main__':
    #download_and_store_stock_data(combined_dow30_nasdaq100_sap500_tickers)
    #download_and_store_stock_data(dow30_tickers, db_name="dow30_data.db")
    #download_and_store_stock_data(nasdaq100_tickers, db_name="nasdaq100_data.db")

    #missing_tickers, extra_tickers = compare_list_to_directory(nasdaq_tickers)

    #download_and_store_stock_data_minutes(chip_tickers, db_name="ARM_TMSC_data.db")
    #download_and_store_stock_data_minutes(sap500_tickers, db_name="sap500_data.db")
    ##add_uniq_to_directory()
    #from nasdaq_list_2025 import nasdaq_tickers
    from nyse_list_2025 import nyse_tickers
    download_and_store_stock_data_minutes(nyse_tickers, db_name="")
    print("Data download and storage complete with technical indicators!")


#using yfinance_downloader.py as ex, write a function that will open given database (passed as parameter) and chk for the given tables for latest info by date column, compare to current date, then find- obtain- and update db with any missing data
