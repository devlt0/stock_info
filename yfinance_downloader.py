import yfinance as yf
import sqlite3
import pandas as pd
import pandas_ta as ta

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

def download_and_store_stock_data(tickers, start_date="1900-01-01", end_date="2025-01-01", db_name="sap500_nasdaq100_dow30_data.db"):
    db_name = db_name.replace(".db", f"{start_date}_to_{end_date}.db")
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()


    columns = [
        'ticker', 'date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume', 'sma', 'ema', 'hma', 'tma', 'rma',
        'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'rsi', 'BBL_14_2', 'BBM_14_2', 'BBU_14_2', 'BBB_14_2',
        'BBP_14_2', 'ADX_14', 'DMP_14', 'DMN_14', 'AROOND_14', 'AROONU_14', 'AROONOSC_14', 'force_index', 'OBV',
        'accumulation_distribution', 'mfi', 'vwap', 'KVO_34_55_13', 'KVOs_34_55_13', 'cmf', 'EFI_13', 'chaikin_oscillator',
        'price_volume_trend', 'pvol', 'FWMA_10', 'atr', 'STOCHRSIk_14_14_3_3', 'STOCHRSId_14_14_3_3', 'TRIX_30_9', 'TRIXs_30_9',
        'PSARl_0_02_0_2', 'PSARs_0_02_0_2', 'PSARaf_0_02_0_2', 'PSARr_0_02_0_2', 'cci', 'donchian_upper', 'donchian_middle',
        'donchian_lower', 'SUPERT_7_3', 'SUPERTd_7_3', 'SUPERTl_7_3', 'SUPERTs_7_3', 'ISA_9', 'ISB_26', 'ITS_9',
        'IKS_26', 'rvi', 'NATR_14', 'sharpe_ratio', 'sortino_ratio', 'max_drawdown'#, 'pure_profit_score'
    ]
    ticker_iter = 1
    num_tickers = len(tickers)
    tickers_processed = cursor.execute(
            f"""SELECT name FROM sqlite_master WHERE type='table'; """).fetchall()
    tickers_processed = [x[0] for x in tickers_processed]
    #AND name='{ticker}'; """).fetchall()

    for ticker in tickers:
        if ticker in tickers_processed:
            print(f"Already processed & stored data for {ticker}... {ticker_iter} out of {num_tickers}")
            continue # already processed

        print(f"Downloading data for {ticker}... {ticker_iter} out of {num_tickers}")
        ticker_iter += 1


        data = yf.download(ticker, start=start_date, end=end_date)
        #data['ticker'] = ticker
        #data["Date"] = data.index
        data = data.xs(ticker, level=1, axis=1)


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
        data['mfi'] = (ta.mfi(data['High'], data['Low'], data['Close'], data['Volume'], length=14)).astype('float64')
        data['vwap'] = ta.vwap(data['High'], data['Low'], data['Close'], data['Volume'])
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
        ##
        data.ta.cdl_doji(appened=True)
        data.ta.cdl_engulfing(appened=True)
        data.ta.cdl_hammer(appened=True)
        data.ta.cdl_morningstar(appened=True)
        data.ta.cdl_eveningstar(appened=True)
        data.ta.cdl_piercing(appened=True)
        data.ta.cdl_darkcloudcover(appened=True)
        data.ta.cdl_marubozu(appened=True)
        data.ta.cdl_shootingstar(appened=True)
        data.ta.cdl_spinningtop(appened=True)
        data.ta.cdl_dragonflydoji(appened=True)
        data.ta.cdl_gravestonedoji(appened=True)
        data.ta.cdl_inside(appened=True)
        data.ta.cdl_harami(appened=True)
        data.ta.cdl_haramicross(appened=True)
        data.ta.cdl_invertedhammer(appened=True)
        data.ta.cdl_abandonedbaby(appened=True)
        data.ta.cdl_threewhitesoldiers(appened=True)
        data.ta.cdl_threeblackcrows(appened=True)
        '''
        # Performance Metrics (Sharpe Ratio, Sortino Ratio, etc.)
        data['sharpe_ratio'] = ta.sharpe_ratio(data['Close'])
        data['sortino_ratio'] = ta.sortino_ratio(data['Close'])
        data['max_drawdown'] = ta.max_drawdown(data['Close'])


        data.insert(0, 'Date', data.index)
        data.insert(0, 'ticker', ticker)



        create_table_sql = f'''
        CREATE TABLE IF NOT EXISTS "{ticker}" (
            {", ".join([f"{col} TEXT" for col in columns])}
        )
        '''
        cursor.execute(create_table_sql)


        data.to_sql(ticker, conn, if_exists='replace', index=False)

        conn.commit()


    conn.close()


if __name__ == '__main__':
    #download_and_store_stock_data(combined_dow30_nasdaq100_sap500_tickers)
    download_and_store_stock_data(dow30_tickers, db_name="dow30_data.db")
    #download_and_store_stock_data(nasdaq100_tickers, db_name="nasdaq100_data.db")
    #download_and_store_stock_data(sap500_tickers, db_name="sap500_data.db")

    print("Data download and storage complete with technical indicators!")
