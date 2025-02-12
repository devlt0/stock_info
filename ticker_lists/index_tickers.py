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