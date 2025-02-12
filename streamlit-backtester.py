#import dtale # not meant for embedding...
from finvizfinance.quote import finvizfinance
import numpy as np
import pandas as pd
import pandas_ta as ta
#from pandasgui import show # more for standalone then web unless some hidden setting not found like pygwalker
import plotly.express as px
import plotly.graph_objects as go
from pygooglenews import GoogleNews
from pymongo import MongoClient
from pygwalker.api.streamlit import StreamlitRenderer
import streamlit as st
import yfinance as yf


import math
import sqlite3

from datetime import datetime, timedelta
from dateutil import parser
from os import getcwd, getenv, listdir






st.set_page_config( layout="wide") #page_title="Stock Strategy Backtester", layout="wide")
st.header("Stock Strategy Backtester")

st.sidebar.header("Strategy Parameters")


#db_files = ['large_db/dow30_data1900-01-01_to_2025-01-01.db',
#            'large_db/nasdaq100_data1900-01-01_to_2025-01-01.db',
#            'large_db/ARM_TMSC_data_2024-12-10_to_2025-01-08_minute.db',]  #["nasdaq100_data.db", "dow30_data.db"]   # "sap500_data.db",

nyse_tickers = listdir('./shards/nyse')
#print(f'nyse - \n{nyse_tickers}')

nasdaq_tickers = listdir('./shards/nasdaq')
#print(f'nasdaq - \n{nasdaq_tickers}')

db_files = []
db_files.extend(nyse_tickers)
db_files.extend(nasdaq_tickers)
db_files.sort()
#print(f'db-files:  \n{db_files}')

selected_db = st.sidebar.selectbox("Select Database / Ticker", db_files)
# given that even stocks with 60yr+ history are <10mb per shard may not need to worry about splitting on time
#  if / when start splitting on time selection needs to change to be based on ticker and add helper utils for pulling data to query/display

# Datetime - minutes
# Date - dailies

@st.cache_resource
def get_available_tickers(shard_name='AAPL.db'):
    #print(shard_name)
    db_name = shard_name
    db_path = './shards/'
    if shard_name in nyse_tickers:
        db_path += 'nyse/'
    elif shard_name in nasdaq_tickers:
        db_path += 'nasdaq/'
    else:
        pass
        # maybe ST error/warning
    db_name = db_path + db_name
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    conn.close()
    return [table[0] for table in tables]


@st.cache_resource
def get_date_range(db_name, ticker):
    db_path = './shards/'
    if db_name in nyse_tickers:
        db_path += 'nyse/'
    elif db_name in nasdaq_tickers:
        db_path += 'nasdaq/'
    else:
        pass
    db_path += db_name
    #print(f'db path {db_path}')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = f"SELECT MIN(date), MAX(date) FROM '{ticker}'"
    cursor.execute(query)
    min_date, max_date = cursor.fetchone()

    conn.close()

    # Convert dates from string to datetime

    min_date = datetime.strptime(min_date.split()[0], "%Y-%m-%d") if min_date else datetime.now() - timedelta(days=365)
    #datetime.strptime(min_date, "%Y-%m-%d") if min_date else datetime.now() - timedelta(days=365)
    max_date = datetime.strptime(max_date.split()[0], "%Y-%m-%d") if max_date else datetime.now()
    #datetime.strptime(max_date, "%Y-%m-%d") if max_date else datetime.now()

    return min_date, max_date




available_tickers = get_available_tickers(selected_db)
selected_ticker = available_tickers[0] if len(available_tickers) > 0 and type(available_tickers) == list else None #st.sidebar.selectbox("Select Ticker", available_tickers)
# given change to shard by ticker, base db name and table name are just ticker



days_in_yr = 365
trading_days_in_yr = 252   # 365 - 52*2 (weekends) - 9 holidays = 365 - 104 - 9 = 365 - 113
min_date, max_date = get_date_range(selected_db, selected_ticker)
#start_date = st.sidebar.date_input("Start Date", max_date - timedelta(days=days_in_yr), min_value=min_date, max_value=max_date)
# lot to be said for choosing last yr of data but...
#  seemingly more intuitive to show all available data and letter user choose

try:
    start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=min(max_date, end_date.date()))
except Exception as e:
    start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)

end_date = st.sidebar.date_input("End Date", max_date, min_value=max(min_date.date(), start_date), max_value=max_date)
# need button or dynamic dependency


#start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
#end_date = st.sidebar.date_input("End Date", datetime.now())


strategy_type = st.sidebar.selectbox(
    "Select Strategy Type",
    [
        "Moving Average Crossover", "RSI", "MACD", "Bollinger Bands", "SuperTrend", "Ichimoku Cloud",
        #"Parabolic SAR", "Aroon Indicator", "Chaikin Oscillator", "Force Index", "Money Flow Index (MFI)",
        #"Commodity Channel Index (CCI)"#, "Exponential Moving Average (EMA) Crossover"
        "Custom Strategy"
        # disabled last options til can get working
    ]
)

# Strategy specific parameters with default values as globals, ugly af but this is quick and dirty for time being
short_winfow = 20
long_window = 50
oversold = 30
overbought = 70
acceleration_facotr = 0.02
max_af = 0.2
aroon_period=14
chaikin_short_window = 10
chaikin_long_window = 30
force_index_window = 13
mfi_period = 14
mfi_oversold =20
mfi_overbought = 80

COMPARISON_OPERATORS = {
    "greater than": ">",
    "less than": "<",
    "equals": "==",
    "greater than or equals": ">=",
    "less than or equals": "<="
}

REFERENCE_TYPES = {
    "value": "fixed value",
    "indicator": "another indicator",
    "price": "price data",
    "moving average": "moving average of self"
}

def get_available_indicators(df):
    """Dynamically get all available indicators from dataframe columns."""
    # Exclude common non-indicator columns
    exclude_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Datetime', 'ticker']
    indicators = [col for col in df.columns if col not in exclude_columns]
    return sorted(indicators)

def create_indicator_condition():
    """Create a new default condition structure"""
    return {
        "indicator": None,
        "comparison": "greater than",
        "reference_type": "value",
        "reference_value": 0,
        "reference_indicator": None,
        "ma_period": 0,
        "logic": "AND"
    }


@st.cache_data
def load_data(ticker, db_name):
    db_path = './shards/'
    if db_name in nyse_tickers:
        db_path += 'nyse/'
    elif db_name in nasdaq_tickers:
        db_path += 'nasdaq/'
    else:
        pass
    db_path += db_name
    conn = sqlite3.connect(db_path)
    query = f'SELECT * FROM "{ticker}"'
    df = pd.read_sql_query(query, conn)
    conn.close()
    #print("columns pre mixup")
    #print(df.columns)
    #print("=="*22)
    df['Date'] = df['Date'].str.strip()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    #print(f"num NaT; {df['Date'].isna().sum()}")

    #new_columns = {col: col.replace('.', '_') for col in df.columns if '.' in col}
    #df = df.rename(columns=new_columns)
    #numeric_columns = df.columns.drop(['ticker', 'Date'])
    #df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    return df


def evaluate_condition(df, condition):
    """Dynamically evaluate a single condition"""
    # Get base indicator values
    try:
        indicator_values = pd.to_numeric(df[condition["indicator"]])
    except:
        st.error(f"Error converting indicator {condition['indicator']} to numeric values")
        return pd.Series(False, index=df.index)

    # Get reference values based on type
    if condition["reference_type"] == "value":
        reference_values = condition["reference_value"]
    elif condition["reference_type"] == "indicator":
        reference_values = pd.to_numeric(df[condition["reference_indicator"]])
    elif condition["reference_type"] == "price":
        reference_values = pd.to_numeric(df[condition["reference_value"]])
    elif condition["reference_type"] == "moving average":
        reference_values = indicator_values.rolling(window=condition["ma_period"]).mean()

    # Create comparison string and evaluate
    operator = COMPARISON_OPERATORS[condition["comparison"]]
    return eval(f"indicator_values {operator} reference_values")



def evaluate_custom_strategy(df, conditions):
    """Evaluate all conditions and combine them according to their logic"""
    if not conditions:
        return pd.DataFrame(0, index=df.index, columns=['position'])

    signals = pd.DataFrame(index=df.index)
    signals['position'] = 0

    combined_condition = None

    for i, condition in enumerate(conditions):
        current_condition = evaluate_condition(df, condition)

        if combined_condition is None:
            combined_condition = current_condition
        else:
            if condition["logic"] == "AND":
                combined_condition = combined_condition & current_condition
            else:  # OR
                combined_condition = combined_condition | current_condition

    signals['position'] = combined_condition.astype(int)
    return signals


def implement_strategy(df, strategy_type):
    signals = pd.DataFrame(index=df.index)
    signals['position'] = 0

    if strategy_type == "Moving Average Crossover":
        indicator = ""
        if ma_type == "SMA":
            indicator = 'SMA'
        elif ma_type == "EMA":
            indicator = 'EMA'
        elif ma_type == "HMA":
            indicator = 'HMA'
        elif ma_type == "TMA":
            indicator = 'TEMA'
        elif ma_type == "RMA":
            indicator = 'RMA'
        if indicator:
            indicator += "_10"

        if len(indicator) > 0:
            df['MA_short'] = pd.to_numeric(df[indicator].shift(short_window), errors='coerce')
            df['MA_long'] = pd.to_numeric(df[indicator].shift(long_window), errors='coerce')

            signals['position'] = np.where(df['MA_short'] > df['MA_long'], 1, 0)
        else:
            raise Exception(f"Unable to find associated indicator for {ma_type} in Moving Avg Crossover")

    elif strategy_type == "RSI":
        signals['position'] = np.where(pd.to_numeric(df['RSI_14']) < oversold, 1,
                                     np.where(pd.to_numeric(df['RSI_14']) > overbought, 0,
                                     signals['position'].shift(1)))

    elif strategy_type == "MACD":
        if signal_crossover:
            signals['position'] = np.where(
                pd.to_numeric(df['MACD_12_26_9']) > pd.to_numeric(df['MACDs_12_26_9']), 1, 0)
        else:
            signals['position'] = np.where(pd.to_numeric(df['MACD_12_26_9']) > 0, 1, 0)

    elif strategy_type == "Bollinger Bands":
        if bb_strategy == "Mean Reversion":
            signals['position'] = np.where(pd.to_numeric(df['Close']) < pd.to_numeric(df['BBL_5_2.0']), 1,
                                         np.where(pd.to_numeric(df['Close']) > pd.to_numeric(df['BBU_5_2.0']), 0,
                                         signals['position'].shift(1)))
        else:  # Breakout
            signals['position'] = np.where(pd.to_numeric(df['Close']) > pd.to_numeric(df['BBU_5_2.0']), 1,
                                         np.where(pd.to_numeric(df['Close']) < pd.to_numeric(df['BBL_5_2.0']), 0,
                                         signals['position'].shift(1)))

    elif strategy_type == "SuperTrend":
        signals['position'] = np.where(
            (pd.to_numeric(df['SUPERTd_7_3.0']) == 1) &
            (pd.to_numeric(df['ADX_14']) > trend_strength), 1, 0)

    elif strategy_type == "Ichimoku Cloud":
        if cloud_strategy == "Cloud Breakout":
            signals['position'] = np.where(
                (pd.to_numeric(df['Close']) > pd.to_numeric(df['ISA_9'])) &
                (pd.to_numeric(df['Close']) > pd.to_numeric(df['ISB_26'])), 1, 0)
        else:  # TK Cross
            signals['position'] = np.where(
                pd.to_numeric(df['ITS_9']) > pd.to_numeric(df['IKS_26']), 1, 0)

    elif strategy_type == "Parabolic SAR":
        df['PSAR'] = pd.to_numeric(df['PSARs_0.02_0.2'], errors='coerce')
        signals['position'] = np.where(df['Close'] > df['PSAR'], 1, 0)

    elif strategy_type == "Aroon Indicator":
        df['AROONU'] = pd.to_numeric(df['AROONU_14'], errors='coerce')
        df['AROOND'] = pd.to_numeric(df['AROOND_14'], errors='coerce')
        signals['position'] = np.where(df['AROONU'] > df['AROOND'], 1, 0)

    elif strategy_type == "Chaikin Oscillator":
        df['chaikin_oscillator'] = pd.to_numeric(df['chaikin_oscillator'], errors='coerce')
        signals['position'] = np.where(df['chaikin_oscillator'] > 0, 1, 0)

    elif strategy_type == "Force Index":
        df['force_index'] = pd.to_numeric(df['force_index'], errors='coerce')
        signals['position'] = np.where(df['force_index'] > 0, 1, 0)

    elif strategy_type == "Money Flow Index (MFI)":
        df['mfi'] = pd.to_numeric(df['mfi'], errors='coerce')
        signals['position'] = np.where(df['mfi'] < 20, 1, np.where(df['mfi'] > 80, 0, signals['position'].shift(1)))

    elif strategy_type == "Commodity Channel Index (CCI)":
        df['cci'] = pd.to_numeric(df['cci'], errors='coerce')
        signals['position'] = np.where(df['cci'] < -100, 1, np.where(df['cci'] > 100, 0, signals['position'].shift(1)))
    elif strategy_type == "Custom Strategy":
        if not st.session_state.conditions:
            st.warning("Please add at least one condition to your custom strategy")
            return signals
        signals = evaluate_custom_strategy(df, st.session_state.conditions)

    signals['position'] = signals['position'].fillna(0)
    return df, signals


#ToDo uncouple globals and pass them as parameters
#ToDo add global labels for repeatedly used cols
def calculate_performance(df, signals, initial_investment):
    slippage_factor = 1 - slippage

    positions = pd.DataFrame(index=signals.index)
    positions['position'] = signals['position']

    # calculate time unit returns // either daily or minutes depending on db
    df['returns'] = pd.to_numeric(df['Close']).pct_change()

    positions['raw_strategy_returns'] = positions['position'].shift(1) * df['returns'] ##* slippage_factor
    positions['raw_strategy_returns'] = positions['raw_strategy_returns'].fillna(0)
    positions['cumulative_returns'] = (1 + positions['raw_strategy_returns']).cumprod()
    positions['raw_portfolio_value'] = initial_investment * positions['cumulative_returns']
    positions['position_change'] = positions['position'].diff()
    positions['position_change'] = positions['position_change'].fillna(0)
    positions['trade_value'] = abs(positions['position_change']) * df['Close'] # issue here doesn't acount for how many shares/stocks
    #positions['slippage_loss'] = positions['trade_value'] * slippage
    #positions['cumulative_slippage_loss'] = positions['slippage_loss'].cumsum()
    #total_slippage_loss = positions['cumulative_slippage_loss'].iloc[-1]


    if fee_type == 'Per Transaction':
        # Apply fees only when there's a trade (position_change != 0)
        positions['transaction_fees'] = np.where(
            positions['position_change'] != 0,
            flat_fee + (positions['raw_portfolio_value'] * perc_fee), # naive, presumes trading everything each time
            0
        )

    #elif fee_type == 'Per Share':
    #    positions['transaction_fees'] = np.where(
    #        positions['position_change'] != 0,
    #        (flat_fee * abs(positions['position_change'])) + (positions['trade_value'] * perc_fee),
    #        0
    #    )

    positions['cumulative_fees']  = positions['transaction_fees'].cumsum()
    positions['slippage_loss']    = positions['raw_portfolio_value'] * slippage
    positions['portfolio_value']  = positions['raw_portfolio_value'] - positions['cumulative_fees'] - positions['slippage_loss']
    positions['strategy_returns'] = positions['portfolio_value'].pct_change()
    positions['strategy_returns'] = positions['strategy_returns'].fillna(0)

    ##positions['strategy_returns_after_fees'] = positions['strategy_returns'] - (positions['transaction_fees'] / (positions['trade_value'].shift(1) + initial_investment))
    ##positions['cumulative_returns'] = (1 + positions['strategy_returns_after_fees']).cumprod()
    positions['cumulative_market_returns'] = (1 + df['returns']).cumprod() # presume no sale
    # would it be meaningful to add faux buy at start and sell at end fees so market return is more apple to apple?

    #positions['cumulative_returns'] = (1 + positions['strategy_returns']).cumprod()
    #positions['cumulative_market_returns'] = (1 + df['returns']).cumprod()

    #positions['portfolio_value'] = initial_investment * positions['cumulative_returns']
    positions['market_value'] = initial_investment * positions['cumulative_market_returns']

    # dbl chk sharpe & sortino ratios use strategy returns and not cumulative returns or something else
    total_return = (positions['portfolio_value'].iloc[-1] - initial_investment) / initial_investment * 100
    market_return = (positions['market_value'].iloc[-1] - initial_investment) / initial_investment * 100
    sharpe_ratio = np.sqrt(252) * positions['strategy_returns'].mean() / positions['strategy_returns'].std()
    max_drawdown = (positions['portfolio_value'] / positions['portfolio_value'].cummax() - 1).min() * 100


    downside_returns = positions['strategy_returns'][positions['strategy_returns'] < 0]
    downside_deviation = downside_returns.std()
    sortino_ratio = np.sqrt(252) * positions['strategy_returns'].mean() / downside_deviation if downside_deviation != 0 else 0

    total_fees = positions['transaction_fees'].sum() #positions['cumulative_fees'].iloc[-1]
    pure_profit = positions['portfolio_value'].iloc[-1] - initial_investment #- total_fees
    total_slippage_loss = positions['slippage_loss'].iloc[-1] #total_return * slippage
    place_holder = float('nan')
    #positions.to_csv(f'{selected_ticker}.csv')

    return positions, total_return, market_return, sharpe_ratio, max_drawdown, sortino_ratio, pure_profit, total_fees, total_slippage_loss



def plot_parabolic_sar(df):
    #af=0.02, max_af=0.2)
    data.ta.psar(append=True, af=acceleration_factor, max_af=max_af)
    ##df['SAR'] = ta.SAR(df['high'], df['low'], acceleration=0.02, maximum=0.2)
    #df['SAR'] = df.ta.psar(af=0.02, max_af=0.2)


    fig = go.Figure()

    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'))

    #fig.add_trace(go.Scatter(x=df.index, y=df['SAR'], mode='markers', name='Parabolic SAR', marker=dict(color='red', size=5))) PSARl_0_02_0_2
    #fig.add_trace(go.Scatter(x=df.index, y=df['PSARl_0_02_0_2'], mode='markers', name='Parabolic SAR long', marker=dict(color='red', size=5)))
    #fig.add_trace(go.Scatter(x=df.index, y=df['PSARs_0_02_0_2'], mode='markers', name='Parabolic SAR short', marker=dict(color='pink', size=5)))
    fig.add_trace(go.Scatter(x=df.index, y=df['PSAR'], mode='markers', name='Parabolic SAR', marker=dict(color='red', size=5))) # PSARl_0_02_0_2

    fig.update_layout(title="Parabolic SAR", xaxis_title='Date', yaxis_title='Price', template="plotly_dark")

    #st.plotly_chart(fig)
    return fig

def plot_aroon(df, period=aroon_period):
    df['AroonUp'], df['AroonDown'] = ta.AROON(df['high'], df['low'], timeperiod=period)


    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['AroonUp'], mode='lines', name='Aroon Up', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=df.index, y=df['AroonDown'], mode='lines', name='Aroon Down', line=dict(color='red')))

    fig.add_trace(go.Scatter(x=df.index, y=[70]*len(df), mode='lines', name='Overbought', line=dict(dash='dash', color='gray')))
    fig.add_trace(go.Scatter(x=df.index, y=[30]*len(df), mode='lines', name='Oversold', line=dict(dash='dash', color='gray')))

    fig.update_layout(title="Aroon Indicator", xaxis_title='Date', yaxis_title='Aroon Value', template="plotly_dark")

    #st.plotly_chart(fig)
    return fig

def plot_chaikin_oscillator(df, short_period=chaikin_short_window, long_period=chaikin_long_window):
    df['Chaikin'] = ta.AD(df['high'], df['low'], df['close'], df['volume'])
    df['ChaikinOscillator'] = ta.EMA(df['Chaikin'], timeperiod=short_period) - ta.EMA(df['Chaikin'], timeperiod=long_period)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df.index, y=df['ChaikinOscillator'], mode='lines', name='Chaikin Oscillator', line=dict(color='purple')))
    fig.add_trace(go.Scatter(x=df.index, y=[0]*len(df), mode='lines', name='Zero Line', line=dict(dash='dash', color='gray')))
    fig.update_layout(title="Chaikin Oscillator", xaxis_title='Date', yaxis_title='Oscillator Value', template="plotly_dark")

    #st.plotly_chart(fig)
    return fig

def plot_force_index(df, period=force_index_window):
    df['ForceIndex'] = ta.EMA(df['close'] - df['close'].shift(1), timeperiod=period) * df['volume']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['ForceIndex'], mode='lines', name='Force Index', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df.index, y=[0]*len(df), mode='lines', name='Zero Line', line=dict(dash='dash', color='gray')))
    fig.update_layout(title="Force Index", xaxis_title='Date', yaxis_title='Force Index Value', template="plotly_dark")
    #st.plotly_chart(fig)
    return fig

def plot_mfi(df, period=mfi_period, oversold=mfi_oversold, overbought=mfi_overbought):
    df['MFI'] = ta.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=period)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['MFI'], mode='lines', name='Money Flow Index', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=[oversold]*len(df), mode='lines', name='Oversold', line=dict(dash='dash', color='red')))
    fig.add_trace(go.Scatter(x=df.index, y=[overbought]*len(df), mode='lines', name='Overbought', line=dict(dash='dash', color='green')))
    fig.update_layout(title="Money Flow Index", xaxis_title='Date', yaxis_title='MFI Value', template="plotly_dark")
    return fig
    #st.plotly_chart(fig)






#st.title("Stock Strategy Backtester")
df = load_data(selected_ticker, selected_db)

if strategy_type == "Moving Average Crossover":
    ma_type = st.sidebar.selectbox("MA Type", ["SMA", "EMA", "HMA", "TMA", "RMA"])
    short_window = st.sidebar.slider("Short Window", 5, 50, 20)
    long_window = st.sidebar.slider("Long Window", 20, 200, 50)
elif strategy_type == "RSI":
    oversold = st.sidebar.slider("Oversold Level", 20, 40, 30)
    overbought = st.sidebar.slider("Overbought Level", 60, 80, 70)
elif strategy_type == "MACD":
    signal_crossover = st.sidebar.checkbox("Use Signal Line Crossover", value=True)
elif strategy_type == "Bollinger Bands":
    bb_strategy = st.sidebar.selectbox("Strategy Type", ["Mean Reversion", "Breakout"])
elif strategy_type == "SuperTrend":
    trend_strength = st.sidebar.slider("Minimum ADX for Trend", 15, 40, 25)
elif strategy_type == "Ichimoku Cloud":
    cloud_strategy = st.sidebar.selectbox("Strategy Type", ["Cloud Breakout", "TK Cross"])
elif strategy_type == "Parabolic SAR":
    acceleration_factor = st.sidebar.slider("Acceleration Factor", 0.01, 0.1, 0.02)
    max_af = st.sidebar.slider("Maximum Acceleration Factor", 0.1, 1.0, 0.2)
elif strategy_type == "Aroon Indicator":
    aroon_period = st.sidebar.slider("Aroon Period", 10, 50, 14)
elif strategy_type == "Chaikin Oscillator":
    chaikin_short_window = st.sidebar.slider("Chaikin Short Window", 5, 30, 10)
    chaikin_long_window = st.sidebar.slider("Chaikin Long Window", 30, 100, 30)
elif strategy_type == "Force Index":
    force_index_window = st.sidebar.slider("Force Index Window", 5, 30, 13)
elif strategy_type == "Money Flow Index (MFI)":
    mfi_period = st.sidebar.slider("MFI Period", 10, 50, 14)
    mfi_oversold = st.sidebar.slider("MFI Oversold Level", 10, 30, 20)
    mfi_overbought = st.sidebar.slider("MFI Overbought Level", 70, 90, 80)
elif strategy_type == "Commodity Channel Index (CCI)":
    cci_period = st.sidebar.slider("CCI Period", 10, 50, 20)
    cci_overbought = st.sidebar.slider("CCI Overbought Level", 100, 200, 100)
    cci_oversold = st.sidebar.slider("CCI Oversold Level", -100, -200, -100)
elif strategy_type == "Custom Strategy":
    st.sidebar.markdown("### Custom Strategy Builder")

    # Initialize session state
    if 'conditions' not in st.session_state:
        st.session_state.conditions = []

    # Get available indicators
    available_indicators = get_available_indicators(df)

    # Add new condition button
    if st.sidebar.button("Add Condition"):
        st.session_state.conditions.append(create_indicator_condition())

    # Display and edit conditions
    for i, condition in enumerate(st.session_state.conditions):
        st.sidebar.markdown(f"#### Condition {i+1}")

        # Select indicator
        condition["indicator"] = st.sidebar.selectbox(
            "Indicator",
            available_indicators,
            key=f"indicator_{i}"
        )

        # Select comparison operator
        condition["comparison"] = st.sidebar.selectbox(
            "Comparison",
            list(COMPARISON_OPERATORS.keys()),
            key=f"comparison_{i}"
        )

        # Select reference type
        condition["reference_type"] = st.sidebar.selectbox(
            "Compare Against",
            list(REFERENCE_TYPES.keys()),
            key=f"reference_type_{i}"
        )

        # Dynamic reference value input based on type
        if condition["reference_type"] == "value":
            condition["reference_value"] = st.sidebar.number_input(
                "Value",
                value=0.0,
                key=f"reference_value_{i}"
            )
        elif condition["reference_type"] == "indicator":
            condition["reference_indicator"] = st.sidebar.selectbox(
                "Reference Indicator",
                available_indicators,
                key=f"reference_indicator_{i}"
            )
        elif condition["reference_type"] == "price":
            condition["reference_value"] = st.sidebar.selectbox(
                "Price Type",
                ["Close", "Open", "High", "Low"],
                key=f"price_type_{i}"
            )
        elif condition["reference_type"] == "moving average":
            condition["ma_period"] = st.sidebar.number_input(
                "MA Period",
                min_value=1,
                value=20,
                key=f"ma_period_{i}"
            )

        # Logic connector (AND/OR)
        if i < len(st.session_state.conditions) - 1:
            condition["logic"] = st.sidebar.selectbox(
                "Logic",
                ["AND", "OR"],
                key=f"logic_{i}"
            )

        # Remove condition button
        if st.sidebar.button("Remove", key=f"remove_{i}"):
            st.session_state.conditions.pop(i)
            #st.experimental_rerun()



initial_investment = st.sidebar.number_input("Initial Investment ($)", value=10000.0, step=1000.0)
#raw_slippage = st.sidebar.number_input("Slippage Percentage", value=1.0, step=1.0)
slippage = st.sidebar.number_input("Slippage Percentage", value=0.0, step=1.0)/100.0
#slippage = raw_slippage/100

fee_type = st.sidebar.selectbox(
    "Select Fee Type",
    ("Per Transaction"),
    #("Per Transaction", "Per Share"),
    help="Choose whether the fee should be applied per share or per transaction"
)

per_suffix = ""
#flat_fee   = 0
#if fee_type == 'Per Share':
#    flat_fee = st.sidebar.number_input("Flat Trade Fee ($) Per Share", value=0.25, step=0.25)
#    per_suffix = "Share"
#elif fee_type == 'Per Transaction':
flat_fee = st.sidebar.number_input("Flat Trade Fee ($) Per Transaction", value=10, step=5)
per_suffix = "Transaction"

raw_perc_fee = st.sidebar.number_input(f"Percent Trade Fee (%) Per {per_suffix}", value=0.0, step=1.0)
perc_fee = raw_perc_fee / 100.0






# ToDo standardize how date is handled b/t daily data and minute data
try:
    df = df[(df['Date'] >= pd.Timestamp(start_date)) & (df['Date'] <= pd.Timestamp(end_date))]
except Exception as e1:
    print(e1)
    try:
        df = df[(df['Date'].tz_convert(None).date() >= start_date) & (df['Date'].tz_convert(None).date() <= end_date)]
    except Exception as e2:
        print(e2)
df.set_index('Date', inplace=True)
df['Date'] = df.index


df, signals = implement_strategy(df, strategy_type)
positions, total_return, market_return, sharpe_ratio, max_drawdown, sortino_ratio, pure_profit, total_fees, total_slippage_loss = calculate_performance(df, signals, initial_investment)











#tab1, tab2, tab3, tab4, tab5 = st.tabs(["Dashboard", "Analysis", "Backtest Results", "Reports", "Settings"])
main_tab, chart_tab, info_tab, gnews_tab, fvf_tab, data_ex_tab, raw_data_tab = \
 st.tabs(['💰 Main', '📈 Charts', '📄 Info', '📰 Google News', '💸 FinViz Insider Transaction & News', '⚒️ Raw Data Viewer', '🔍 Chart Builder'])




market_comp_col, ratio_col, fee_slip_col, profit_drwdwn_col  = main_tab.columns(4)
market_comp_col.metric("Strategy Return", f"{total_return:.2f}%")
market_comp_col.metric("Market Return", f"{market_return:.2f}%")
ratio_col.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
ratio_col.metric("Sortino Ratio", f"{sortino_ratio:.2f}")
profit_drwdwn_col.metric("Max Drawdown", f"{max_drawdown:.2f}%")
profit_drwdwn_col.metric("Pure Profit", f"${pure_profit:.2f}")
fee_slip_col.metric("Total Transaction Fees", f"${total_fees:.2f}")
fee_slip_col.metric("Total Slippage Loss", f"${total_slippage_loss:.2f}")



pv_fig = go.Figure()
pv_fig.add_trace(go.Scatter(x=positions.index, y=positions['portfolio_value'],
                        mode='lines', name='Strategy', line=dict(color='blue')))
pv_fig.add_trace(go.Scatter(x=positions.index, y=positions['market_value'],
                        mode='lines', name='Buy & Hold', line=dict(color='gray')))
pv_fig.update_layout(title='Portfolio Value Over Time',
                 xaxis_title='Date',
                 yaxis_title='Portfolio Value ($)',
                 hovermode='x unified')
#st.plotly_chart(fig, use_container_width=True)

#top_row_left, top_row_right = st.columns([2, 2])  # Top row with two equal columns
#bottom_row_left, bottom_row_right = st.columns([2, 2])  # Bottom row with two equal columns

#top_row_left.plotly_chart(pv_fig, use_container_width=True)

chart_tab.plotly_chart(pv_fig, use_container_width=True, key=777)

strat_fig = go.Figure()
if strategy_type == "Moving Average Crossover":
    strat_fig.add_trace(go.Scatter(x=df.index, y=pd.to_numeric(df['Close']), mode='lines', name='Close Price'))
    strat_fig.add_trace(go.Scatter(x=df.index, y=df['MA_short'], mode='lines', name=f'{ma_type} Short'))
    strat_fig.add_trace(go.Scatter(x=df.index, y=df['MA_long'], mode='lines', name=f'{ma_type} Long'))
    strat_fig.update_layout(title=f'{ma_type} Crossover',
                     xaxis_title='Date',
                     yaxis_title='Price',
                     hovermode='x unified')
    #st.plotly_chart(fig, use_container_width=True)

elif strategy_type == "RSI":

    strat_fig.add_trace(go.Scatter(x=df.index, y=pd.to_numeric(df['RSI_14']), mode='lines', name='RSI'))
    strat_fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1], y0=oversold, y1=oversold,
                 line=dict(color="green", width=2, dash="dash"))
    strat_fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1], y0=overbought, y1=overbought,
                 line=dict(color="red", width=2, dash="dash"))
    strat_fig.update_layout(title='Relative Strength Index',
                     xaxis_title='Date',
                     yaxis_title='RSI',
                     hovermode='x unified')
    #st.plotly_chart(fig, use_container_width=True)

elif strategy_type == "MACD":
    strat_fig.add_trace(go.Scatter(x=df.index, y=pd.to_numeric(df['MACD_12_26_9']), mode='lines', name='MACD'))
    strat_fig.add_trace(go.Scatter(x=df.index, y=pd.to_numeric(df['MACDs_12_26_9']), mode='lines', name='Signal'))
    strat_fig.add_trace(go.Bar(x=df.index, y=pd.to_numeric(df['MACDh_12_26_9']), name='Histogram'))
    strat_fig.update_layout(title='MACD',
                     xaxis_title='Date',
                     yaxis_title='Value',
                     hovermode='x unified')
    #st.plotly_chart(fig, use_container_width=True)

elif strategy_type == "Bollinger Bands":
    strat_fig.add_trace(go.Scatter(x=df.index, y=pd.to_numeric(df['Close']), mode='lines', name='Close Price'))
    strat_fig.add_trace(go.Scatter(x=df.index, y=pd.to_numeric(df['BBU_5_2.0']), mode='lines', name='Upper Band'))
    strat_fig.add_trace(go.Scatter(x=df.index, y=pd.to_numeric(df['BBL_5_2.0']), mode='lines', name='Lower Band'))
    strat_fig.add_trace(go.Scatter(x=df.index, y=pd.to_numeric(df['BBM_5_2.0']), mode='lines', name='Middle Band'))
    strat_fig.update_layout(title='Bollinger Bands',
                     xaxis_title='Date',
                     yaxis_title='Price',
                     hovermode='x unified')
    #st.plotly_chart(fig, use_container_width=True)

elif strategy_type == "SuperTrend":
    strat_fig.add_trace(go.Scatter(x=df.index, y=pd.to_numeric(df['Close']), mode='lines', name='Close Price'))
    strat_fig.add_trace(go.Scatter(x=df.index, y=pd.to_numeric(df['SUPERT_7_3.0']), mode='lines', name='SuperTrend'))
    strat_fig.add_trace(go.Scatter(x=df.index, y=pd.to_numeric(df['ADX_14']), mode='lines', name='ADX'))
    strat_fig.update_layout(title='SuperTrend with ADX',
                     xaxis_title='Date',
                     yaxis_title='Price',
                     hovermode='x unified')
    #st.plotly_chart(fig, use_container_width=True)

elif strategy_type == "Ichimoku Cloud":
    strat_fig.add_trace(go.Scatter(x=df.index, y=pd.to_numeric(df['Close']), mode='lines', name='Close Price'))
    strat_fig.add_trace(go.Scatter(x=df.index, y=pd.to_numeric(df['ISA_9']), mode='lines', name='Leading Span A'))
    strat_fig.add_trace(go.Scatter(x=df.index, y=pd.to_numeric(df['ISB_26']), mode='lines', name='Leading Span B'))
    strat_fig.add_trace(go.Scatter(x=df.index, y=pd.to_numeric(df['ITS_9']), mode='lines', name='Conversion Line'))
    strat_fig.add_trace(go.Scatter(x=df.index, y=pd.to_numeric(df['IKS_26']), mode='lines', name='Base Line'))

    # fill
    strat_fig.add_trace(go.Scatter(
        x=df.index.tolist() + df.index.tolist()[::-1],
        y=pd.to_numeric(df['ISA_9']).tolist() + pd.to_numeric(df['ISB_26']).tolist()[::-1],
        fill='toself',
        fillcolor='rgba(0,176,246,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Cloud',
        showlegend=False
    ))

    strat_fig.update_layout(title='Ichimoku Cloud',
                     xaxis_title='Date',
                     yaxis_title='Price',
                     hovermode='x unified')
    #st.plotly_chart(fig, use_container_width=True)

elif strategy_type == "Parabolic SAR":
    strat_fig = plot_parabolic_sar(df)
elif strategy_type == "Aroon Indicator":
    #aroon_period = st.sidebar.slider("Aroon Period", 10, 50, 14)
    strat_fig = plot_aroon(df, period=aroon_period)
elif strategy_type == "Chaikin Oscillator":
    #chaikin_short_window = st.sidebar.slider("Chaikin Short Window", 5, 30, 10)
    #chaikin_long_window = st.sidebar.slider("Chaikin Long Window", 30, 100, 30)
    strat_fig = plot_chaikin_oscillator(df, short_period=chaikin_short_window, long_period=chaikin_long_window)
elif strategy_type == "Force Index":
    #force_index_window = st.sidebar.slider("Force Index Window", 5, 30, 13)
    strat_fig = plot_force_index(df, period=force_index_window)
elif strategy_type == "Money Flow Index (MFI)":
    #mfi_period = st.sidebar.slider("MFI Period", 10, 50, 14)
    #mfi_oversold = st.sidebar.slider("MFI Oversold Level", 10, 30, 20)
    #mfi_overbought = st.sidebar.slider("MFI Overbought Level", 70, 90, 80)
    strat_fig = plot_mfi(df, period=mfi_period, oversold=mfi_oversold, overbought=mfi_overbought)
#top_row_right.plotly_chart(strat_fig, use_container_width=True)
#main_tab.plotly_chart(strat_fig, use_container_width=True)
chart_tab.plotly_chart(strat_fig, use_container_width=True)
#st.plotly_chart(fig, use_container_width=True)



signals_df = None
#print(df.columns)
try:
    # ToDo improve this to make more elegant or standardize the naming of Date vs Datetime
    signals_df = pd.DataFrame({
        'Date': df['Date'],
        'Close': pd.to_numeric(df['Close']),
        'Position': signals['position']
    })
except Exception as e:
    #print(f'date- {e}')
    #print(df.columns)
    try:
        signals_df = pd.DataFrame({
            'Date': df['Datetime'],
            'Close': pd.to_numeric(df['Close']),
            'Position': signals['position']
        })
    except Exception as e:
        main_tab.warning("Could not load signals")
        #print(f'datetime- {e}')
        #print(df.columns)
#st.subheader("Recent Trade Signals") # temp remove trade signal
if signals_df is not None:
    signals_df['Signal'] = signals_df['Position'].diff()
    trades = signals_df[signals_df['Signal'] != 0].tail(10)
    trades['Action'] = trades['Signal'].map({1: 'Buy', -1: 'Sell'})
    #st.dataframe(trades[['Date', 'Close', 'Action']])

#main_tab.subheader("Additional Performance Metrics")
main_tab.divider()
wr_col, avg_wl_col, ann_vol_col, day_vol_col = main_tab.columns(4)
#market_comp_col.subheader("Additional")
#ratio_col.subheader("Performance")
#profit_drwdwn_col.subheader("Metrics")
#fee_slip_col.subheader("")
#st.subheader("Additional Performance Metrics")
# wr_col, avg_col = st.columns(2)
# market_comp_col, ratio_col, profit_drwdwn_col, fee_slip_col
if signals_df is not None:
    trades_all = signals_df[signals_df['Signal'] != 0].copy()
    trades_all['Next_Close'] = trades_all['Close'].shift(-1)
    trades_all['Trade_Return'] = (trades_all['Next_Close'] - trades_all['Close']) * trades_all['Signal']
    win_rate = (trades_all['Trade_Return'] > 0).mean() * 100

    avg_win = trades_all[trades_all['Trade_Return'] > 0]['Trade_Return'].mean()
    avg_loss = trades_all[trades_all['Trade_Return'] < 0]['Trade_Return'].mean()
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

    wr_col.metric("Win Rate", f"{win_rate:.2f}%")
    avg_wl_col.metric("Average Win", f"${avg_win:.2f}" if not pd.isna(avg_win) else "N/A")
    avg_wl_col.metric("Average Loss", f"${avg_loss:.2f}" if not pd.isna(avg_loss) else "N/A")

#chart_tab.subheader("Trade Return Distribution")
if signals_df is not None:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=trades_all['Trade_Return'], nbinsx=50))
    fig.update_layout(title='Distribution of Trade Returns',
                     xaxis_title='Return ($)',
                     yaxis_title='Frequency')
    chart_tab.plotly_chart(fig, use_container_width=True)

    #st.subheader("Risk Analysis")
    #col1, col2, col3 = st.columns(3)

    annual_volatility = np.sqrt(252) * positions['strategy_returns'].std() * 100

    var_95 = np.percentile(positions['strategy_returns'], 5) * 100
    var_99 = np.percentile(positions['strategy_returns'], 1) * 100

    ann_vol_col.metric("Annualized Volatility", f"{annual_volatility:.2f}%")
    day_vol_col.metric("95% VaR (Daily)", f"{var_95:.2f}%")
    day_vol_col.metric("99% VaR (Daily)", f"{var_99:.2f}%")

main_tab.plotly_chart(pv_fig, use_container_width=True)

@st.cache_resource
def get_pyg_renderer() -> "StreamlitRenderer":
    #df = pd.read_csv("./bike_sharing_dc.csv")
    # If you want to use feature of saving chart config, set `spec_io_mode="rw"`
    return StreamlitRenderer(df, spec="./gw_config.json", spec_io_mode="rw")

company_name = ""
with info_tab:
    conn_str = getenv("MDB_CLOUD_CNN")
    conn_str = conn_str.replace('"', '') # strip quotes from env variable being set
    if not conn_str:
        raise ValueError("MDB_CLOUD_CNN environment variable is not set.")

    if not conn_str.startswith(("mongodb://", "mongodb+srv://")):
        raise ValueError(f"Invalid MongoDB URI. It must start with 'mongodb://' or 'mongodb+srv://'.\n{conn_str}")

    client = MongoClient(conn_str,
        tlsAllowInvalidCertificates=True,
        tlsCAFile=None
    )
    db = client["company_db"]
    collection = db["company_info"]

    available_tickers = [doc["symbol"] for doc in collection.find({}, {"symbol": 1})]
    available_tickers.sort()

    st.title("Stock Ticker Information")


    ticker_symbol = selected_ticker #st.selectbox("Enter or select a stock ticker symbol:", available_tickers) # selected_ticker

    # Fetch data from yfinance
    if ticker_symbol:
        company_info = collection.find_one({"symbol": ticker_symbol})
        if company_info:
            try:
                avg_age = company_info.get("average_age", 0)
                avg_pay = company_info.get("average_pay", 0)
                std_dev_age = company_info.get("std_dev_age", 0)
                std_dev_pay = company_info.get("std_dev_pay", 0)
                full_time_employees = company_info.get("fullTimeEmployees", "N/A")
                total_officers = company_info.get("total_officers", 0)
                # Display basic information
                st.subheader(f"Information for {company_info.get('longName', ticker_symbol)}")
                company_name = company_info.get('longName', ticker_symbol)
                st.write(f"**Sector:** {company_info.get('sector', 'N/A')}")
                st.write(f"**Industry:** {company_info.get('industry', 'N/A')}")
                st.write(f"**Market Cap:** ${company_info.get('marketCap', 'N/A'):,}")
                st.write(f"**Current Price:** ${company_info.get('currentPrice', 'N/A'):.2f}")
                st.write(f"**52 Week High:** ${company_info.get('fiftyTwoWeekHigh', 'N/A'):.2f}")
                st.write(f"**52 Week Low:** ${company_info.get('fiftyTwoWeekLow', 'N/A'):.2f}")
                st.write(f"Average Age of Company Officers: {avg_age:.2f} years")
                st.write(f"Standard Deviation of Age: {std_dev_age:.2f} years")
                st.write(f"Average Pay of Company Officers: ${avg_pay:,.2f}")
                st.write(f"Standard Deviation of Pay: ${std_dev_pay:,.2f}")
                st.write(f"Total Number of Company Officers Listed: {total_officers}")
                st.write(f"Number of Full-Time Employees: {full_time_employees:,}")

                # Display additional info in an expandable section
                with st.expander("Show More Details"):
                    st.write(company_info)

            except Exception as e:
                st.error(f"Error fetching data for {ticker_symbol}. Please check the ticker symbol and try again.")
                st.write(f"Error details: {e}")
        else:
            st.warning(f"No data found for {ticker_symbol}. Please try another ticker.")
    else:
        st.warning("Please enter a valid stock ticker symbol.")





with gnews_tab:
    gn = GoogleNews()
    news_data = gn.search(company_name, when='1y')

    # Manually sort first 50 entries by published date
    if "entries" in news_data and news_data["entries"]:
        def safe_parse_date(date_str):
            try:
                return parser.parse(date_str)
            except Exception:
                return datetime.min

        entries = news_data["entries"][:50]  # Limit to first 50 entries
        sorted_entries = sorted(
            [(safe_parse_date(entry.get("published", "")), entry) for entry in entries],
            key=lambda x: x[0],
            reverse=True
        )
        sorted_entries = [entry[1] for entry in sorted_entries]
    else:
        sorted_entries = []

    with st.container():#(height=600):
        st.header(f"Latest News for {company_name}")
        if sorted_entries:
            for entry in sorted_entries[:50]:
                published_date = safe_parse_date(entry.get("published", "Unknown"))
                formatted_date = published_date.strftime("%Y-%m-%d %H:%M:%S") if published_date != datetime.min else "Unknown"
                #st.subheader(entry.get("title", "No Title"))
                #st.write(f"**Published:** {formatted_date}")
                st.markdown(f"### {entry.get('title', 'No Title')}")
                st.markdown(f"**Published:** {formatted_date}")
                st.markdown(entry.get("summary", "No Summary Available"),unsafe_allow_html=True)
                #st.markdown(f"[Read more]({entry.get('link')})")
                st.markdown("---")
        else:
            st.error("No news found for this ticker.")


with fvf_tab:
    #st.header("Insider Transactions")

    first_res = 0

    stock = finvizfinance(selected_ticker)
    company_name = stock.ticker_description().split(",")[first_res].strip()
    company_name_short = company_name.split(" ")[first_res].strip()
    try:
        insider_trades = stock.ticker_inside_trader()

        pd.set_option("display.max_columns", None)  # Show all columns
        pd.set_option("display.max_colwidth", None)  # Show all columns

        st.subheader(f"Latest Insider Transactions for {company_name} ({selected_ticker})")
        if not insider_trades.empty:
            columns_to_exclude = ["SEC Form 4", "SEC Form 4 Link", "Insider_id"]
            insider_trades_filtered = insider_trades.drop(columns=[col for col in columns_to_exclude if col in insider_trades.columns])
            st.dataframe(insider_trades_filtered, hide_index=True)
        else:
            st.error("No recent insider transactions found.")

        news = stock.ticker_news()


        if not news.empty:
            news_filtered = \
                            news[
                                  news["Link"].str.contains(company_name_short, case=False, na=False) \
                                | news["Title"].str.contains(company_name_short, case=False, na=False)
                                ]
        else:
            news_filtered = pd.DataFrame()
        st.subheader(f"Latest News for {company_name} seaching with {company_name_short}")
        if not news.empty:
            st.dataframe(news_filtered, hide_index=True)
        else:
            st.error("No recent insider transactions found.")
    except Exception as e:
        st.error(f"Unable to get FinViz Financial Info due to;\n{e}")



with data_ex_tab:
    renderer = get_pyg_renderer()
    renderer.explorer()


with raw_data_tab:
    st.subheader("Interactive Plotly Chart with Multiple Y-Axis Selections")
    #st.write("The X-axis will default to the 'Date' column.") # redundent/obvious?

    # ToDo add dropdown with different presets of default columns
    # ie bollinger bands info, momentum info,
    cols = df.columns.values
    #print(cols)
    #valid_cols = cols.remove("Date").remove("ticker")
    invalid_cols = ['Date', 'ticker', 'Ticker', 'Datetime']
    cols = cols.tolist()
    for cur_inv in invalid_cols:
        try:
            cols.remove(cur_inv)
        except ValueError as verr:
            pass # silently pass over missing keys
        except Exception as e:
            print(e)
    y_columns = st.multiselect("Select Y-axis metrics", options=cols, default=['Close', 'Adj Close', 'High', 'Low'])

    # Create a Plotly line chart based on user selection
    if y_columns:
        fig = px.line(df, x="Date", y=y_columns, title="Selected Metrics Over Time")
        st.plotly_chart(fig)
    else:
        st.write("Please select at least one metric to plot.")

    #dtale.show(df, subprocess=False)
    #dateless_df = df.drop('Date', axis=1)
    #show(dateless_df) # pandas gui - more standalone than web
    #dtale.show(dateless_df)#, subprocess=False) # doesn't show?


main_tab.subheader("Download Data")
col1, col2 = main_tab.columns(2)

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

if signals_df is not None:
    trades_csv = convert_df_to_csv(trades_all)
    signals_csv = convert_df_to_csv(signals_df)

    col1.download_button(
        label="Download All Trades",
        data=trades_csv,
        file_name='trades.csv',
        mime='text/csv',
    )

    col2.download_button(
        label="Download All Signals",
        data=signals_csv,
        file_name='signals.csv',
        mime='text/csv',
    )

#ToDo - refactor, split out util funcs, split out each tab + side bar
# given the added co info, google news, finviz news + insider transactions- really nice foundation
# would be interesting to see how to integrate insider transactions albeit given form4 / 10-Q is quarterly
#  not sure how much help it would be in strategies


## ToDo - features to add
## ? helper button that popups translation of indicators to meaning, maybe as separate tab vs popup
## cleaner custom strategy builder - pop up?
## experiment features
#  - option to test on all stocks in same industry + sub category // needs summary report
#  - option to mod selected input variables by selected amt (default 1 std dev or 10%)
#   build out report feature allowing user to drill into/open selected strategy
#    base report feature should have overview of which weights/input values used had best results
#    nasty implement nice feature to have, combining experiment with select input variables + test on all stocks in same industry + sub category
#  - option to do to best of, ie if custom strategy includes 3 or more indicators, use # indicators -1 rotate
#    again nasty implement nice feature to pair this with testing on stocks in same industry + sub cat // sounds like useful table to have or helper func
#    then imagine pairing that with variability / exp on input values, holy processing mackeral
#    num indicators 3 * num input variants on 1 input 3 * # of stocks in same industry + sub category
#     leaves min 9x # stocks to compare
#     say 2 inputs exp, 3*2*3*#stocks = 18 * # stocks
#    parallel + distributed seem like a good idea
## email + text alerts // based on daily, common gateways;
#  Verizon: {number}@vtext.com
#  AT&T: {number}@txt.att.net
#  T-Mobile: {number}@tmomail.net
#  Sprint: {number}@messaging.sprintpcs.com
#  Google Voice: {number}@txt.voice.google.com
#    figure out if gmail allows for sending as, ie you have example@gmail.com
#     but send email as alert.example@gmail.com
## accounts / save custom strategies / load custom strategies
## forums? disque overflow something else?
## monthly/quarterly competitions for best performing strategy
# - need lots of caveats and solid account setup, same strat can't be reused, can be team but then rewards split, must be reproducible, realistic
