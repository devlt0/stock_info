import streamlit as st
import pandas as pd
import pandas_ta as ta
import numpy as np
import yfinance as yf
import sqlite3
import plotly.graph_objects as go

from datetime import datetime, timedelta
from os import getcwd, listdir







st.set_page_config(page_title="Stock Strategy Backtester", layout="wide")


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

selected_db = st.sidebar.selectbox("Select Database", db_files)

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
    print(f'db path {db_path}')
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
selected_ticker = st.sidebar.selectbox("Select Ticker", available_tickers)
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
            indicator = 'sma'
        elif ma_type == "EMA":
            indicator = 'ema'
        elif ma_type == "HMA":
            indicator = 'hma'
        elif ma_type == "TMA":
            indicator = 'tma'
        elif ma_type == "RMA":
            indicator = 'rma'

        if len(indicator) > 0:
            df['MA_short'] = pd.to_numeric(df[indicator].shift(short_window), errors='coerce')
            df['MA_long'] = pd.to_numeric(df[indicator].shift(long_window), errors='coerce')

            signals['position'] = np.where(df['MA_short'] > df['MA_long'], 1, 0)
        else:
            raise Exception(f"Unable to find associated indicator for {ma_type} in Moving Avg Crossover")

    elif strategy_type == "RSI":
        signals['position'] = np.where(pd.to_numeric(df['rsi']) < oversold, 1,
                                     np.where(pd.to_numeric(df['rsi']) > overbought, 0,
                                     signals['position'].shift(1)))

    elif strategy_type == "MACD":
        if signal_crossover:
            signals['position'] = np.where(
                pd.to_numeric(df['MACD_12_26_9']) > pd.to_numeric(df['MACDs_12_26_9']), 1, 0)
        else:
            signals['position'] = np.where(pd.to_numeric(df['MACD_12_26_9']) > 0, 1, 0)

    elif strategy_type == "Bollinger Bands":
        if bb_strategy == "Mean Reversion":
            signals['position'] = np.where(pd.to_numeric(df['Close']) < pd.to_numeric(df['BBL_14_2.0']), 1,
                                         np.where(pd.to_numeric(df['Close']) > pd.to_numeric(df['BBU_14_2.0']), 0,
                                         signals['position'].shift(1)))
        else:  # Breakout
            signals['position'] = np.where(pd.to_numeric(df['Close']) > pd.to_numeric(df['BBU_14_2.0']), 1,
                                         np.where(pd.to_numeric(df['Close']) < pd.to_numeric(df['BBL_14_2.0']), 0,
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


def calculate_performance(df, signals, initial_investment):
    slippage_factor = 1 - slippage

    positions = pd.DataFrame(index=signals.index)
    positions['position'] = signals['position']

    # calculate time unit returns // either daily or minutes depending on db
    df['returns'] = pd.to_numeric(df['Close']).pct_change()

    positions['strategy_returns'] = positions['position'].shift(1) * df['returns'] * slippage_factor
    positions['strategy_returns'] = positions['strategy_returns'].fillna(0)

    positions['cumulative_returns'] = (1 + positions['strategy_returns']).cumprod()
    positions['cumulative_market_returns'] = (1 + df['returns']).cumprod()

    positions['portfolio_value'] = initial_investment * positions['cumulative_returns']
    positions['market_value'] = initial_investment * positions['cumulative_market_returns']

    total_return = (positions['portfolio_value'].iloc[-1] - initial_investment) / initial_investment * 100
    market_return = (positions['market_value'].iloc[-1] - initial_investment) / initial_investment * 100
    sharpe_ratio = np.sqrt(252) * positions['strategy_returns'].mean() / positions['strategy_returns'].std()
    max_drawdown = (positions['portfolio_value'] / positions['portfolio_value'].cummax() - 1).min() * 100


    downside_returns = positions['strategy_returns'][positions['strategy_returns'] < 0]
    downside_deviation = downside_returns.std()
    sortino_ratio = np.sqrt(252) * positions['strategy_returns'].mean() / downside_deviation if downside_deviation != 0 else 0


    pure_profit = positions['portfolio_value'].iloc[-1] - initial_investment

    return positions, total_return, market_return, sharpe_ratio, max_drawdown, sortino_ratio, pure_profit




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

    st.plotly_chart(fig)

def plot_aroon(df, period=aroon_period):
    df['AroonUp'], df['AroonDown'] = ta.AROON(df['high'], df['low'], timeperiod=period)


    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['AroonUp'], mode='lines', name='Aroon Up', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=df.index, y=df['AroonDown'], mode='lines', name='Aroon Down', line=dict(color='red')))

    fig.add_trace(go.Scatter(x=df.index, y=[70]*len(df), mode='lines', name='Overbought', line=dict(dash='dash', color='gray')))
    fig.add_trace(go.Scatter(x=df.index, y=[30]*len(df), mode='lines', name='Oversold', line=dict(dash='dash', color='gray')))

    fig.update_layout(title="Aroon Indicator", xaxis_title='Date', yaxis_title='Aroon Value', template="plotly_dark")

    st.plotly_chart(fig)

def plot_chaikin_oscillator(df, short_period=chaikin_short_window, long_period=chaikin_long_window):
    df['Chaikin'] = ta.AD(df['high'], df['low'], df['close'], df['volume'])
    df['ChaikinOscillator'] = ta.EMA(df['Chaikin'], timeperiod=short_period) - ta.EMA(df['Chaikin'], timeperiod=long_period)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df.index, y=df['ChaikinOscillator'], mode='lines', name='Chaikin Oscillator', line=dict(color='purple')))
    fig.add_trace(go.Scatter(x=df.index, y=[0]*len(df), mode='lines', name='Zero Line', line=dict(dash='dash', color='gray')))
    fig.update_layout(title="Chaikin Oscillator", xaxis_title='Date', yaxis_title='Oscillator Value', template="plotly_dark")

    st.plotly_chart(fig)

def plot_force_index(df, period=force_index_window):
    df['ForceIndex'] = ta.EMA(df['close'] - df['close'].shift(1), timeperiod=period) * df['volume']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['ForceIndex'], mode='lines', name='Force Index', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df.index, y=[0]*len(df), mode='lines', name='Zero Line', line=dict(dash='dash', color='gray')))
    fig.update_layout(title="Force Index", xaxis_title='Date', yaxis_title='Force Index Value', template="plotly_dark")
    st.plotly_chart(fig)

def plot_mfi(df, period=mfi_period, oversold=mfi_oversold, overbought=mfi_overbought):
    df['MFI'] = ta.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=period)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['MFI'], mode='lines', name='Money Flow Index', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=[oversold]*len(df), mode='lines', name='Oversold', line=dict(dash='dash', color='red')))
    fig.add_trace(go.Scatter(x=df.index, y=[overbought]*len(df), mode='lines', name='Overbought', line=dict(dash='dash', color='green')))
    fig.update_layout(title="Money Flow Index", xaxis_title='Date', yaxis_title='MFI Value', template="plotly_dark")

    st.plotly_chart(fig)






st.title("Stock Strategy Backtester")
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
raw_slippage = st.sidebar.number_input("Slippage Percentage", value=1.0, step=1.0)
slippage = raw_slippage/100






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
positions, total_return, market_return, sharpe_ratio, max_drawdown, sortino_ratio, pure_profit = calculate_performance(df, signals, initial_investment)


market_comp_col, ratio_col, profit_drwdwn_col = st.columns(3)
market_comp_col.metric("Strategy Return", f"{total_return:.2f}%")
market_comp_col.metric("Market Return", f"{market_return:.2f}%")
ratio_col.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
ratio_col.metric("Sortino Ratio", f"{sortino_ratio:.2f}")
profit_drwdwn_col.metric("Max Drawdown", f"{max_drawdown:.2f}%")
profit_drwdwn_col.metric("Pure Profit", f"${pure_profit:.2f}")


fig = go.Figure()
fig.add_trace(go.Scatter(x=positions.index, y=positions['portfolio_value'],
                        mode='lines', name='Strategy', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=positions.index, y=positions['market_value'],
                        mode='lines', name='Buy & Hold', line=dict(color='gray')))
fig.update_layout(title='Portfolio Value Over Time',
                 xaxis_title='Date',
                 yaxis_title='Portfolio Value ($)',
                 hovermode='x unified')
st.plotly_chart(fig, use_container_width=True)


if strategy_type == "Moving Average Crossover":
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=pd.to_numeric(df['Close']), mode='lines', name='Close Price'))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_short'], mode='lines', name=f'{ma_type} Short'))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_long'], mode='lines', name=f'{ma_type} Long'))
    fig.update_layout(title=f'{ma_type} Crossover',
                     xaxis_title='Date',
                     yaxis_title='Price',
                     hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

elif strategy_type == "RSI":
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=pd.to_numeric(df['rsi']), mode='lines', name='RSI'))
    fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1], y0=oversold, y1=oversold,
                 line=dict(color="green", width=2, dash="dash"))
    fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1], y0=overbought, y1=overbought,
                 line=dict(color="red", width=2, dash="dash"))
    fig.update_layout(title='Relative Strength Index',
                     xaxis_title='Date',
                     yaxis_title='RSI',
                     hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

elif strategy_type == "MACD":
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=pd.to_numeric(df['MACD_12_26_9']), mode='lines', name='MACD'))
    fig.add_trace(go.Scatter(x=df.index, y=pd.to_numeric(df['MACDs_12_26_9']), mode='lines', name='Signal'))
    fig.add_trace(go.Bar(x=df.index, y=pd.to_numeric(df['MACDh_12_26_9']), name='Histogram'))
    fig.update_layout(title='MACD',
                     xaxis_title='Date',
                     yaxis_title='Value',
                     hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

elif strategy_type == "Bollinger Bands":
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=pd.to_numeric(df['Close']), mode='lines', name='Close Price'))
    fig.add_trace(go.Scatter(x=df.index, y=pd.to_numeric(df['BBU_14_2.0']), mode='lines', name='Upper Band'))
    fig.add_trace(go.Scatter(x=df.index, y=pd.to_numeric(df['BBL_14_2.0']), mode='lines', name='Lower Band'))
    fig.add_trace(go.Scatter(x=df.index, y=pd.to_numeric(df['BBM_14_2.0']), mode='lines', name='Middle Band'))
    fig.update_layout(title='Bollinger Bands',
                     xaxis_title='Date',
                     yaxis_title='Price',
                     hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

elif strategy_type == "SuperTrend":
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=pd.to_numeric(df['Close']), mode='lines', name='Close Price'))
    fig.add_trace(go.Scatter(x=df.index, y=pd.to_numeric(df['SUPERT_7_3.0']), mode='lines', name='SuperTrend'))
    fig.add_trace(go.Scatter(x=df.index, y=pd.to_numeric(df['ADX_14']), mode='lines', name='ADX'))
    fig.update_layout(title='SuperTrend with ADX',
                     xaxis_title='Date',
                     yaxis_title='Price',
                     hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

elif strategy_type == "Ichimoku Cloud":
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=pd.to_numeric(df['Close']), mode='lines', name='Close Price'))
    fig.add_trace(go.Scatter(x=df.index, y=pd.to_numeric(df['ISA_9']), mode='lines', name='Leading Span A'))
    fig.add_trace(go.Scatter(x=df.index, y=pd.to_numeric(df['ISB_26']), mode='lines', name='Leading Span B'))
    fig.add_trace(go.Scatter(x=df.index, y=pd.to_numeric(df['ITS_9']), mode='lines', name='Conversion Line'))
    fig.add_trace(go.Scatter(x=df.index, y=pd.to_numeric(df['IKS_26']), mode='lines', name='Base Line'))

    # fill
    fig.add_trace(go.Scatter(
        x=df.index.tolist() + df.index.tolist()[::-1],
        y=pd.to_numeric(df['ISA_9']).tolist() + pd.to_numeric(df['ISB_26']).tolist()[::-1],
        fill='toself',
        fillcolor='rgba(0,176,246,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Cloud',
        showlegend=False
    ))

    fig.update_layout(title='Ichimoku Cloud',
                     xaxis_title='Date',
                     yaxis_title='Price',
                     hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

elif strategy_type == "Parabolic SAR":
    plot_parabolic_sar(df)
elif strategy_type == "Aroon Indicator":
    #aroon_period = st.sidebar.slider("Aroon Period", 10, 50, 14)
    plot_aroon(df, period=aroon_period)
elif strategy_type == "Chaikin Oscillator":
    #chaikin_short_window = st.sidebar.slider("Chaikin Short Window", 5, 30, 10)
    #chaikin_long_window = st.sidebar.slider("Chaikin Long Window", 30, 100, 30)
    plot_chaikin_oscillator(df, short_period=chaikin_short_window, long_period=chaikin_long_window)
elif strategy_type == "Force Index":
    #force_index_window = st.sidebar.slider("Force Index Window", 5, 30, 13)
    plot_force_index(df, period=force_index_window)
elif strategy_type == "Money Flow Index (MFI)":
    #mfi_period = st.sidebar.slider("MFI Period", 10, 50, 14)
    #mfi_oversold = st.sidebar.slider("MFI Oversold Level", 10, 30, 20)
    #mfi_overbought = st.sidebar.slider("MFI Overbought Level", 70, 90, 80)
    plot_mfi(df, period=mfi_period, oversold=mfi_oversold, overbought=mfi_overbought)


st.subheader("Recent Trade Signals")
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
        st.warning("Could not load signals")
        #print(f'datetime- {e}')
        #print(df.columns)

if signals_df is not None:
    signals_df['Signal'] = signals_df['Position'].diff()
    trades = signals_df[signals_df['Signal'] != 0].tail(10)
    trades['Action'] = trades['Signal'].map({1: 'Buy', -1: 'Sell'})
    st.dataframe(trades[['Date', 'Close', 'Action']])


st.subheader("Additional Performance Metrics")
col1, col2, col3 = st.columns(3)
if signals_df is not None:
    trades_all = signals_df[signals_df['Signal'] != 0].copy()
    trades_all['Next_Close'] = trades_all['Close'].shift(-1)
    trades_all['Trade_Return'] = (trades_all['Next_Close'] - trades_all['Close']) * trades_all['Signal']
    win_rate = (trades_all['Trade_Return'] > 0).mean() * 100

    avg_win = trades_all[trades_all['Trade_Return'] > 0]['Trade_Return'].mean()
    avg_loss = trades_all[trades_all['Trade_Return'] < 0]['Trade_Return'].mean()
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

    col1.metric("Win Rate", f"{win_rate:.2f}%")
    col2.metric("Average Win", f"${avg_win:.2f}" if not pd.isna(avg_win) else "N/A")
    col3.metric("Average Loss", f"${avg_loss:.2f}" if not pd.isna(avg_loss) else "N/A")

st.subheader("Trade Return Distribution")
if signals_df is not None:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=trades_all['Trade_Return'], nbinsx=50))
    fig.update_layout(title='Distribution of Trade Returns',
                     xaxis_title='Return ($)',
                     yaxis_title='Frequency')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Risk Analysis")
    col1, col2, col3 = st.columns(3)

    annual_volatility = np.sqrt(252) * positions['strategy_returns'].std() * 100

    var_95 = np.percentile(positions['strategy_returns'], 5) * 100
    var_99 = np.percentile(positions['strategy_returns'], 1) * 100

    col1.metric("Annualized Volatility", f"{annual_volatility:.2f}%")
    col2.metric("95% VaR (Daily)", f"{var_95:.2f}%")
    col3.metric("99% VaR (Daily)", f"{var_99:.2f}%")


st.subheader("Download Data")
col1, col2 = st.columns(2)

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

