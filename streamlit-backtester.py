import streamlit as st
import pandas as pd
import pandas_ta as ta
import numpy as np
import yfinance as yf
import sqlite3
import plotly.graph_objects as go

from datetime import datetime, timedelta


st.set_page_config(page_title="Stock Strategy Backtester", layout="wide")


st.sidebar.header("Strategy Parameters")


db_files = ['index_dbs/dow30_data1900-01-01_to_2025-01-01.db',
            'index_dbs/nasdaq100_data1900-01-01_to_2025-01-01.db',
            'index_dbs/ARM_TMSC_data_2024-12-10_to_2025-01-08_minute.db',]  #["nasdaq100_data.db", "dow30_data.db"]   # "sap500_data.db",
selected_db = st.sidebar.selectbox("Select Database", db_files)


@st.cache_resource
def get_available_tickers(db_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    conn.close()
    return [table[0] for table in tables]


available_tickers = get_available_tickers(selected_db)
selected_ticker = st.sidebar.selectbox("Select Ticker", available_tickers)


start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", datetime.now())


strategy_type = st.sidebar.selectbox(
    "Select Strategy Type",
    [
        "Moving Average Crossover", "RSI", "MACD", "Bollinger Bands", "SuperTrend", "Ichimoku Cloud",
        #"Parabolic SAR", "Aroon Indicator", "Chaikin Oscillator", "Force Index", "Money Flow Index (MFI)",
        #"Commodity Channel Index (CCI)"#, "Exponential Moving Average (EMA) Crossover"
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


initial_investment = st.sidebar.number_input("Initial Investment ($)", value=10000.0, step=1000.0)


@st.cache_data
def load_data(ticker, db_name):
    conn = sqlite3.connect(db_name)
    query = f'SELECT * FROM "{ticker}"'
    df = pd.read_sql_query(query, conn)
    conn.close()
    try:
        df['Date'] = pd.to_datetime(df['Date'])
    except Exception as e:
        print(e)

        try:
            df['Date'] = pd.to_datetime(df['Datetime'])
        except Exception as e:
            print(e)

    #new_columns = {col: col.replace('.', '_') for col in df.columns if '.' in col}
    #df = df.rename(columns=new_columns)
    #numeric_columns = df.columns.drop(['ticker', 'Date'])
    #df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    return df


def implement_strategy(df, strategy_type):
    signals = pd.DataFrame(index=df.index)
    signals['position'] = 0

    if strategy_type == "Moving Average Crossover":
        if ma_type == "SMA":
            df['MA_short'] = pd.to_numeric(df['sma'].shift(short_window), errors='coerce')
            df['MA_long'] = pd.to_numeric(df['sma'].shift(long_window), errors='coerce')
        elif ma_type == "EMA":
            df['MA_short'] = pd.to_numeric(df['ema'].shift(short_window), errors='coerce')
            df['MA_long'] = pd.to_numeric(df['ema'].shift(long_window), errors='coerce')
        elif ma_type == "HMA":
            df['MA_short'] = pd.to_numeric(df['hma'].shift(short_window), errors='coerce')
            df['MA_long'] = pd.to_numeric(df['hma'].shift(long_window), errors='coerce')
        elif ma_type == "TMA":
            df['MA_short'] = pd.to_numeric(df['tma'].shift(short_window), errors='coerce')
            df['MA_long'] = pd.to_numeric(df['tma'].shift(long_window), errors='coerce')
        elif ma_type == "RMA":
            df['MA_short'] = pd.to_numeric(df['rma'].shift(short_window), errors='coerce')
            df['MA_long'] = pd.to_numeric(df['rma'].shift(long_window), errors='coerce')

        signals['position'] = np.where(df['MA_short'] > df['MA_long'], 1, 0)

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


    signals['position'] = signals['position'].fillna(0)
    return df, signals


def calculate_performance(df, signals, initial_investment):
    positions = pd.DataFrame(index=signals.index)
    positions['position'] = signals['position']

    # calculate time unit returns // either daily or minutes depending on db
    df['returns'] = pd.to_numeric(df['Close']).pct_change()

    positions['strategy_returns'] = positions['position'].shift(1) * df['returns']
    positions['strategy_returns'] = positions['strategy_returns'].fillna(0)

    positions['cumulative_returns'] = (1 + positions['strategy_returns']).cumprod()
    positions['cumulative_market_returns'] = (1 + df['returns']).cumprod()

    positions['portfolio_value'] = initial_investment * positions['cumulative_returns']
    positions['market_value'] = initial_investment * positions['cumulative_market_returns']

    total_return = (positions['portfolio_value'].iloc[-1] - initial_investment) / initial_investment * 100
    market_return = (positions['market_value'].iloc[-1] - initial_investment) / initial_investment * 100
    sharpe_ratio = np.sqrt(252) * positions['strategy_returns'].mean() / positions['strategy_returns'].std()
    max_drawdown = (positions['portfolio_value'] / positions['portfolio_value'].cummax() - 1).min() * 100

    return positions, total_return, market_return, sharpe_ratio, max_drawdown




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
#df = df[(df['Date'] >= pd.Timestamp(start_date)) & (df['Date'] <= pd.Timestamp(end_date))]
#pd.to_datetime(df['Date'])
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


df, signals = implement_strategy(df, strategy_type)
positions, total_return, market_return, sharpe_ratio, max_drawdown = calculate_performance(df, signals, initial_investment)


col1, col2, col3, col4 = st.columns(4)
col1.metric("Strategy Return", f"{total_return:.2f}%")
col2.metric("Market Return", f"{market_return:.2f}%")
col3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
col4.metric("Max Drawdown", f"{max_drawdown:.2f}%")


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
signals_df = pd.DataFrame({
    'Date': df['Datetime'],
    'Close': pd.to_numeric(df['Close']),
    'Position': signals['position']
})
signals_df['Signal'] = signals_df['Position'].diff()
trades = signals_df[signals_df['Signal'] != 0].tail(10)
trades['Action'] = trades['Signal'].map({1: 'Buy', -1: 'Sell'})
st.dataframe(trades[['Date', 'Close', 'Action']])


st.subheader("Additional Performance Metrics")
col1, col2, col3 = st.columns(3)

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

#using yfinance_downloader.py as ex, write a function that will open given database (passed as parameter) and chk for the given tables for latest info by date column, compare to current date, then find- obtain- and update db with any missing data
