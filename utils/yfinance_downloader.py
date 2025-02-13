from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
from os import listdir, getcwd, path
from time import sleep
from traceback import print_exc

#from memory_profiler import profile
#import objgraph
#import tracemalloc

import gc
import sqlite3
import warnings

import pandas as pd
import pandas_ta as ta
import yfinance as yf



'''
Pandas TA - Technical Analysis Indicators - v0.3.14b0
Total Indicators & Utilities: 205
Abbreviations:
    aberration, above, above_value, accbands, ad, adosc, adx, alma, amat, ao, aobv, apo, aroon, atr, bbands, below, below_value, bias, bop, brar, cci, cdl_pattern, cdl_z, cfo, cg, chop, cksp, cmf, cmo, coppock, cross, cross_value, cti, decay, decreasing, dema, dm, donchian, dpo, ebsw, efi, ema, entropy, eom, er, eri, fisher, fwma, ha, hilo, hl2, hlc3, hma, hwc, hwma, ichimoku, increasing, inertia, jma, kama, kc, kdj, kst, kurtosis, kvo, linreg, log_return, long_run, macd, mad, massi, mcgd, median, mfi, midpoint, midprice, mom, natr, nvi, obv, ohlc4, pdist, percent_return, pgo, ppo, psar, psl, pvi, pvo, pvol, pvr, pvt, pwma, qqe, qstick, quantile, rma, roc, rsi, rsx, rvgi, rvi, short_run, sinwma, skew, slope, sma, smi, squeeze, squeeze_pro, ssf, stc, stdev, stoch, stochrsi, supertrend, swma, t3, td_seq, tema, thermo, tos_stdevall, trima, trix, true_range, tsi, tsignals, ttm_trend, ui, uo, variance, vhf, vidya, vortex, vp, vwap, vwma, wcp, willr, wma, xsignals, zlma, zscore

Candle Patterns:
    2crows, 3blackcrows, 3inside, 3linestrike, 3outside, 3starsinsouth, 3whitesoldiers, abandonedbaby, advanceblock, belthold, breakaway, closingmarubozu, concealbabyswall, counterattack, darkcloudcover, doji, dojistar, dragonflydoji, engulfing, eveningdojistar, eveningstar, gapsidesidewhite, gravestonedoji, hammer, hangingman, harami, haramicross, highwave, hikkake, hikkakemod, homingpigeon, identical3crows, inneck, inside, invertedhammer, kicking, kickingbylength, ladderbottom, longleggeddoji, longline, marubozu, matchinglow, mathold, morningdojistar, morningstar, onneck, piercing, rickshawman, risefall3methods, separatinglines, shootingstar, shortline, spinningtop, stalledpattern, sticksandwich, takuri, tasukigap, thrusting, tristar, unique3river, upsidegap2crows, xsidegap3methods
'''
def get_latest_date(database_path, table_name, date_column='Date'):
    latest_date = ""
    try:
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()

        query = f"SELECT MAX({date_column}) FROM '{table_name}'"

        cursor.execute(query)
        latest_date = cursor.fetchone()[0]
        conn.close()
    except Exception as e:
        print(e)

    return latest_date

# ToDo, improve download such that only seeks to download missing data,
#  merges that missing with existing data-
#  then calcs missing values where appropriate
#    vs recalc all or only calc on new data leaving gaps where min # of days not met
#@profile
def download_and_store_stock_data_minutes(tickers, start_date="1900-01-01", end_date="", get_minutes=False,
    _shard_on_ticker=True, _def_db_basename="stock_data", _ticker_unknown_state=True,
    _output_dir="", _dl_slack_days=2, _skip_indicators=False, _cpu_cores = 6):

    # use start date as current date - 30 d
    # end date as current date
    # iterate 7 days at a time b/c yahoo
    # limit to 1.8/1.9s per query to avoid rate limit

    yahoo_lookback_max_limit = 29 #still causes issues b/c long run time and now not in loop x_X #days # tried 30 fails,  with today can go back 29 additional
    yahoo_efficient_lkbk_lmt = 28 # reduce 5 query per ticker to 4
    tickers_to_dbl_chk = []
    db_name = _def_db_basename + datetime.now().strftime('%Y-%m-%d') + '.db'
    if _output_dir:
        db_name = _output_dir + "/" + db_name if not _output_dir.endswith("/") else _output_dir + db_name
    ticker_iter = 1
    num_tickers = len(tickers)
    rate_limit_delay = 1.9 # could go 1.8 for exactly 2000 req in one hour but on chance get blocked, add buffer of 0.1s
    performance_indicators      = ["drawdown", "log_return", "percent_return"] # 3
    utility_indicators          = ['above', 'above_value', 'below', 'below_value', 'cross'] # 5
    beta_performance_indicators = ['cagr', 'calmar_ratio', 'downside_deviation', 'jensens_alpha', 'log_max_drawdown', 'max_drawdown', 'pure_profit_score', 'sharpe_ratio', 'sortino_ratio', 'volatility'] # 10
    problematic_indicators      = ['mgcd'] # 1
    indicators_to_skip = []     # 19 filtered out
    indicators_to_skip.extend( performance_indicators )
    indicators_to_skip.extend( beta_performance_indicators )
    indicators_to_skip.extend( utility_indicators )
    indicators_to_skip.extend( problematic_indicators )

    all_ta_indicators = pd.DataFrame().ta.indicators(as_list=True) # don't do it the normal way b/c it name doesn't match get this..., exclude=indicators_to_skip) # 143 before filtering
    filtered_ta_indicators = [indicator for indicator in all_ta_indicators if indicator.lower() not in indicators_to_skip] # 124 indicators
    date_vs_datetime = "Datetime" if get_minutes else "Date"

    for ticker in tickers:
        data = None
        existing_latest_date = None
        if _shard_on_ticker:
            db_name = f'{ticker}.db'
            if _output_dir:
                db_name = _output_dir + "/" + db_name if not _output_dir.endswith("/") else _output_dir + db_name
        if path.exists(db_name):
            print(f"attempting to get latest date for {db_name}")
            existing_latest_date = get_latest_date(database_path=db_name, table_name=ticker, date_column=date_vs_datetime)
            try:
                date_format_str = "%Y-%m-%d %H:%M:%S"
                if '+' in existing_latest_date:
                    date_format_str += "%z"
                ex_latest_date_obj = datetime.strptime(existing_latest_date, date_format_str).date() #datetime.strptime(existing_latest_date, "%Y-%m-%d %H:%M:%S").date()
            except Exception as e:

                try:
                    ex_latest_date_obj = datetime.strptime(existing_latest_date, "%Y-%m-%d").date()
                except Exception as e:
                    print(existing_latest_date)
                    print(f"failed to convert existing latest date {existing_latest_date} for {ticker}")
                    raise(e)
        print(f"Downloading data for {ticker}... {ticker_iter} out of {num_tickers}")
        ticker_iter += 1
        first_pass = True
        end_date_obj   = datetime.now()

        # minutes vs daily info download logic
        if get_minutes:
            if existing_latest_date:
                if ex_latest_date_obj >= end_date_obj.date()-timedelta(days=_dl_slack_days):
                    print(f"Skipping {ticker}, already up to date")
                    continue
                #else:
                # add logic to make start date stop at latest date from pick up
            start_date_obj = end_date_obj - timedelta(days=yahoo_efficient_lkbk_lmt)
            current_start_obj = start_date_obj #

            data = pd.DataFrame()
            while current_start_obj < end_date_obj:

                current_end_obj = current_start_obj + timedelta(days=7)
                cur_data = yf.download(ticker, start=current_start_obj.strftime('%Y-%m-%d'), end=current_end_obj.strftime('%Y-%m-%d'), interval='1m')
                #sleep(rate_limit_delay)
                #data['ticker'] = ticker
                #data["Date"] = data.index
                cur_data = cur_data.xs(ticker, level=1, axis=1)

                data = pd.concat([data, cur_data]) if data is not None else cur_data #, ignore_index=True)
                data.drop_duplicates(inplace = True)
                current_start_obj += timedelta(days=7)
                sleep(rate_limit_delay)
        else:
            if existing_latest_date:

                if ex_latest_date_obj >= end_date_obj.date()-timedelta(days=_dl_slack_days):
                    print(f"Skipping {ticker}, already up to date")
                    continue
                else:
                    new_start_date = ex_latest_date_obj + timedelta(days=1)
                    data = yf.download(ticker, start=new_start_date, end=end_date_obj.strftime('%Y-%m-%d'))
                    #ToDo implement logic to handle updating partial
                    # need to grab missing data
                    # merge with existing data
                    # only apply tickers to new data
                    # only add new date/datetime rows
            else:
                try:
                    data = yf.download(ticker, period='max')#start=start_date, end=end_date_obj.strftime('%Y-%m-%d'))
                except ConnectionResetError as not_there_err:
                    print(not_there_err)
                    tickers_to_dbl_chk.append(ticker)
                    continue

            data = data.xs(ticker, level=1, axis=1)
            if _skip_indicators:
                sleep(rate_limit_delay)
            # else indicators will take long enough don't need hard delay to avoid rate limiting yahoo finance api

        if data is None or data.empty:
            print(f"Was unable to obtain data for ticker {ticker}")
            tickers_to_dbl_chk.append(ticker)
            continue
        # strip multi index, move ticker to column instead of row

        data.drop_duplicates(inplace = True) # for rows
        conn = None

        try:
            # dynamically apply all indicators
            # since some of the indicators written for earlier vers of pandas
            #  try default way of append=True,that fails, try pandas.concat, that fails we mark it and move on
            if not _skip_indicators:
                indicators_to_concat = []
                indicators_to_concat = apply_indicators_parallel(data, filtered_ta_indicators, _cpu_cores)
                #= pool.starmap(apply_indicator, [(data, indicator) for indicator in filtered_ta_indicators])
                #with Pool(_cpu_cores) as pool:
                #    indicators_to_concat = pool.starmap(apply_indicator, [(data, indicator) for indicator in indicators])
                '''
                for cur_indicator in filtered_ta_indicators:
                    #print(f"Attempting to apply indicator       {cur_indicator}")
                    ta_function = getattr(data.ta, cur_indicator)
                    try:
                        ta_function(append=True)
                        continue
                    except Exception as e:
                        print(e)
                        try:
                            indicator_res = ta_function(append=False)
                            indicators_to_concat.append( indicator_res )
                            del indicator_res

                        except Exception as e:
                            print(e)

                    del ta_function
                    gc.collect()


                if len(indicators_to_concat)>0:
                    #data = pd.concat([data, indicator_res], axis=1)
                    indicators_to_concat.insert(0, data)
                    try:
                        data = pd.concat(indicators_to_concat, axis=1)
                    except Exception as e:
                        print(e)
                        tickers_to_dbl_chk.append(ticker)
                    del indicators_to_concat
                    gc.collect()
                '''
                indicator_count = 0
                percent_range_match = 0.05
                for result in indicators_to_concat:
                    headers = result.columns if isinstance(result, type(pd.DataFrame())) else result.name if isinstance(result, type(pd.Series()) ) else result
                    #indicator_added = False
                    if result is not None:
                        try:
                            len_data_index = len(data.index)
                            len_result_index = len(result.index)
                            if data.index.equals(result.index): #or \
                            #('Date' in data.columns and 'Date' in result.columns \
                            #and data['Date'].equals(result['Date']) ):
                                #data = pd.concat([data, result], axis=1)
                                #indicator_count += 1
                                pass # skip to end, no prep or changes required
                            elif len_data_index == len_result_index:
                                result = result.set_index(data.index)
                                #data = pd.concat([data, result], axis=1)
                                #indicator_count += 1
                            elif len_result_index > len_data_index * (1-percent_range_match) \
                            and len_result_index < len_data_index:
                                offset = len_data_index - len_result_index  # Calculate offset
                                result = result.set_index(data.index[offset:])  # Shift index
                                #data = pd.concat([data, result], axis=1)
                                #indicator_count += 1

                            else:
                                print(f"Skipping concatenation: indices do not match.\n{headers}")
                                print(f"data index: {len(data.index)}")
                                print(f"result index: {len(result.index)}")
                                continue # will skip the concat
                            data = pd.concat([data, result], axis=1)
                            indicator_count += 1


                        except Exception as e:
                            print(e)
                    else:
                        print(f"Skipping concatenation: no result data.\n{headers}")
                #data = data.T.drop_duplicates().T # remove duplicate columns does NOT work
                data = data.loc[:, ~data.columns.duplicated()]
                #data = data.drop(columns=['DMP_14', 'DMN_14', 'Adj Close']) # not sure why we duplicate ere
                #data.columns = pd.Index([f"{col}___{i}" if col in data.columns[:i] else col for i, col in enumerate(data.columns)])

                print(f"used {indicator_count} # indicators")
                del indicators_to_concat
                gc.collect()
            #date_vs_datetime = ""
            if get_minutes:
                data.insert(0, 'Datetime', data.index)
                #date_vs_datetime = "Datetime"
            else:
                data.insert(0, 'Date', data.index)
                #date_vs_datetime = "Date"
            data.insert(0, 'Ticker', ticker)

            data[f'{date_vs_datetime}'] = data[f'{date_vs_datetime}'].astype(str)

            conn = sqlite3.connect(db_name)
            cursor = conn.cursor()


            create_table_sql = f'''
            CREATE TABLE IF NOT EXISTS "{ticker}" (
                {" , ".join([f"'{col}' TEXT" if col in ['Date', 'Datetime', 'Ticker'] else
                             f"'{col}' TINYINT" if col.upper().startswith('CDL_') else
                             f"'{col}' DOUBLE" for col in data.columns])},
                UNIQUE ("Ticker", "{date_vs_datetime}")
            )
            '''
            if first_pass or _ticker_unknown_state:
                try:
                    cursor.execute(create_table_sql)
                    first_pass = False
                    _ticker_unknown_state = False
                    sleep(0.333)
                except Exception as e:
                    print(e)
                    tickers_to_dbl_chk.append(ticker)

            print("pushing data to db")
            data.to_sql(ticker, conn, if_exists='append', index=False)
            sleep(0.333)
            conn.commit()
            sleep(0.333)
            #conn.close()
        except Exception as e:
            print(e)
            print_exc()
            tickers_to_dbl_chk.append(ticker)
            #current_start_obj = end_date_obj
            #break
        finally:
            if conn is not None:
                conn.close()
                del conn
            # ToDo chk if this fixes memory leak
        try:
            del data
        except Exception as e:
            print(e)
        gc.collect()

    tickers_to_dbl_chk = list(set(tickers_to_dbl_chk))
    tickers_to_dbl_chk.sort()
    if tickers_to_dbl_chk:
        print(f"Double check following tickers for potential issues during download/update;\n{tickers_to_dbl_chk}")

    return tickers_to_dbl_chk

def apply_indicators_parallel(data, indicators, _cpus_to_use=2):
    """Apply indicators in parallel using multiprocessing."""
    with Pool(_cpus_to_use) as pool:
        results = pool.starmap(apply_indicator, [(data, indicator) for indicator in indicators])
    return results



def apply_indicator(data, indicator):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    """Helper function to apply a single indicator to the data."""
    try:
        ta_function = getattr(data.ta, indicator)
        result = ta_function(append=False)
        return result
    except Exception as e:
        print(f"Failed to apply indicator {indicator}: {e}")
        return None



def compare_list_to_directory(list_of_names, directory_path=getcwd()):
    directory_contents = listdir(directory_path)
    formated_dir_contents = [cur_db.replace(".db", "").strip() for cur_db in directory_contents]

    missing_items = set(list_of_names) - set(formated_dir_contents)
    extra_items = set(formated_dir_contents) - set(list_of_names)

    return missing_items, extra_items


def add_uniq_to_dbs(directory_path="", files_to_mod=[], date_col="Date", shard_dir='shards/nyse/'):
    if directory_path:
        directory_contents = listdir(directory_path)
    elif files_to_mod:
        directory_contents = files_to_mod
    else:
        exit()
    formatted_dir_contents = [cur_db.strip() for cur_db in directory_contents]

    dbs_to_edit = []
    for cur_file in formatted_dir_contents:
        if cur_file.endswith(".db"):
            dbs_to_edit.append(cur_file)


    sql_delay = 0.333
    num_dbs = len(dbs_to_edit)
    db_iter = 0
    for cur_db_to_edit in dbs_to_edit:

        db_iter += 1
        print(f"Working on {cur_db_to_edit} --- {db_iter} / {num_dbs}")

        ticker = cur_db_to_edit.replace(".db", "")
        cur_db_to_edit = shard_dir+cur_db_to_edit
        dedup_query = \
        f'''
        WITH cte AS (
            SELECT
                *,
                ROW_NUMBER() OVER (PARTITION BY Close, High, Low, Open, {date_col} ORDER BY {date_col}) AS row_num
            FROM "{ticker}"
        )
        DELETE FROM "{ticker}"
        WHERE EXISTS (
            SELECT 1
            FROM cte
            WHERE cte.row_num > 1
            AND "{ticker}".Close = cte.Close
            AND "{ticker}".High = cte.High
            AND "{ticker}".Low = cte.Low
            AND "{ticker}".{date_col} = cte.{date_col}

        );
        '''

        constraint_sql = \
        f'''
        ALTER TABLE "{ticker}"
        ADD CONSTRAINT unique_ticker_datetime UNIQUE ("Ticker", "{date_col}");
        '''
        conn = None
        try:
            conn = sqlite3.connect(cur_db_to_edit)
            cursor = conn.cursor()

            cursor.execute(dedup_query)
            sleep(sql_delay)
            #cursor.execute(constraint_sql)

            conn.commit()
        except Exception as e:
            print(e)
            print_exc()
        finally:
            if conn is not None:
                conn.close()



    #missing_items = set(list_of_names) - set(formated_dir_contents)
    #extra_items = set(formated_dir_contents) - set(list_of_names)

    #return missing_items, extra_items

#ToDo figure out why when running from cmd line seemingly runs without memory issue but doesn't update db
# when ran from pyscripter, actually updates db but also has memory leak

if __name__ == '__main__':
    # since print statements addup and slow down processing speed,
    # summary of warnings being a lot of pandas_ta functionality currently won't work with next major vers of pandas


    #from shards.index_tickers import dow30_tickers, nasdaq100_tickers, sap500_tickers

    from stock_info.ticker_lists.nyse_list_merged import nyse_tickers # 3676
    from stock_info.ticker_lists.nasdaq_list_merged import nasdaq_tickers # 5181
    #nyse_set = set(nyse_tickers)
    #nasdaq_set = set(nasdaq_tickers)
    #unique_nyse = nyse_set.difference(nasdaq_set) # 3676 // was ~1900 in 2025 ticker only
    #unique_nasdaq = nasdaq_set.difference(nyse_set) # 5181 // was ~3300 in 2025 ticker only
    #print(f"original nyse {len(nyse_tickers)} vs unique nyse {len(list(nyse_set))}")
    #print(f"original nasdaq {len(nasdaq_tickers)} vs unique nasdaq {len(list(nasdaq_set))}")

    nyse_dblchk = download_and_store_stock_data_minutes(nyse_tickers, get_minutes=False, _output_dir="./shards/nyse/", _skip_indicators=False, _dl_slack_days=3)
    ##nyse_dblchk = download_and_store_stock_data_minutes(missingnyse3, get_minutes=True, _output_dir="nyse_feb/", _skip_indicators=True, _dl_slack_days=5)
    #print("NYSE - Data download and storage complete with technical indicators!")

    #nyse_dblchk = download_and_store_stock_data_minutes(['GOOG'], get_minutes=False, _output_dir="./shards/nyse/", _skip_indicators=False, _dl_slack_days=3)


    #from stock_info.ticker_lists.nasdaq_list_merged import nasdaq_tickers
    #nasdaq_dblchk = download_and_store_stock_data_minutes(nasdaq_tickers, get_minutes=False, _output_dir="./shards/nasdaq/", _skip_indicators=False, _dl_slack_days=3)
    ##nasdaq_dblchk = download_and_store_stock_data_minutes(nasdaq_tickers, get_minutes=True, _output_dir="nasdaq_feb/", _skip_indicators=True, _dl_slack_days=5)
    ##print("NASDAQ - Data download and storage complete with technical indicators!")
