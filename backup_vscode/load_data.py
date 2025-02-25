from tqdm.notebook import tqdm
from itertools import islice
from file_names_for_data import unique_combined_symbol_prefix_list
from file_names_for_data import S3_CLIENT, S3_BUCKET_NAME, S3_FILE_SUFFIX
from algo_feature_engineering.data_load_utils import load_ticker_list, download_ticker_data
from my_functions import (
    calculate_daily_trend_consolidated,
    calculate_daily_ema,
    calculate_daily_hma,
    calculate_daily_rsi,
    add_initial_price_increase_z_score,
    calculate_daily_slope,
    add_initial_price_increase,
    calculate_daily_percentage_above,
    calculate_daily_cumulative_min
)
import talib

def load_and_process_data_dict(use_selected_symbols=True, selected_symbols=None):
    """
    Process ticker data for each symbol, downloading data from S3 and calculating a range of technical indicators.
    
    Parameters:
        use_selected_symbols (bool): If True, only process the symbols provided in `selected_symbols`.
        selected_symbols (list or None): List of symbol strings to process. If None, defaults to ["TSLA"].
    
    Returns:
        dict: A dictionary mapping each symbol to its processed pandas DataFrame.
    """
    if selected_symbols is None:
        selected_symbols = ["TQQQ"]
    
    data_dict = {}

    df_col_names = [
        "agg_timestamp", "open", "close", "low", "high", "volume", "slope_2", "slope_2_trend_buy_A",
        "slope_2_trend_buy_B", "slope_2_trend_buy_C", "slope_2_trend_buy_D",
        "slope_2_trend_medium_term_buy", "trend_buy", "trend_buy_slope1",
        "trend_buy_slope2", "trend_buy_slope3", "trend_slope_combined_buy",
        "trend_slope_2nd_derivative_buy", "atr_kv", "relative_atr_kv",
        "relative_atr_kv_slope_2", "slope_2_trend_buy_A_threshold",
        "slope_2_trend_buy_B_threshold", "slope_2_trend_buy_C_threshold",
        "slope_2_trend_buy_D_threshold", "slope_2_trend_medium_term_buy_threshold",
        "trend_buy_threshold", "slope_2_trend_buy_A_short_term",
        "slope_2_trend_buy_B_short_term", "slope_2_trend_medium_term_buy_short_term",
        "trend_buy_short_term", "jma_short_term", "price_slope_short_term",
        "rolling_mean_price_slope_short_term", "rolling_std_price_slope_short_term",
        "z_score_price_slope_short_term", "buy", "trend_buy_autocorr_150_1",
        "close_autocorr_150_1", "price_slope_hma_long_term_250",
        "hma_long_term_slope_250", "hma_long_term_slope_percent_positive_250",
        "hma_long_term_slope_percent_positive_250_autocorr_150_1",
        "price_slope_hma_long_term_150", "hma_long_term_slope_150",
        "hma_long_term_slope_150_autocorr_150_1", "trend_buy_150",
        "trend_buy_150_autocorr_150_1"
    ]

    # Ensure the symbol list is in list form
    symbol_prefix_list = list(unique_combined_symbol_prefix_list)

    if use_selected_symbols:
        # Filter to include only selected symbols
        top_symbols = [
            (symbol, s3_prefix)
            for symbol, s3_prefix in symbol_prefix_list
            if symbol in selected_symbols
        ]
    else:
        # Use all symbols
        top_symbols = symbol_prefix_list

    for symbol, s3_prefix in tqdm(top_symbols, desc="Processing symbols"):
        print(f"\nProcessing Details:")
        print(f"S3 Client: {S3_CLIENT}")
        print(f"S3 Bucket Name: {S3_BUCKET_NAME}")
        print(f"S3 Prefix: {s3_prefix}")
        print(f"S3 File Suffix: {S3_FILE_SUFFIX}")
        print(f"Symbol: {symbol}")

        data = download_ticker_data(
            S3_CLIENT,
            S3_BUCKET_NAME,
            s3_prefix,
            S3_FILE_SUFFIX,
            symbol,
            df_col_names=df_col_names
        )
        data_dict[symbol] = data
        if data.empty:
            print(f"Data for {symbol} is empty or could not be loaded. Skipping.")
            continue
        min_date = data.index.min()
        max_date = data.index.max()
        print(f"Symbol: {symbol}, Min Date: {min_date}, Max Date: {max_date}")

        print(f"\nCalculating technical indicators for {symbol}...")

        # Use the dataframe reference for cleaner code
        ticker_df_adjusted = data_dict[symbol]

        # Day trend calculations
        ticker_df_adjusted['day_trend_3'] = calculate_daily_trend_consolidated(ticker_df_adjusted, 'close', window_size=3, hard_reset=False)
        ticker_df_adjusted['day_trend_5'] = calculate_daily_trend_consolidated(ticker_df_adjusted, 'close', window_size=5, hard_reset=False)
        ticker_df_adjusted['day_trend_7'] = calculate_daily_trend_consolidated(ticker_df_adjusted, 'close', window_size=7, hard_reset=False)
        ticker_df_adjusted['day_trend_15'] = calculate_daily_trend_consolidated(ticker_df_adjusted, 'close', window_size=15, hard_reset=False)
        ticker_df_adjusted['day_trend_20'] = calculate_daily_trend_consolidated(ticker_df_adjusted, 'close', window_size=20, hard_reset=False)
        ticker_df_adjusted['day_trend_25'] = calculate_daily_trend_consolidated(ticker_df_adjusted, 'close', window_size=25, hard_reset=False)
        ticker_df_adjusted['day_trend_400'] = calculate_daily_trend_consolidated(ticker_df_adjusted, 'close', window_size=180, hard_reset=False)
        ticker_df_adjusted['day_trend_400_v2'] = calculate_daily_trend_consolidated(ticker_df_adjusted, 'close', window_size=180, hard_reset=False)

        # Day trend EMA calculations
        ticker_df_adjusted['day_trend_15_ema_10'] = calculate_daily_ema(ticker_df_adjusted, 'day_trend_15', window_size=10)
        ticker_df_adjusted['day_trend_15_ema_15'] = calculate_daily_ema(ticker_df_adjusted, 'day_trend_15', window_size=15)
        ticker_df_adjusted['day_trend_15_ema_20'] = calculate_daily_ema(ticker_df_adjusted, 'day_trend_15', window_size=20)
        ticker_df_adjusted['day_trend_20_ema_15'] = calculate_daily_ema(ticker_df_adjusted, 'day_trend_20', window_size=15)
        ticker_df_adjusted['day_trend_25_ema_15'] = calculate_daily_ema(ticker_df_adjusted, 'day_trend_25', window_size=15)
        ticker_df_adjusted['day_trend_400_ema_10'] = calculate_daily_ema(ticker_df_adjusted, 'day_trend_400', window_size=10)
        ticker_df_adjusted['day_trend_400_ema_10_v2'] = calculate_daily_ema(ticker_df_adjusted, 'day_trend_400_v2', window_size=10)

        # HMA calculations
        ticker_df_adjusted['day_trend_15_hma_15'] = calculate_daily_hma(ticker_df_adjusted, 'day_trend_15', window_size=15)
        ticker_df_adjusted['day_trend_15_hma_20'] = calculate_daily_hma(ticker_df_adjusted, 'day_trend_15', window_size=20)
        ticker_df_adjusted['day_trend_15_hma_25'] = calculate_daily_hma(ticker_df_adjusted, 'day_trend_15', window_size=25)
        ticker_df_adjusted['day_trend_15_hma_30'] = calculate_daily_hma(ticker_df_adjusted, 'day_trend_15', window_size=30)
        ticker_df_adjusted['day_trend_20_hma_25'] = calculate_daily_hma(ticker_df_adjusted, 'day_trend_20', window_size=25)
        ticker_df_adjusted['day_trend_25_hma_25'] = calculate_daily_hma(ticker_df_adjusted, 'day_trend_25', window_size=25)

        # Slope calculations
        ticker_df_adjusted['day_trend_15_ema_15_slope'] = talib.LINEARREG_SLOPE(ticker_df_adjusted['day_trend_15_ema_15'], timeperiod=2)
        ticker_df_adjusted['day_trend_15_ema_20_slope'] = talib.LINEARREG_SLOPE(ticker_df_adjusted['day_trend_15_ema_20'], timeperiod=2)
        ticker_df_adjusted['day_trend_25_hma_25_slope'] = talib.LINEARREG_SLOPE(ticker_df_adjusted['day_trend_25_hma_25'], timeperiod=2)
        ticker_df_adjusted['day_trend_400_ema_10_v2_slope'] = talib.LINEARREG_SLOPE(ticker_df_adjusted['day_trend_400_ema_10_v2'], timeperiod=30)

        # EMA of slopes
        ticker_df_adjusted['day_trend_15_ema_20_slope_ema_20'] = calculate_daily_ema(ticker_df_adjusted, 'day_trend_15_ema_20_slope', window_size=10)
        ticker_df_adjusted['day_trend_25_hma_25_slope_hma_10'] = calculate_daily_hma(ticker_df_adjusted, 'day_trend_25_hma_25_slope', window_size=10)

        # Second derivatives
        ticker_df_adjusted['day_trend_15_ema_20_2nd_deriv'] = talib.LINEARREG_SLOPE(ticker_df_adjusted['day_trend_15_ema_20_slope_ema_20'], timeperiod=2)
        ticker_df_adjusted['day_trend_25_hma_25_2nd_deriv'] = talib.LINEARREG_SLOPE(ticker_df_adjusted['day_trend_25_hma_25_slope_hma_10'], timeperiod=2)
        ticker_df_adjusted['day_trend_15_ema_20_2nd_deriv_ema_10'] = calculate_daily_ema(ticker_df_adjusted, 'day_trend_15_ema_20_2nd_deriv', window_size=20)

        # Additional indicators
        ticker_df_adjusted['day_rsi_20'] = calculate_daily_rsi(df=ticker_df_adjusted, column_name='close', window_size=25)
        ticker_df_adjusted = add_initial_price_increase_z_score(ticker_df_adjusted, price_col='close', window=50)

        # Positive Thresholds
        ticker_df_adjusted['initial_price_change_zscore_gt_1'] = (ticker_df_adjusted['initial_price_increase_z_score'] > 1).astype(int)
        ticker_df_adjusted['initial_price_change_zscore_gt_2'] = (ticker_df_adjusted['initial_price_increase_z_score'] > 2).astype(int)
        ticker_df_adjusted['initial_price_change_zscore_gt_3'] = (ticker_df_adjusted['initial_price_increase_z_score'] > 3).astype(int)

        # Negative Thresholds
        ticker_df_adjusted['initial_price_change_zscore_lt_-1'] = (ticker_df_adjusted['initial_price_increase_z_score'] < -1).astype(int)
        ticker_df_adjusted['initial_price_change_zscore_lt_-2'] = (ticker_df_adjusted['initial_price_increase_z_score'] < -2).astype(int)
        ticker_df_adjusted['initial_price_change_zscore_lt_-3'] = (ticker_df_adjusted['initial_price_increase_z_score'] < -3).astype(int)

        ticker_df_adjusted['day_slope_15'] = calculate_daily_slope(ticker_df_adjusted, 'close', window_size=15)
        ticker_df_adjusted['day_slope_20'] = calculate_daily_slope(ticker_df_adjusted, 'close', window_size=20)
        ticker_df_adjusted['day_slope_25'] = calculate_daily_slope(ticker_df_adjusted, 'close', window_size=25)
        ticker_df_adjusted = add_initial_price_increase(ticker_df_adjusted, price_col='close')

        # Above 0.5 calculations
        ticker_df_adjusted['day_trend_15_above_0.5'] = calculate_daily_percentage_above(ticker_df_adjusted, 'day_trend_15', window_size=400)
        ticker_df_adjusted['day_trend_20_above_0.5'] = calculate_daily_percentage_above(ticker_df_adjusted, 'day_trend_20', window_size=400)
        ticker_df_adjusted['day_trend_25_above_0.5'] = calculate_daily_percentage_above(ticker_df_adjusted, 'day_trend_25', window_size=400)
        ticker_df_adjusted['day_trend_400_v2_above_0.5'] = calculate_daily_percentage_above(ticker_df_adjusted, 'day_trend_400_v2', window_size=400)
        ticker_df_adjusted['day_trend_15_ema_15_above_0.5'] = calculate_daily_percentage_above(ticker_df_adjusted, 'day_trend_15_ema_15', window_size=400)
        ticker_df_adjusted['day_trend_20_ema_15_above_0.5'] = calculate_daily_percentage_above(ticker_df_adjusted, 'day_trend_20_ema_15', window_size=400)
        ticker_df_adjusted['day_trend_25_ema_15_above_0.5'] = calculate_daily_percentage_above(ticker_df_adjusted, 'day_trend_25_ema_15', window_size=400)
        ticker_df_adjusted['day_trend_400_ema_10_v2_above_0.5'] = calculate_daily_percentage_above(ticker_df_adjusted, 'day_trend_400_ema_10_v2', window_size=400)

        # Difference calculations
        ticker_df_adjusted['day_trend_25_hma_25_minus_400_ema_10_v2'] = (
            ticker_df_adjusted['day_trend_25_hma_25'] - 
            ticker_df_adjusted['day_trend_400_ema_10_v2']
        )

        # Slope of difference
        ticker_df_adjusted['day_trend_25_hma_25_minus_400_ema_10_v2_slope'] = talib.LINEARREG_SLOPE(
            ticker_df_adjusted['day_trend_25_hma_25_minus_400_ema_10_v2'], 
            timeperiod=2
        )

        # HMA of slope
        ticker_df_adjusted['day_trend_25_hma_25_minus_400_ema_10_v2_slope_hma'] = calculate_daily_hma(
            ticker_df_adjusted, 
            'day_trend_25_hma_25_minus_400_ema_10_v2_slope', 
            window_size=12
        )

        # Second derivative of difference
        ticker_df_adjusted['day_trend_25_hma_25_minus_400_ema_10_v2_2nd_deriv'] = talib.LINEARREG_SLOPE(
            ticker_df_adjusted['day_trend_25_hma_25_minus_400_ema_10_v2_slope_hma'], 
            timeperiod=2
        )

        # Daily minimum value
        ticker_df_adjusted['daily_min_value'] = calculate_daily_cumulative_min(
            ticker_df_adjusted,
            'day_trend_25_hma_25_minus_400_ema_10_v2'
        )

        # Update the dictionary with the processed dataframe
        data_dict[symbol] = ticker_df_adjusted

        print(f"Completed processing for {symbol}")
    
    return data_dict
