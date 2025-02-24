import numpy as np
import pandas as pd
import talib
from numba import njit
from algo_feature_engineering.features.ma import (
    moving_average_features_normalized,
    moving_average_features_slopes,
    moving_average_features_percent_diff,
    calculate_hma
)



def compute_hma_trend_consistency(time_series, period=20):
    # Convert input to float64 numpy array
    time_series = np.array(time_series, dtype=np.float64)
    
    # Calculate HMA
    hma = calculate_hma(time_series, period=period)
    
    # Calculate trend direction (1 for positive, -1 for negative)
    trend_direction = np.where(np.diff(hma, prepend=hma[0]) > 0.0, 1, -1)
    
    # Calculate time spent in positive trend
    time_in_positive_trend = np.sum(trend_direction == 1)
    total_time = len(trend_direction)
    
    # Calculate consistency
    positive_trend_consistency = time_in_positive_trend / total_time if total_time > 0 else 0.0
    
    return positive_trend_consistency

def calculate_trend_consistency_for_decile(df_with_signals, decile_results):
    """
    Calculate HMA trend consistency for all symbols in a decile
    
    Parameters:
    df_with_signals (pd.DataFrame): Raw dataframe containing close_raw prices
    decile_results (BacktestResults): Results object containing symbol metrics
    
    Returns:
    dict: Dictionary mapping symbols to their trend consistency values
    """
    trend_consistencies = {}
    
    for symbol in decile_results.symbol_metrics['symbol']:
        # Get the symbol's data from the signals dataframe
        symbol_data = df_with_signals[df_with_signals['symbol'] == symbol]
        if not symbol_data.empty:
            # Calculate trend consistency from the close prices
            consistency = compute_hma_trend_consistency(symbol_data['close_raw'].values)
            trend_consistencies[symbol] = consistency
        else:
            trend_consistencies[symbol] = 0.0  # Default value if no data for this symbol
            
    return trend_consistencies


import numpy as np
import pandas as pd

def calc_intraday_atr(df, col_high='high_raw', col_low='low_raw', col_close='close_raw', 
                      atr_period=14, debug=False):
    """
    Calculate an "intraday-only" ATR, ignoring overnight gaps for the first bar of each day.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns with the high, low, and close data.
        The index should be a datetime-like index (e.g. pd.DatetimeIndex).
    col_high : str
        Name of the column that contains the intraday high.
    col_low : str
        Name of the column that contains the intraday low.
    col_close : str
        Name of the column that contains the intraday close.
    atr_period : int
        The period for the ATR calculation (default = 14).
    debug : bool
        If True, prints debugging statements and a small table for the first 10 rows.

    Returns
    -------
    pd.Series
        A Series of intraday ATR values, indexed like df.
    """

    # 1) Ensure columns are present and cast to float64
    high = df[col_high].astype(np.float64)
    low  = df[col_low].astype(np.float64)
    close = df[col_close].astype(np.float64)
    
    # 2) Convert the index into a Series so we can do a row-wise shift
    #    (instead of attempting a frequency-based shift on a DatetimeIndex).
    df_dates = pd.Series(df.index.normalize(), index=df.index, name='date')

    # 3) Create shifted versions to reference previous row's data
    prev_close = close.shift(1)
    prev_date = df_dates.shift(1)
    
    # 4) Identify rows that are the first bar of a new day
    #    (i.e., date changes compared to the previous row)
    is_new_day = (df_dates != prev_date)

    # 5) Calculate the standard True Range
    tr_standard = np.maximum(
        high - low,
        np.maximum(
            (high - prev_close).abs(),
            (low - prev_close).abs()
        )
    )

    # 6) Calculate the intraday-only True Range
    #    - If it's a new day, just take (high - low)
    #    - Otherwise, use the standard TR
    tr_intraday = np.where(is_new_day, (high - low), tr_standard)

    # Convert TR to a Series for convenience
    df_tr = pd.Series(tr_intraday, index=df.index, name='TR_intraday')
    
    # 7) Perform the Wilder-like ATR smoothing:
    #    ATR_t = ATR_(t-1) + alpha * (TR_t - ATR_(t-1)), where alpha = 1 / atr_period
    alpha = 1.0 / atr_period
    atr = pd.Series(np.nan, index=df.index, name='ATR_intraday')
    
    # Initialize the first ATR value
    atr.iloc[0] = df_tr.iloc[0]

    # Compute ATR iteratively
    for i in range(1, len(df_tr)):
        atr.iloc[i] = atr.iloc[i-1] + alpha * (df_tr.iloc[i] - atr.iloc[i-1])

    # 8) If debug is True, show detailed steps for the first 10 rows
    if debug:
        print("=== DEBUG MODE ENABLED ===")
        print("Below is a breakdown of the calculations for the first 10 rows.")

        # Build a temporary DataFrame to show intermediate columns and final ATR
        debug_df = pd.DataFrame({
            'Date': df_dates,
            'Is_New_Day': is_new_day,
            'High': high,
            'Low': low,
            'Prev_Close': prev_close,
            'High-Low': high - low,
            'abs(High - PrevClose)': (high - prev_close).abs(),
            'abs(Low - PrevClose)': (low - prev_close).abs(),
            'TR_Standard': tr_standard,
            'TR_Intraday': df_tr,
            'ATR_Intraday': atr
        })

        # Print first 10 rows of this table
        print(debug_df.head(10).to_string())

        print("\nExplanation of key columns:")
        print("- Date / Is_New_Day: Used to identify the first bar of each day.")
        print("- High, Low, Prev_Close: Inputs to the TR calculation.")
        print("- High-Low: Baseline range for the bar.")
        print("- abs(High - PrevClose) / abs(Low - PrevClose): Used to account for gaps.")
        print("- TR_Standard: Standard True Range (max of above three).")
        print("- TR_Intraday: Same as TR_Standard except on new-day bars, we ignore the gap.")
        print("- ATR_Intraday: Wilder-smoothed ATR based on TR_Intraday.")
        print("=======================================\n")

    return atr








def calculate_daily_trend_consolidated(
    df, 
    column_name, 
    window_size, 
    cutoff_time_str="06:45:00",
    hard_reset=False
):
    """
    Consolidated daily-reset trend indicator with two modes:
    
    1) Standard mode (hard_reset=False) – mimics `calculate_daily_trend`:
       - Does NOT force the first row of each day to be NaN.
       - Replaces all non-finite daily_trend values with 0.

    2) "Hard Reset" mode (hard_reset=True) – mimics `calculate_daily_trend_v2`:
       - The first row of each day is forced to be NaN (no previous day's data).
       - Only the second row onward replaces any non-finite (NaN/Inf) with 0.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame with a DateTimeIndex and at least one price column.
    column_name : str
        The name of the column to use for calculation (e.g., 'close_raw').
    window_size : int
        The size of the rolling window for computing the trend.
    cutoff_time_str : str or None, default "06:45:00"
        If provided, any row whose time is <= that cutoff will be set to NaN.
        If None, no time-based nullification occurs.
    hard_reset : bool, default False
        Toggle between two behaviors:
          - False: (like `calculate_daily_trend`)
              * Do not force the first row of the day to be NaN.
              * Replace all inf/-inf/NaN with 0.
          - True: (like `calculate_daily_trend_v2`)
              * Force the first row of the day to be NaN.
              * For each day, only rows after the first replace non-finite with 0.

    Returns
    -------
    trend_series : pd.Series
        A Series with the same index as `df`, containing the trend indicator.
    """
    # Ensure data is sorted by time
    df = df.sort_index()

    # Compute price changes
    changes = df[column_name].astype(float).diff()

    # Precompute absolute and positive changes
    abs_changes = changes.abs()
    pos_changes = changes.clip(lower=0)

    # Factorize the dates to identify each day's rows
    day_ids, unique_days = pd.factorize(df.index.normalize())

    # Prepare an array to hold the results
    result = np.zeros(len(df), dtype=float)

    # Process each day separately
    for day_id in range(len(unique_days)):
        # Identify the rows for this specific day
        mask = (day_ids == day_id)
        if not np.any(mask):
            continue

        # Slicing the arrays for this day
        day_abs = abs_changes.values[mask]
        day_pos = pos_changes.values[mask]

        # If in "hard_reset" mode, force the first row of each day to NaN
        if hard_reset:
            day_abs[0] = np.nan
            day_pos[0] = np.nan

        # Compute cumulative sums for the day
        # Convert any NaN to 0 for the sake of cumsum
        cum_abs = np.cumsum(np.nan_to_num(day_abs))
        cum_pos = np.cumsum(np.nan_to_num(day_pos))

        # Compute rolling sums using cumsum differences
        shifted_abs = np.roll(cum_abs, window_size)
        shifted_pos = np.roll(cum_pos, window_size)

        # Zero out the rolled values for the first window_size entries
        shifted_abs[:window_size] = 0
        shifted_pos[:window_size] = 0

        abs_rolling_sum = cum_abs - shifted_abs
        pos_rolling_sum = cum_pos - shifted_pos

        # Compute the daily trend; guard against division / NaN / inf
        with np.errstate(divide='ignore', invalid='ignore'):
            daily_trend = pos_rolling_sum / abs_rolling_sum

        # Different handling depending on the mode
        if hard_reset:
            # Replace ±inf with NaN for the entire array
            daily_trend = np.where(np.isfinite(daily_trend), daily_trend, np.nan)
            # For rows after the first, replace non-finite with 0
            if len(daily_trend) > 1:
                daily_trend[1:] = np.where(np.isfinite(daily_trend[1:]), 
                                           daily_trend[1:], 0)
        else:
            # Replace all non-finite values with 0 immediately
            daily_trend[~np.isfinite(daily_trend)] = 0

        # Store results in the master array
        result[mask] = daily_trend

    # Create a Series
    trend_series = pd.Series(result, index=df.index, name='trend')

    # If a cutoff time is provided, nullify (set to NaN) any row whose time is <= cutoff
    if cutoff_time_str is not None:
        cutoff_time = pd.to_datetime(cutoff_time_str).time()
        trend_series.loc[trend_series.index.time <= cutoff_time] = np.nan

    return trend_series



def calculate_daily_slope(df, column_name, window_size):
    """
    Calculate a daily-reset rolling slope indicator based on the line of best fit 
    over the last 'window_size' points of the day's price data.

    Parameters:
    - df: pandas.DataFrame with a DateTimeIndex and at least a price column.
    - column_name: str, the name of the column containing the price data (e.g. 'close_raw').
    - window_size: int, the size of the rolling window.

    Returns:
    - A pandas.Series with the same index as df, containing the slope values.
    """
    # Ensure data is sorted by time
    df = df.sort_index()

    # Extract the price array
    prices = df[column_name].astype(float).values

    # Factorize the dates to identify each day's rows
    day_ids, unique_days = pd.factorize(df.index.normalize())

    # Prepare an array for the slope results
    result = np.zeros(len(df), dtype=float)

    for day_id in range(len(unique_days)):
        # Mask for the current day
        mask = (day_ids == day_id)
        if not np.any(mask):
            continue

        # Extract this day's prices and indices
        day_prices = prices[mask]
        n_day = len(day_prices)
        day_indices = np.where(mask)[0]  # Get the actual indices in the original DataFrame

        # Create an array of x-values for this day: 0, 1, 2, ..., n_day-1
        x = np.arange(n_day)

        # Precompute cumulative sums for x, x², y, and xy
        cum_x = np.cumsum(x)
        cum_x2 = np.cumsum(x*x)
        cum_y = np.cumsum(day_prices)
        cum_xy = np.cumsum(x*day_prices)

        # Compute slope for each index i in the day
        for i in range(n_day):
            # Determine the start of the rolling window
            start = max(0, i - window_size + 1)
            # Number of points in the current window
            N = i - start + 1

            # Compute sums over the window [start, i]
            sum_x = cum_x[i] - (cum_x[start-1] if start > 0 else 0)
            sum_y = cum_y[i] - (cum_y[start-1] if start > 0 else 0)
            sum_x2_window = cum_x2[i] - (cum_x2[start-1] if start > 0 else 0)
            sum_xy_window = cum_xy[i] - (cum_xy[start-1] if start > 0 else 0)

            # If we only have one data point, slope = 0
            if N == 1:
                slope = 0.0
            else:
                # Calculate the denominator
                denom = (N * sum_x2_window) - (sum_x**2)
                if denom == 0:
                    slope = 0.0
                else:
                    slope = ((N * sum_xy_window) - (sum_x * sum_y)) / denom

            # Assign the slope value directly using the day_indices
            result[day_indices[i]] = slope

    # Create a Series for the results with the same index
    slope_series = pd.Series(result, index=df.index, name='slope')
    return slope_series


import numpy as np
import pandas as pd

def add_initial_price_increase(df, price_col='close_raw'):
    # Ensure DataFrame is time-sorted
    df = df.sort_index()

    # Extract prices and factorize days
    prices = df[price_col].values
    day_ids, unique_days = pd.factorize(df.index.normalize())

    n = len(df)
    initial_price_increase = np.zeros(n, dtype=int)

    # Find boundaries where day changes
    # day changes occur where day_ids[i] != day_ids[i-1]
    changes = np.where(np.diff(day_ids) != 0)[0] + 1

    # The start of the first day is at index 0
    day_starts = np.concatenate(([0], changes))
    # The end of each day is right before the next start, and the last day ends at n-1
    day_ends = np.concatenate((changes - 1, [n - 1]))

    # Loop through each day and set initial_price_increase for that day
    # If there's no previous day, set to 0 for the entire day
    for i, day_id in enumerate(unique_days):
        start_idx = day_starts[i]
        end_idx = day_ends[i]

        if i == 0:
            # First day in the dataset, no previous day to compare
            increase = 0
        else:
            # Compare the first price of this day with the last price of the previous day
            prev_day_end_idx = day_ends[i - 1]
            increase = 1 if prices[start_idx] > prices[prev_day_end_idx] else 0

        # Assign to all rows of this day
        initial_price_increase[start_idx:end_idx+1] = increase

    # Add the result to the DataFrame
    df['initial_price_increase'] = initial_price_increase
    return df


def add_initial_price_increase_z_score(df, price_col='close_raw', window=50):
    # Ensure DataFrame is time-sorted
    df = df.sort_index()

    # Extract prices and factorize days
    prices = df[price_col].values
    day_ids, unique_days = pd.factorize(df.index.normalize())

    n = len(df)
    z_scores = np.zeros(n, dtype=float)

    # Identify daily boundaries
    changes = np.where(np.diff(day_ids) != 0)[0] + 1
    day_starts = np.concatenate(([0], changes))
    day_ends = np.concatenate((changes - 1, [n - 1]))

    # We'll store daily percent changes for each day
    # This will have length = number_of_unique_days
    daily_changes = np.zeros(len(unique_days), dtype=float)

    # Compute daily changes
    for i, day_id in enumerate(unique_days):
        start_idx = day_starts[i]
        end_idx = day_ends[i]

        if i == 0:
            # First day: no previous day
            daily_changes[i] = 0.0
        else:
            prev_day_end_idx = day_ends[i - 1]
            # Calculate percent change
            prev_day_price = prices[prev_day_end_idx]
            current_day_price = prices[start_idx]
            if prev_day_price != 0:
                daily_changes[i] = (current_day_price - prev_day_price) / prev_day_price
            else:
                # In case prev_day_price is zero (unexpected but safe check)
                daily_changes[i] = 0.0

    # Now compute z-scores for each day based on the last 'window' days of daily_changes
    for i, day_id in enumerate(unique_days):
        if i == 0:
            # First day, no history
            z_val = 0.0
        else:
            # Determine the slice of recent history
            start_hist = max(0, i - window)
            recent_changes = daily_changes[start_hist:i]  # exclude today's change itself
            if len(recent_changes) == 0:
                # If no history, z-score defaults to 0
                z_val = 0.0
            else:
                mean_val = recent_changes.mean()
                std_val = recent_changes.std()
                if std_val == 0:
                    z_val = 0.0
                else:
                    z_val = (daily_changes[i] - mean_val) / std_val

        # Assign z_val to all rows of this day
        start_idx = day_starts[i]
        end_idx = day_ends[i]
        z_scores[start_idx:end_idx+1] = z_val

    # Add the result to the DataFrame
    df['initial_price_increase_z_score'] = z_scores
    return df


def calculate_daily_ema(df, column_name, window_size, reset_flag=1):
    """
    Calculate a daily-reset EMA for a given column, with partial window handling at the start of each day.
    The EMA "resets" at the start of each trading day and does not use data from previous days.
    Handles NaN values by skipping them in the calculation.

    If reset_flag == 1:
        The very first valid row of each day is set to the raw price 
        (i.e. no averaging or EMA is applied to that first bar).
    """
    # Ensure data is sorted by time
    df = df.sort_index()
    prices = df[column_name].astype(float).values

    # Factorize dates to identify each day's rows
    day_ids, unique_days = pd.factorize(df.index.normalize())

    # Prepare result array
    result = np.full(len(df), np.nan, dtype=float)
    alpha = 2.0 / (window_size + 1.0)  # standard EMA alpha

    # Process each day separately
    for day_id in range(len(unique_days)):
        mask = (day_ids == day_id)
        if not np.any(mask):
            continue
        
        day_indices = np.where(mask)[0]    # array of positions in df corresponding to this day
        day_prices = prices[mask]         # the actual price values for this day
        
        n_day = len(day_prices)
        if n_day == 0:
            continue
        
        # Find first non-NaN price within the day
        valid_prices_mask = ~np.isnan(day_prices)
        if not np.any(valid_prices_mask):
            # entire day is NaNs -> skip
            continue
        
        first_valid_idx_local = np.where(valid_prices_mask)[0][0]  # index *within this day_prices array*
        first_valid_global = day_indices[first_valid_idx_local]    # index in the original df
        
        # -----------------------------------------------------------
        # If reset_flag == 1, set the very first valid row to the raw price
        # -----------------------------------------------------------
        if reset_flag == 1:
            result[first_valid_global] = day_prices[first_valid_idx_local]
        
        # Now proceed to fill in the rest of the day's EMA
        # We start from the next row after 'first_valid_idx_local'
        for i_local in range(first_valid_idx_local + 1, n_day):
            # Corresponding global index
            i_global = day_indices[i_local]
            if np.isnan(day_prices[i_local]):
                # skip NaNs
                continue
                
            prev_global = day_indices[i_local - 1]
            prev_ema = result[prev_global]
            current_price = day_prices[i_local]
            
            # If there's no prior EMA (NaN), we do the fallback
            # unless reset_flag already set it for the first bar
            if np.isnan(prev_ema):
                # Option A: Simple average of current & previous price
                #   (this is your existing “fallback” logic)
                result[i_global] = (day_prices[i_local - 1] + current_price) / 2.0
            else:
                # Standard EMA update
                result[i_global] = alpha * current_price + (1 - alpha) * prev_ema

    # Return as a Series
    return pd.Series(result, index=df.index, name='daily_ema')



def weighted_moving_average(values, length):
    """
    Compute the WMA of the last 'length' non-NaN values.
    If there are fewer than 'length' valid (non-NaN) values, use what is available.
    'values' is a 1D numpy array.
    """
    # 1) Filter out NaNs first
    valid_values = values[~np.isnan(values)]
    n = len(valid_values)
    if n == 0:
        return np.nan  # No valid data
    
    # 2) Determine how many valid points to use
    use_length = min(length, n)
    
    # 3) Weights = 1, 2, ..., use_length
    w = np.arange(1, use_length + 1, dtype=float)
    
    # 4) Get the last 'use_length' valid points
    subset = valid_values[-use_length:]
    
    # 5) Weighted sum / sum of weights
    return np.sum(subset * w) / np.sum(w)

import math
def calculate_daily_hma(df, column_name, window_size):
    """
    Calculate a "daily-reset" Hull Moving Average (HMA) for each day independently,
    ignoring NaN values within any window slice.

    The procedure:
      1) For each row (within each day), compute:
         - WMA of the last 'window_size/2' points
         - WMA of the last 'window_size' points
         => intermediate = 2 * WMA_half - WMA_full

      2) For each row, compute the WMA of all intermediate values up to that row
         with length = sqrt(window_size).

      3) The result is stored in 'daily_hma'.

    The first few rows of the day might still be NaN if not enough data has accumulated.
    """
    # Ensure we have a sorted DataFrame
    df = df.sort_index()

    # Extract the price array
    prices = df[column_name].astype(float).values

    # Factorize by day
    day_ids, unique_days = pd.factorize(df.index.normalize())

    # This will hold our final result
    result = np.full(len(df), np.nan, dtype=float)

    # For the HMA:
    half_length = max(1, int(round(window_size / 2.0)))
    sqrt_length = max(1, int(math.sqrt(window_size)))

    # Process each day separately
    for day_id in range(len(unique_days)):
        # Identify which rows belong to this day
        mask = (day_ids == day_id)
        if not np.any(mask):
            continue

        # Indices and price data for just this day
        day_indices = np.where(mask)[0]
        day_prices = prices[mask]
        n_day = len(day_prices)

        # Intermediate WMA values (2*WMA_half - WMA_full) for each row
        intermediate_values = []

        for i in range(n_day):
            # Slice the prices up to the current row (inclusive)
            current_slice = day_prices[: i + 1]

            # Compute WMA of half-length window
            wma_half = weighted_moving_average(current_slice, half_length)
            # Compute WMA of full window_size
            wma_full = weighted_moving_average(current_slice, window_size)

            # If either is NaN, we don't have enough valid data yet
            if np.isnan(wma_half) or np.isnan(wma_full):
                intermediate_values.append(np.nan)
                result[day_indices[i]] = np.nan
                continue

            # 2 * WMA(half) - WMA(full)
            intermediate = 2 * wma_half - wma_full
            intermediate_values.append(intermediate)

            # Now compute WMA of the intermediate_values so far,
            # with window length = sqrt_length
            intermediate_array = np.array(intermediate_values[: i + 1])
            hma_value = weighted_moving_average(intermediate_array, sqrt_length)

            result[day_indices[i]] = hma_value

    return pd.Series(result, index=df.index, name='daily_hma')




import numpy as np
import pandas as pd

def calculate_daily_rsi(df, column_name, window_size):
    """
    Calculate a daily-reset RSI for a given column, with partial window handling at the start of each day.
    The RSI "resets" at the start of each trading day and does not use data from previous days.

    Parameters:
    - df: pandas.DataFrame with a DateTimeIndex and at least a price column.
    - column_name: str, the name of the price column (e.g. 'close_raw').
    - window_size: int, the size of the window for RSI calculation.

    Returns:
    - A pandas.Series with the same index as df containing the daily-reset RSI.
      Early in the day when insufficient data is available, RSI is computed using whatever data is available.
    """
    # Ensure data is time-sorted
    df = df.sort_index()

    prices = df[column_name].astype(float).values
    # Compute price changes
    changes = np.diff(prices, prepend=np.nan)  # first row of each day will be NaN anyway

    # Gains and Losses
    gains = np.where(changes > 0, changes, 0.0)
    losses = np.where(changes < 0, -changes, 0.0)  # losses as positive numbers

    # Factorize the dates to identify each day's rows
    day_ids, unique_days = pd.factorize(df.index.normalize())

    result = np.full(len(df), np.nan, dtype=float)

    for day_id in range(len(unique_days)):
        mask = (day_ids == day_id)
        if not np.any(mask):
            continue

        day_indices = np.where(mask)[0]
        day_gains = gains[mask]
        day_losses = losses[mask]

        n_day = len(day_gains)
        if n_day == 0:
            continue

        # Compute cumulative sums for gains and losses for this day
        cum_gains = np.cumsum(np.nan_to_num(day_gains))
        cum_losses = np.cumsum(np.nan_to_num(day_losses))

        # For each index i in the day, compute RSI based on up to i bars
        for i in range(n_day):
            # Determine how many bars to use (partial window at start)
            start = max(0, i - window_size + 1)
            length = i - start + 1  # actual number of bars in the window

            # Rolling sums for gains and losses
            sum_gains = cum_gains[i] - (cum_gains[start-1] if start > 0 else 0.0)
            sum_losses = cum_losses[i] - (cum_losses[start-1] if start > 0 else 0.0)

            # Compute average gain and average loss
            avg_gain = sum_gains / length
            avg_loss = sum_losses / length

            if avg_loss == 0:
                # If no losses, RSI = 100
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100.0 - (100.0 / (1.0 + rs))

            result[day_indices[i]] = rsi

    return pd.Series(result, index=df.index, name='rsi')



def calculate_daily_percentage_above(df, column_name, window_size, cutoff_time_str="06:45:00"):
    """
    Similar to calculate_daily_percentage_above, but ignores (drops) rows where the timestamp
    is before the specified cutoff_time_str. For each day, calculates a rolling window of size 
    `window_size` to find the percentage of values > 0.5 (ignoring NaNs in the denominator).
    The calculation resets daily.
    
    Parameters:
    - df: pandas.DataFrame with a DateTimeIndex
    - column_name: str, the target column to evaluate against the 0.5 threshold
    - window_size: int, the rolling window size (per day)
    - cutoff_time_str: str (e.g., "09:45:00"), the time before which rows 
      will be filtered out. Defaults to "09:45:00". Pass None to disable filtering.
    
    Returns:
    - A pandas.Series with the same index (minus any filtered rows) containing
      the rolling percentage of values > 0.5, ignoring NaNs.
    """
    # 1) Sort the DataFrame by index to ensure chronological order
    df = df.sort_index()

    # 2) If a cutoff time string is provided, parse it and filter out rows before it
    if cutoff_time_str is not None:
        cutoff_time = pd.to_datetime(cutoff_time_str).time()
        df = df[df.index.time >= cutoff_time].copy()

    # 3) Create two float Series for the rolling ratio logic:
    #    - cond_series = 1.0 if value > 0.5 and not NaN, else 0.0
    #    - valid_series = 1.0 if value is not NaN, else 0.0
    cond_series = (df[column_name].gt(0.5) & df[column_name].notna()).astype(float)
    valid_series = df[column_name].notna().astype(float)

    # 4) Group the data by normalized date to reset calculations each day
    grouped = df.groupby(df.index.normalize())

    def _rolling_ratio_for_one_day(subdf):
        """
        For one day, compute rolling sums of 'cond_series' and 'valid_series',
        then take their ratio. Each day is handled independently.
        """
        day_cond = cond_series.loc[subdf.index]
        day_valid = valid_series.loc[subdf.index]

        # Rolling sums (per day)
        cond_roll = day_cond.rolling(window_size, min_periods=1).sum()
        valid_roll = day_valid.rolling(window_size, min_periods=1).sum()

        # Ratio: cond_count / valid_count, handle division by zero as NaN
        ratio = cond_roll / valid_roll
        ratio[valid_roll == 0] = np.nan

        return ratio

    # 5) Apply rolling logic day by day, then droplevel(0) to remove group keys
    ratio_series = grouped.apply(_rolling_ratio_for_one_day).droplevel(0)
    ratio_series.name = f"percentage_above_0.5_ignore_nans_cutoff_{cutoff_time_str}"

    # 6) Ensure the result is in ascending order of the original index
    ratio_series = ratio_series.sort_index()

    return ratio_series




def kalman_filter(data, Q=0.00037, R=0.01):
    """
    Apply Kalman filter to a time series.
    
    Parameters:
    - data: input time series
    - Q: process variance (how fast the system changes)
    - R: measurement variance (how noisy the measurements are)

Higher Q: More responsive to changes (follows data more closely)
Lower Q: More smooth, less responsive
Higher R: More smoothing (assumes more measurement noise)
Lower R: Less smoothing (assumes measurements are more accurate
    
    Returns:
    - filtered series
    """
    n = len(data)
    # Initialize state estimates, error covariance
    x_hat = np.zeros(n)  # State estimate
    P = np.zeros(n)      # Error covariance
    
    # Initialize first values
    x_hat[0] = data[0]
    P[0] = 1.0
    
    # Forward pass
    for k in range(1, n):
        # Predict
        x_hat_minus = x_hat[k-1]
        P_minus = P[k-1] + Q
        
        # Update
        K = P_minus / (P_minus + R)  # Kalman gain
        x_hat[k] = x_hat_minus + K * (data[k] - x_hat_minus)
        P[k] = (1 - K) * P_minus
        
    return x_hat



def calculate_daily_cumulative_min(df, column_name, skip_minutes=0):
    """
    For each row, compute the minimum value of `column_name` from `skip_minutes` after 
    the first row of the day up to the current row, ignoring NaNs in the column. 
    Rows before this cutoff will be NaN.

    Parameters:
    - df: pandas.DataFrame with a DateTimeIndex and at least one data column.
    - column_name: str, the name of the column to compute the daily cumulative minimum for.
    - skip_minutes: int, how many minutes from the first row of the day to skip. 
      (e.g., 30 means rows in the first 30 minutes of the day are set to NaN.)

    Returns:
    - A pandas.Series with the same index as df containing the daily-reset cumulative 
      minimum values, ignoring any NaNs before and after skip_minutes.
    """
    # Ensure data is sorted chronologically
    df = df.sort_index()

    # Convert target column to float array, replacing NaN with +∞ so they don't affect the min
    values = df[column_name].astype(float).values
    values_filled = np.where(np.isnan(values), np.inf, values)

    # Factorize the dates to identify each day's rows
    day_ids, unique_days = pd.factorize(df.index.normalize())

    # Prepare an array for results
    result = np.full(len(df), np.nan, dtype=float)

    for day_id in range(len(unique_days)):
        # Identify rows belonging to this day
        mask = (day_ids == day_id)
        if not np.any(mask):
            continue

        # Extract array indices, timestamps, and values for this day
        day_indices = np.where(mask)[0]
        day_values = values_filled[mask]
        day_timestamps = df.index[mask]
        n_day = len(day_indices)

        if n_day == 0:
            continue

        # Determine the day_start (first timestamp of the day) and the cutoff
        day_start_ts = day_timestamps[0]
        day_cutoff_ts = day_start_ts + pd.Timedelta(minutes=skip_minutes)

        # Create a mask for rows on/after the cutoff
        skip_mask = (day_timestamps >= day_cutoff_ts)

        # If no rows pass the cutoff, they all remain NaN
        if not np.any(skip_mask):
            continue

        # Extract subset of day_values after the cutoff
        day_values_after_skip = day_values[skip_mask]
        day_indices_after_skip = day_indices[skip_mask]

        # Compute the cumulative min for rows on/after the cutoff
        day_cum_min = np.minimum.accumulate(day_values_after_skip)

        # Convert any +∞ (all-NaN scenario) back to NaN in the cumulative min array
        day_cum_min[day_cum_min == np.inf] = np.nan

        # Assign the computed minimums to the corresponding positions
        result[day_indices_after_skip] = day_cum_min

        # Rows before the cutoff remain NaN in the result

    # Return the final Series
    return pd.Series(result, 
                     index=df.index, 
                     name=f"daily_cum_min_of_{column_name}_skip_{skip_minutes}m")
















