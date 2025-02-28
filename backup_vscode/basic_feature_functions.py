from algo_feature_engineering.features.ma import (
    moving_average_features_normalized,
    moving_average_features_slopes,
    moving_average_features_percent_diff,
    calculate_hma
)
import talib
import numpy as np
from algo_feature_engineering.features.utils import calculate_trend_vectorized 
import numpy as np
import pandas as pd

def features_for_RNN_models(df, periods=[45, 35, 25, 20, 15, 10], slope_periods=[2, 5]):
    """
    Add features including HMA values, slopes, z-scores, RSI, ROC, squeeze momentum,
    and new VWAP-based features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'close_raw', 'high_raw', 'low_raw', and 'volume' columns
    periods : list, optional
        List of periods for HMA calculation (default=[45, 35, 25, 15, 10])
    slope_periods : list, optional
        List of periods for slope calculations (default=[2, 5])
        
    Returns:
    --------
    tuple
        (pandas.DataFrame, list)
        - DataFrame with added features
        - List of all feature names created by the function
    """
    import talib
    from ta.volume import VolumeWeightedAveragePrice
    
    df_copy = df.copy()
    created_features = []
    
    # Original HMA and slope features for close_raw
    for period in periods:
        hma_values = calculate_hma(df_copy['close_raw'], period=period)
        hma_col = f'close_{period}_raw'
        df_copy[hma_col] = hma_values
        created_features.append(hma_col)
        
        for slope_period in slope_periods:
            slope_col = f'close_slope_{period}_{slope_period}_raw'
            df_copy[slope_col] = talib.LINEARREG_SLOPE(hma_values, timeperiod=slope_period)
            created_features.append(slope_col)
            
            # Z-score calculations for slopes
            rolling_mean = df_copy[slope_col].rolling(window=250).mean()
            rolling_std = df_copy[slope_col].rolling(window=250).std()
            zscore_col = f'close_slope_{period}_{slope_period}_zscore'
            df_copy[zscore_col] = (df_copy[slope_col] - rolling_mean) / rolling_std
            created_features.append(zscore_col)
            
            if slope_period == 2:
                pos_col = f'{slope_col}_positive'
                df_copy[pos_col] = (df_copy[slope_col] > 0).astype(int)
                created_features.append(pos_col)
            
            if slope_period == 5:
                second_deriv_col = f'close_2nd_deriv_{period}_{slope_period}_raw'
                df_copy[second_deriv_col] = talib.LINEARREG_SLOPE(df_copy[slope_col], timeperiod=2)
                created_features.append(second_deriv_col)
                
                pos_deriv_col = f'close_2nd_deriv_{period}_{slope_period}_positive'
                df_copy[pos_deriv_col] = (df_copy[second_deriv_col] > 0).astype(int)
                created_features.append(pos_deriv_col)
    
    # RSI features
    for period in [7, 14, 28]:
        rsi_col = f'close_RSI_{period}_raw'
        df_copy[rsi_col] = talib.RSI(df_copy['close_raw'], timeperiod=period)
        created_features.append(rsi_col)
        
        hma_rsi_col = f'close_RSI_{period}_hma_15'
        df_copy[hma_rsi_col] = calculate_hma(df_copy[rsi_col], period=15)
        created_features.append(hma_rsi_col)
    
    # ROC features
    for period in [7, 14, 28]:
        roc_col = f'close_ROC_{period}_raw'
        df_copy[roc_col] = talib.ROC(df_copy['close_raw'], timeperiod=period)
        created_features.append(roc_col)
        
        hma_roc_col = f'close_ROC_{period}_hma_15'
        df_copy[hma_roc_col] = calculate_hma(df_copy[roc_col], period=15)
        created_features.append(hma_roc_col)
    
    # Squeeze Momentum features
    for period in [20, 30, 40]:
        high = df_copy['high_raw'].rolling(window=period).max()
        low = df_copy['low_raw'].rolling(window=period).min()
        sma = df_copy['close_raw'].rolling(window=period).mean()
        base = df_copy['close_raw'] - ((high + low + sma) / 3)
        
        sqz_col = f'sqz_momentum_{period}'
        df_copy[sqz_col] = talib.LINEARREG_SLOPE(base, timeperiod=period)
        created_features.append(sqz_col)
        
        slope_col = f'sqz_momentum_slope_{period}'
        df_copy[slope_col] = talib.LINEARREG_SLOPE(df_copy[sqz_col], timeperiod=period)
        created_features.append(slope_col)
    
    # --- New VWAP-based features ---
    # Calculate VWAP using a rolling window of 26 bars
    vwap_indicator = VolumeWeightedAveragePrice(
        high=df_copy['high_raw'],
        low=df_copy['low_raw'],
        close=df_copy['close_raw'],
        volume=df_copy['volume'],
        window=26,  # rolling lookback period of 26 bars
        fillna=True
    )
    df_copy['vwap'] = vwap_indicator.volume_weighted_average_price()
    created_features.append('vwap')
    
    # Calculate the rolling z-score of VWAP using a 250-bar lookback
    vwap_roll_mean = df_copy['vwap'].rolling(window=250).mean()
    vwap_roll_std = df_copy['vwap'].rolling(window=250).std()
    df_copy['vwap_zscore'] = (df_copy['vwap'] - vwap_roll_mean) / vwap_roll_std
    created_features.append('vwap_zscore')
    
    # For each HMA period, compute the HMA of the vwap_zscore and then its slope.
    vwap_hma_periods = [5, 10, 15, 20, 25]
    for period in vwap_hma_periods:
        hma_col = f'hma_vwap_zscore_{period}'
        df_copy[hma_col] = calculate_hma(df_copy['vwap_zscore'], period=period)
        created_features.append(hma_col)
        
        slope_col = f'{hma_col}_slope'
        df_copy[slope_col] = talib.LINEARREG_SLOPE(df_copy[hma_col], timeperiod=2)
        created_features.append(slope_col)

    ##### RETURNS FEATURES #####
    trailing_return_periods = [5, 10, 20, 30]  # Different lookback periods for returns
    zscore_lookback = 250  # Rolling window for z-score calculation
    ema_period = 15  # EMA smoothing period
    
    # Loop over different trailing return periods
    for period in trailing_return_periods:
        prefix = f"trailing_return_{period}"
        
        # Step 1: Calculate trailing returns over the given period
        df_copy[f'{prefix}'] = df_copy['close_raw'].pct_change(periods=period)
        
        # Step 2: Compute rolling mean and standard deviation for z-score calculation
        df_copy[f'{prefix}_mean'] = df_copy[f'{prefix}'].rolling(window=zscore_lookback).mean()
        df_copy[f'{prefix}_std'] = df_copy[f'{prefix}'].rolling(window=zscore_lookback).std()
        
        # Step 3: Compute rolling z-score for the trailing return
        df_copy[f'{prefix}_zscore'] = (
            (df_copy[f'{prefix}'] - df_copy[f'{prefix}_mean']) /
            df_copy[f'{prefix}_std']
        )
        
        # Step 4: Apply EMA smoothing to the z-score
        df_copy[f'{prefix}_zscore_ema'] = (
            df_copy[f'{prefix}_zscore'].ewm(span=ema_period, adjust=False).mean()
        )

    # T3 Moving Average 
    import pandas as pd
    import numpy as np
    
    def calculate_t3(series, period, factor=0.7):
        """
        Computes the T3 moving average.
        
        This implementation mimics the PineScript version:
          gd(src, period) = EMA(src, period) * (1 + factor) - EMA(EMA(src, period), period) * factor
          T3 = gd(gd(gd(src, period), period), period)
          
        Parameters:
            series (pd.Series): The input data series (e.g. close prices).
            period (int): The length parameter for the EMA calculations.
            factor (float): Smoothing factor (default is 0.7).
        
        Returns:
            pd.Series: The T3 moving average.
        """
        def gd(src, period, factor):
            ema1 = src.ewm(span=period, adjust=False).mean()
            ema2 = ema1.ewm(span=period, adjust=False).mean()
            return ema1 * (1 + factor) - ema2 * factor
    
        t3 = gd(gd(gd(series, period, factor), period, factor), period, factor)
        return t3

    # For a Hull period of 60, we use T3 period = 36
    df_copy['close_T3_36_raw'] = calculate_t3(df_copy['close_raw'], period=36, factor=0.7)
    df_copy['close_T3_36_slope_2_raw'] = talib.LINEARREG_SLOPE(df_copy['close_T3_36_raw'], timeperiod=2)
    df_copy['close_T3_36_slope_5_raw'] = talib.LINEARREG_SLOPE(df_copy['close_T3_36_raw'], timeperiod=5)
    
    # For a Hull period of 55, we use T3 period = 33
    df_copy['close_T3_33_raw'] = calculate_t3(df_copy['close_raw'], period=33, factor=0.7)
    df_copy['close_T3_33_slope_2_raw'] = talib.LINEARREG_SLOPE(df_copy['close_T3_33_raw'], timeperiod=2)
    df_copy['close_T3_33_slope_5_raw'] = talib.LINEARREG_SLOPE(df_copy['close_T3_33_raw'], timeperiod=5)
    
    # For a Hull period of 45, we use T3 period = 27
    df_copy['close_T3_27_raw'] = calculate_t3(df_copy['close_raw'], period=27, factor=0.7)
    df_copy['close_T3_27_slope_2_raw'] = talib.LINEARREG_SLOPE(df_copy['close_T3_27_raw'], timeperiod=2)
    df_copy['close_T3_27_slope_5_raw'] = talib.LINEARREG_SLOPE(df_copy['close_T3_27_raw'], timeperiod=5)
    
    # For a Hull period of 35, we use T3 period = 21
    df_copy['close_T3_21_raw'] = calculate_t3(df_copy['close_raw'], period=21, factor=0.7)
    df_copy['close_T3_21_slope_2_raw'] = talib.LINEARREG_SLOPE(df_copy['close_T3_21_raw'], timeperiod=2)
    df_copy['close_T3_21_slope_5_raw'] = talib.LINEARREG_SLOPE(df_copy['close_T3_21_raw'], timeperiod=5)
    
    # For a Hull period of 25, we use T3 period = 15
    df_copy['close_T3_15_raw'] = calculate_t3(df_copy['close_raw'], period=15, factor=0.7)
    df_copy['close_T3_15_slope_2_raw'] = talib.LINEARREG_SLOPE(df_copy['close_T3_15_raw'], timeperiod=2)
    df_copy['close_T3_15_slope_2_raw_positive'] = (df_copy['close_T3_15_slope_2_raw'] > 0).astype(int)
    df_copy['close_T3_15_slope_5_raw'] = talib.LINEARREG_SLOPE(df_copy['close_T3_15_raw'], timeperiod=5)
    df_copy['close_T3_15_2nd_deriv_5_raw'] = talib.LINEARREG_SLOPE(df_copy['close_T3_15_slope_5_raw'], timeperiod=2)
    df_copy['close_T3_15_2nd_deriv_5_positive'] = (df_copy['close_T3_15_2nd_deriv_5_raw'] > 0).astype(int)
    
    # For a Hull period of 20, we use T3 period = 12
    df_copy['close_T3_12_raw'] = calculate_t3(df_copy['close_raw'], period=12, factor=0.7)
    df_copy['close_T3_12_slope_2_raw'] = talib.LINEARREG_SLOPE(df_copy['close_T3_12_raw'], timeperiod=2)
    df_copy['close_T3_12_slope_2_raw_positive'] = (df_copy['close_T3_12_slope_2_raw'] > 0).astype(int)
    df_copy['close_T3_12_slope_5_raw'] = talib.LINEARREG_SLOPE(df_copy['close_T3_12_raw'], timeperiod=5)
    df_copy['close_T3_12_2nd_deriv_5_raw'] = talib.LINEARREG_SLOPE(df_copy['close_T3_12_slope_5_raw'], timeperiod=2)
    df_copy['close_T3_12_2nd_deriv_5_positive'] = (df_copy['close_T3_12_2nd_deriv_5_raw'] > 0).astype(int)
    
    # For a Hull period of 15, we use T3 period = 9
    df_copy['close_T3_9_raw'] = calculate_t3(df_copy['close_raw'], period=9, factor=0.7)
    df_copy['close_T3_9_slope_2_raw'] = talib.LINEARREG_SLOPE(df_copy['close_T3_9_raw'], timeperiod=2)
    df_copy['close_T3_9_slope_5_raw'] = talib.LINEARREG_SLOPE(df_copy['close_T3_9_raw'], timeperiod=5)
    df_copy['close_T3_9_2nd_deriv_5_raw'] = talib.LINEARREG_SLOPE(df_copy['close_T3_9_slope_5_raw'], timeperiod=2)
    df_copy['close_T3_9_2nd_deriv_5_positive'] = (df_copy['close_T3_9_2nd_deriv_5_raw'] > 0).astype(int)
    
    # For a Hull period of 10, we use T3 period = 6
    df_copy['close_T3_6_raw'] = calculate_t3(df_copy['close_raw'], period=6, factor=0.7)
    df_copy['close_T3_6_slope_2_raw'] = talib.LINEARREG_SLOPE(df_copy['close_T3_6_raw'], timeperiod=2)
    df_copy['close_T3_6_slope_5_raw'] = talib.LINEARREG_SLOPE(df_copy['close_T3_6_raw'], timeperiod=5)
    df_copy['close_T3_6_2nd_deriv_5_raw'] = talib.LINEARREG_SLOPE(df_copy['close_T3_6_slope_5_raw'], timeperiod=2)
    df_copy['close_T3_6_2nd_deriv_5_positive'] = (df_copy['close_T3_6_2nd_deriv_5_raw'] > 0).astype(int)
    
    # For a Hull period of 5, we use T3 period = 3
    df_copy['close_T3_3_raw'] = calculate_t3(df_copy['close_raw'], period=3, factor=0.7)
    df_copy['close_T3_3_slope_2_raw'] = talib.LINEARREG_SLOPE(df_copy['close_T3_3_raw'], timeperiod=2)
    df_copy['close_T3_3_slope_5_raw'] = talib.LINEARREG_SLOPE(df_copy['close_T3_3_raw'], timeperiod=5)

    # T3 moving average + z-score 
    # Define the list of slope features
    slope_features = [
        'close_T3_36_slope_2_raw', 'close_T3_36_slope_5_raw',
        'close_T3_33_slope_2_raw', 'close_T3_33_slope_5_raw',
        'close_T3_27_slope_2_raw', 'close_T3_27_slope_5_raw',
        'close_T3_21_slope_2_raw', 'close_T3_21_slope_5_raw',
        'close_T3_15_slope_2_raw', 'close_T3_15_slope_5_raw', 'close_T3_15_2nd_deriv_5_raw',
        'close_T3_12_slope_2_raw', 'close_T3_12_slope_5_raw', 'close_T3_12_2nd_deriv_5_raw',
        'close_T3_9_slope_2_raw', 'close_T3_9_slope_5_raw', 'close_T3_9_2nd_deriv_5_raw',
        'close_T3_6_slope_2_raw', 'close_T3_6_slope_5_raw', 'close_T3_6_2nd_deriv_5_raw',
        'close_T3_3_slope_2_raw', 'close_T3_3_slope_5_raw'
    ]
    
    # Define the rolling window size (for trailing calculation, e.g., 250 periods)
    window_size = 250
    
    # Loop through each slope feature, calculate its trailing z-score, and add as a new column.
    for col in slope_features:
        rolling_mean = df_copy[col].rolling(window=window_size).mean()
        rolling_std = df_copy[col].rolling(window=window_size).std()
        df_copy[f"{col}_zscore"] = (df_copy[col] - rolling_mean) / rolling_std
    
    
    return df_copy, created_features

def remove_RNN_features(df, feature_names):
    """
    Remove all features created by features_for_RNN_models function from the DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the RNN features to be removed
    feature_names : list
        List of feature names to remove, as returned by features_for_RNN_models
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with RNN features removed
    """
    df_copy = df.copy()
    
    # Remove all features in the feature_names list
    df_copy = df_copy.drop(columns=feature_names, errors='ignore')
    
    return df_copy


def add_trend_buy_threshold_columns(df, step=0.05, start=0.25, end=0.70):
    """
    Add trend buy threshold columns to the dataframe based on specified ranges.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing 'trend_buy' and 'trend_buy_threshold' columns
    step (float): Size of each threshold range (default: 0.05)
    start (float): Starting threshold value (default: 0.25)
    end (float): Ending threshold value (default: 0.70)
    
    Returns:
    pd.DataFrame: DataFrame with added threshold columns
    """
    result_df = df.copy()
    
    # Generate ranges from start to end
    thresholds = np.arange(start, end, step)
    
    # Add columns for each range except the last one
    for lower in thresholds:
        upper = lower + step
        # Round the values to avoid floating point precision issues
        column_name = f'trend_buy_threshold_0_{int(round(lower*100))}_{int(round(upper*100))}'
        result_df[column_name] = ((result_df['trend_buy_2_30_24'] >= lower) & 
                                (result_df['trend_buy_2_30_24'] < upper)).astype(int)
    
    # Add the final "plus" column
    final_column = f'trend_buy_threshold_0_{int(round(end*100))}_plus'
    result_df[final_column] = (result_df['trend_buy_2_30_24'] >= end).astype(int)
    
    return result_df


def add_raw_distance_metrics(df, epsilon=1e-9):
    """
    Add percentage distance metrics between different HMA periods and their slopes.
    Uses columns with '_raw' suffix.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing HMA and slope percentage raw columns
    epsilon : float, optional
        Small constant to prevent division by zero (default=1e-9)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added distance metrics
    """
    df_copy = df.copy()
    
    # Percentage distances between slope positivity percentages
    metrics = [
        {
            'short_period': 50,
            'long_period': 350,
            'type': 'slope_percent_positive',
            'output_name': 'hma_long_term_slope_percent_positive_percentage_distance_50_350_raw'
        },
        {
            'short_period': 20,
            'long_period': 350,
            'type': 'slope_percent_positive',
            'output_name': 'hma_long_term_slope_percent_positive_percentage_distance_20_350_raw'
        },
        {
            'short_period': 10,
            'long_period': 75,
            'type': 'hma',
            'output_name': 'hma_long_term_distance_10_75_raw'
        },
        {
            'short_period': 20,
            'long_period': 200,
            'type': 'hma',
            'output_name': 'hma_long_term_distance_20_200_raw'
        }
    ]
    
    for metric in metrics:
        if metric['type'] == 'slope_percent_positive':
            short_col = f'hma_long_term_slope_percent_positive_{metric["short_period"]}_raw'
            long_col = f'hma_long_term_slope_percent_positive_{metric["long_period"]}_raw'
        else:  # type == 'hma'
            short_col = f'price_slope_hma_long_term_{metric["short_period"]}_raw'
            long_col = f'price_slope_hma_long_term_{metric["long_period"]}_raw'
            
        df_copy[metric['output_name']] = (
            df_copy[short_col] - df_copy[long_col]
        ) / (df_copy[long_col] + epsilon)
    
    return df_copy

def add_raw_slope_trend_percentages(df, periods=[350, 300, 250, 200, 150, 100, 75, 50, 20]):
    """
    Add percentage of time HMA slopes have been positive over different lookback periods.
    Uses columns with '_raw' suffix.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'hma_long_term_slope_*_raw' columns
    periods : list, optional
        List of periods for trend calculation, ordered from largest to smallest
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added trend percentage features
    """
    df_copy = df.copy()
    
    for period in periods:
        input_col = f'hma_long_term_slope_{period}_raw'
        output_col = f'hma_long_term_slope_percent_positive_{period}_raw'
        
        df_copy[output_col] = calculate_trend_vectorized(df_copy, input_col, period)
    
    return df_copy

def add_all_trend_features(df):
    """
    Add all trend features with different configurations.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with all trend features added
    """
    df_copy = df.copy()
    
    # Short-term trend configurations
    short_term_configs = [
        {'slope_lookback': 2, 'trend_line_buy_A': 2},
        {'slope_lookback': 2, 'trend_line_buy_A': 3},
        {'slope_lookback': 2, 'trend_line_buy_A': 4},
        {'slope_lookback': 3, 'trend_line_buy_A': 4},
        {'slope_lookback': 4, 'trend_line_buy_A': 4},
        {'slope_lookback': 5, 'trend_line_buy_A': 4},
    ]
    
    # Apply short-term configurations
    for config in short_term_configs:
        df_copy = add_custom_trend_feature(
            df=df_copy,
            slope_lookback=config['slope_lookback'],
            trend_line_buy_A=config['trend_line_buy_A'],
            hma_period=5,
            slope_buy=3,
            buy_2nd_derivative_slope=3
        )
    
    # Long-term trend configurations
    long_term_trend_line_periods = [5, 15, 30, 60, 100, 200, 300]
    
    # Apply long-term configurations
    for trend_line_period in long_term_trend_line_periods:
        df_copy = add_custom_trend_feature(
            df=df_copy,
            slope_lookback=2,
            trend_line_buy_A=trend_line_period,
            hma_period=24,
            slope_buy=9,
            buy_2nd_derivative_slope=5
        )
    
    return df_copy


def add_all_crossover_features(df, config=None):
   """
   Add periods since crossover features for multiple columns with their respective thresholds.
   
   Parameters:
   -----------
   df : pandas.DataFrame
       Input DataFrame
   config : list of dict or None, optional
       List of configurations for crossover calculations. Each dict should have 'column' and 'threshold'.
       If None, uses default configuration.
       
   Returns:
   --------
   pandas.DataFrame
       DataFrame with added crossover features
   """
   df_copy = df.copy()
   
   if config is None:
       # Default configuration
       hma_periods = [20, 50, 75, 100, 150, 200, 250, 300, 350]
       
       config = [
           {'column': 'trend_slope_2nd_derivative_buy', 'threshold': 0.0},
           {'column': 'trend_buy', 'threshold': 0.5}
       ]
       
       # Add HMA slope configurations
       for period in hma_periods:
           config.append({
               'column': f'hma_long_term_slope_{period}',
               'threshold': 0.0
           })
   
   for cfg in config:
       df_copy = add_periods_since_crossover(
           df_copy,
           column=cfg['column'],
           threshold=cfg['threshold']
       )
   
   return df_copy


def add_periods_since_crossover(df, column, threshold=0):
    """
    Adds a new column to the DataFrame indicating the number of periods since the last
    crossover of the specified column based on the given threshold.

    A crossover is defined as the column value transitioning from below the threshold
    to equal or above the threshold.

    The new column is named as '{column}_periods_since_cross_over'.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - column (str): The column name to check for crossovers.
    - threshold (float): The threshold value to define a crossover. Default is 0.

    Returns:
    - pd.DataFrame: The DataFrame with the new column added.
    """
    # Generate the new column name
    new_column = f'{column}_periods_since_cross_over'
    
    # Identify crossover points: previous < threshold and current >= threshold
    crossover = (df[column].shift(1) < threshold) & (df[column] >= threshold)
    
    # Create a group identifier that increments at each crossover
    group = crossover.cumsum()
    
    # Calculate periods since last crossover within each group
    # Convert to float32 immediately after calculation
    df[new_column] = df.groupby(group).cumcount().astype('float32')
    
    return df

def add_atr_percent_change(df, atr_period_labeling):
    """
    Adds atr_percent_change column to the dataframe based on high, low, and close prices.
    
    Args:
        df: DataFrame containing 'high_raw', 'low_raw', and 'close_raw' columns
        atr_period_labeling: Integer period for ATR calculation
        
    Returns:
        DataFrame with added 'atr_percent_change' column
    """
    high_prices = df['high_raw'].values
    low_prices = df['low_raw'].values
    close_prices = df['close_raw'].values
    
    atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=atr_period_labeling)
    df['atr_percent_change'] = (atr / df['close_raw']) * 100
    
    return df

def add_trailing_pct_change_atr_features(df, lookbacks=[300, 200, 100, 20, 15, 10, 7, 5, 3], columns=['close', 'close_raw']):
    """
    Add trailing percent change ATR adjusted features for multiple lookback periods and columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    lookbacks : list, optional
        List of lookback periods for calculation
    columns : list, optional
        List of columns to calculate features for
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added trailing percent change ATR adjusted features
    """
    df_copy = df.copy()
    
    for column in columns:
        for lookback in lookbacks:
            df_copy = add_trailing_percent_change_atr_adjusted(df_copy, column, lookback)
    
    return df_copy

def add_trailing_percent_change_atr_adjusted(df, column, lookback):
    """
    Adds a trailing percent change adjusted by ATR to the DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - column (str): The name of the column to calculate percent change on (e.g., 'close').
    - lookback (int): The number of periods for the trailing percent change.

    Returns:
    - pd.DataFrame: The DataFrame with the new adjusted trailing percent change column added.
    """
    # Calculate trailing percent change
    pct_change_col = f'trailing_percent_change_{lookback}'
    df[pct_change_col] = df[column].pct_change(periods=lookback) * 100

    # Handle division by zero by replacing zeros with NaN
    atr_col = 'atr_percent_change'
    adjusted_atr_col = f'{atr_col}_adjusted'
    df[adjusted_atr_col] = df[atr_col].replace(0, pd.NA)

    # Calculate the adjusted trailing percent change
    new_column = f'trailing_percent_change_atr_adjusted_{column}_{lookback}'
    df[new_column] = df[pct_change_col] / df[adjusted_atr_col]

    # Optionally, drop the temporary adjusted ATR column
    df.drop(columns=[adjusted_atr_col], inplace=True)

    return df

def add_rolling_zscore_features(df, windows=[500, 200, 100, 20], column='close_raw'):
    """
    Add rolling z-score features for multiple window sizes.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    windows : list, optional
        List of window sizes for z-score calculation
    column : str, optional
        Name of the column to calculate z-scores for
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added z-score features
    """
    df_copy = df.copy()
    
    for window in windows:
        col_name = f'{column}_zscore_{window}'
        df_copy[col_name] = calculate_rolling_zscore(df_copy, column, window)
    
    return df_copy

def add_rolling_percentile_features(df, windows=[200, 100, 50, 20, 10, 5], column='close_raw'):
    """
    Add rolling percentile features for multiple window sizes.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    windows : list, optional
        List of window sizes for percentile calculation
    column : str, optional
        Name of the column to calculate percentiles for
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added percentile features
    """
    df_copy = df.copy()
    
    for window in windows:
        col_name = f'{column}_percentile_{window}'
        df_copy[col_name] = calculate_rolling_percentile(df_copy, column, window)
    
    return df_copy

def calculate_rolling_percentile(df, column_name, window_size):
    """
    Calculate the rolling percentile rank of values in a given column.
    Optimized version using numpy operations instead of apply.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing the time series data
    column_name : str
        Name of the column to calculate percentile ranks for
    window_size : int
        Size of the lookback window
        
    Returns:
    --------
    pandas.Series
        Series containing rolling percentile ranks (0 to 1)
    """
    # Convert to numpy array for faster operations
    values = df[column_name].to_numpy()
    n = len(values)
    
    # Pre-allocate output array
    result = np.empty(n)
    result[0] = 0  # First value has no previous values to compare against
    
    # For each position, calculate how many values in the window are less than current value
    for i in range(1, n):
        # Define window start (handle cases where window would extend before start of data)
        start_idx = max(0, i - window_size + 1)
        window = values[start_idx:i + 1]
        
        # Count values less than current value (including ties)
        # Using <= gives same behavior as scipy.stats.percentileofscore with 'weak' kind
        curr_val = values[i]
        count_less_equal = np.sum(window <= curr_val)
        
        # Calculate percentile
        result[i] = (count_less_equal - 1) / len(window)
    
    return pd.Series(result, index=df.index)


import pandas as pd
import numpy as np

def calculate_rolling_zscore(df, column_name, window_size):
    """
    Calculate the rolling z-score for a given column.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing the time series data
    column_name : str
        Name of the column to calculate z-scores for
    window_size : int
        Size of the lookback window
        
    Returns:
    --------
    pandas.Series
        Series containing rolling z-scores
    """
    
    # Calculate rolling mean
    rolling_mean = df[column_name].rolling(window=window_size, min_periods=1).mean()
    
    # Calculate rolling standard deviation
    rolling_std = df[column_name].rolling(window=window_size, min_periods=1).std()
    
    # Calculate z-score
    zscore = (df[column_name] - rolling_mean) / rolling_std
    
    # Handle cases where std is 0 (z-score would be undefined)
    zscore = zscore.replace([np.inf, -np.inf], np.nan)
    
    return zscore

def add_custom_trend_feature(df, slope_lookback, trend_line_buy_A, hma_period, slope_buy, buy_2nd_derivative_slope):
    """
    Adds custom trend features and additional slope/second derivative features to the dataframe.
    
    The feature names will be automatically generated based on the input parameters:
    - 'trend_buy_{slope_lookback}_{trend_line_buy_A}_{hma_period}'
    - 'trend_slope_combined_buy_{slope_lookback}_{trend_line_buy_A}_{hma_period}_{slope_buy}'
    - 'trend_slope_2nd_derivative_buy_{slope_lookback}_{trend_line_buy_A}_{hma_period}_{slope_buy}_{buy_2nd_derivative_slope}'
    
    Parameters:
    - df: pandas.DataFrame, the input dataframe (e.g., ticker_df_adjusted)
    - slope_lookback: int, the lookback period for the slope calculation (e.g., 2 for 'slope_2')
    - trend_line_buy_A: int, the period for trend_line_buy_A (used in calculating trends)
    - hma_period: int, the period for calculating HMA
    - slope_buy: int, the period for calculating the combined slope
    - buy_2nd_derivative_slope: int, the period for calculating the second derivative slope
    
    Returns:
    - pandas.DataFrame: The updated dataframe with new features added.
    """
    # 1. Generate dynamic feature names
    trend_buy_feature = f'trend_buy_{slope_lookback}_{trend_line_buy_A}_{hma_period}'
    trend_slope_combined_feature = f'trend_slope_combined_buy_{slope_lookback}_{trend_line_buy_A}_{hma_period}_{slope_buy}'
    trend_slope_2nd_derivative_feature = (
        f'trend_slope_2nd_derivative_buy_{slope_lookback}_{trend_line_buy_A}_{hma_period}_'
        f'{slope_buy}_{buy_2nd_derivative_slope}'
    )
    
    #print(f"Generating features: {trend_buy_feature}, {trend_slope_combined_feature}, {trend_slope_2nd_derivative_feature}")
    
    # 2. Calculate the slope for the given lookback period
    slope_col = f'slope_{slope_lookback}'
    df[slope_col] = talib.LINEARREG_SLOPE(df['close'], timeperiod=slope_lookback)
    #print(f"Created column: {slope_col}")
    
    # 3. Define additional trend line periods
    trend_line_buy_B = trend_line_buy_A + 2
    trend_line_buy_C = trend_line_buy_A + 4
    trend_line_buy_D = trend_line_buy_A + 6
    
    #print(f"Trend line periods: A={trend_line_buy_A}, B={trend_line_buy_B}, C={trend_line_buy_C}, D={trend_line_buy_D}")
    
    # 4. Calculate trend vectors using the provided 'calculate_trend_vectorized' function
    df[f'{slope_col}_trend_buy_A'] = calculate_trend_vectorized(df, slope_col, trend_line_buy_A)
    #print(f"Created column: {slope_col}_trend_buy_A")
    df[f'{slope_col}_trend_buy_B'] = calculate_trend_vectorized(df, slope_col, trend_line_buy_B)
    #print(f"Created column: {slope_col}_trend_buy_B")
    df[f'{slope_col}_trend_buy_C'] = calculate_trend_vectorized(df, slope_col, trend_line_buy_C)
    #print(f"Created column: {slope_col}_trend_buy_C")
    df[f'{slope_col}_trend_buy_D'] = calculate_trend_vectorized(df, slope_col, trend_line_buy_D)
    #print(f"Created column: {slope_col}_trend_buy_D")
    
    # 5. Calculate the medium-term trend by averaging the trends
    medium_term_cols = [f'{slope_col}_trend_buy_A', f'{slope_col}_trend_buy_B', 
                        f'{slope_col}_trend_buy_C', f'{slope_col}_trend_buy_D']
    df[f'{slope_col}_trend_medium_term_buy'] = df[medium_term_cols].mean(axis=1)
    #print(f"Created column: {slope_col}_trend_medium_term_buy")
    
    # 6. Calculate the HMA of the medium-term trend
    df[trend_buy_feature] = calculate_hma(
        df[f'{slope_col}_trend_medium_term_buy'].to_numpy(), 
        period=hma_period
    )
    #print(f"Created column: {trend_buy_feature}")
    
    # 7. Calculate combined slopes for buy trends
    buy_slope_2nd = slope_buy + 2
    buy_slope_3rd = slope_buy + 4
    
    #print(f"buy_slope_2nd={buy_slope_2nd}, buy_slope_3rd={buy_slope_3rd}")
    
    # 8. Calculate slopes based on the 'trend_buy' feature
    df[f'trend_buy_slope1'] = talib.LINEARREG_SLOPE(df[trend_buy_feature], timeperiod=slope_buy)
    #print(f"Created column: trend_buy_slope1")
    df[f'trend_buy_slope2'] = talib.LINEARREG_SLOPE(df[trend_buy_feature], timeperiod=buy_slope_2nd)
    #print(f"Created column: trend_buy_slope2")
    df[f'trend_buy_slope3'] = talib.LINEARREG_SLOPE(df[trend_buy_feature], timeperiod=buy_slope_3rd)
    #print(f"Created column: trend_buy_slope3")
    
    # 9. Correctly calculate the combined slope as a Series and assign it to a single column
    df[trend_slope_combined_feature] = df[['trend_buy_slope1', 'trend_buy_slope2', 'trend_buy_slope3']].mean(axis=1)
    #print(f"Created column: {trend_slope_combined_feature}")
    
    # 10. Calculate the second derivative slope based on the combined slope feature
    df[trend_slope_2nd_derivative_feature] = talib.LINEARREG_SLOPE(
        df[trend_slope_combined_feature], 
        timeperiod=buy_2nd_derivative_slope
    )
    #print(f"Created column: {trend_slope_2nd_derivative_feature}")
    
    # 11. Return the updated dataframe with new features
    return df

def calculate_hma_distance_features(df, epsilon=1e-9):
    """
    Calculate basic HMA distance features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame with required HMA columns
    epsilon : float
        Small number to prevent division by zero
    """
    df = df.copy()
    
    # Basic HMA close distance features
    hma_periods = [10, 15, 20, 25, 30, 40, 75, 150]
    for period in hma_periods:
        hma_col = f'price_slope_hma_long_term_{period}'
        df[f'close_distance_hma_{period}'] = (
            df['close'] - df[hma_col]
        ) / (df[hma_col] + epsilon)
        
    return df

def calculate_hma_distance_features_raw(df, epsilon=1e-9):
    """
    Calculate basic HMA distance features using close_raw price data.
    Features are named with pattern 'close_*_raw'.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame with required HMA columns and close_raw
    epsilon : float
        Small number to prevent division by zero
    """
    df = df.copy()
    
    # Basic HMA close distance features
    hma_periods = [10, 15, 20, 25, 30, 40, 75, 150]
    for period in hma_periods:
        hma_col = f'price_slope_hma_long_term_{period}_raw'
        df[f'close_distance_hma_{period}_raw'] = (
            df['close_raw'] - df[hma_col]
        ) / (df[hma_col] + epsilon)
        
    return df

def calculate_hma_cross_period_features_raw(df, epsilon=1e-9):
    """
    Calculate differences between HMA periods, normalized by the longer period.
    Features are named with pattern 'close_*_raw'.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame with required HMA columns (price_slope_hma_long_term_*_raw)
    epsilon : float, optional
        Small number to prevent division by zero, defaults to 1e-9
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added HMA difference features
        
    Notes:
    ------
    Calculates normalized differences between:
    - 5 and 20 period HMA
    - 5 and 40 period HMA
    """
    df = df.copy()
    
    # Validate required columns exist
    required_columns = [
        'price_slope_hma_long_term_5_raw',
        'price_slope_hma_long_term_20_raw',
        'price_slope_hma_long_term_40_raw'
    ]
    
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column {col} not found in dataframe")
    
    # Calculate HMA differences
    df['close_distance_hma_5_20_raw'] = (
        df['price_slope_hma_long_term_5_raw'] - 
        df['price_slope_hma_long_term_20_raw']
    ) / (df['price_slope_hma_long_term_20_raw'] + epsilon)
    
    df['close_distance_hma_5_40_raw'] = (
        df['price_slope_hma_long_term_5_raw'] - 
        df['price_slope_hma_long_term_40_raw']
    ) / (df['price_slope_hma_long_term_40_raw'] + epsilon)
    
    return df

def calculate_hma_cross_period_features(df, epsilon=1e-9):
    """
    Calculate cross-period HMA features.
    """
    df = df.copy()
    
    # HMA cross-period distances
    short_long_pairs = [(5, 20), (5, 40)]
    for short_period, long_period in short_long_pairs:
        short_col = f'price_slope_hma_long_term_{short_period}'
        long_col = f'price_slope_hma_long_term_{long_period}'
        df[f'distance_hma_{short_period}_{long_period}'] = (
            df[short_col] - df[long_col]
        ) / (df[long_col] + epsilon)
        
    return df

def calculate_trend_coherence_features(df):
    """
    Calculate multi-timeframe trend coherence features.
    """
    df = df.copy()
    
    coherence_windows = [20, 40, 75]
    for i in range(len(coherence_windows)-1):
        short_window = coherence_windows[i]
        long_window = coherence_windows[i+1]
        
        short_hma = f'price_slope_hma_long_term_{short_window}'
        long_hma = f'price_slope_hma_long_term_{long_window}'
        
        # Rolling correlation between short and long HMAs
        df[f'trend_coherence_{short_window}_{long_window}'] = (
            df[short_hma].rolling(30)
            .corr(df[long_hma])
            .fillna(0)
        )
        
        # Directional agreement component
        short_direction = df[short_hma].diff().rolling(10).mean()
        long_direction = df[long_hma].diff().rolling(10).mean()
        directional_agreement = np.sign(short_direction) == np.sign(long_direction)
        
        # Combine correlation with directional agreement
        df[f'trend_coherence_{short_window}_{long_window}'] *= directional_agreement.astype(float)
        
    return df

def calculate_volatility_adjusted_momentum_features(df, epsilon=1e-9):
    """
    Calculate volatility-adjusted momentum features with quality components.
    """
    df = df.copy()
    
    if 'atr_percent_change_for_labeling_sav_gol' not in df.columns:
        return df
        
    for period in [10, 20, 40]:
        # Momentum component
        momentum = df['close'].pct_change(period)
        
        # Volatility scaling
        volatility = df['atr_percent_change_for_labeling_sav_gol']
        volatility_ma = volatility.rolling(period).mean()
        
        # Basic volatility-adjusted momentum
        df[f'vol_adj_momentum_{period}'] = (
            momentum / (volatility_ma + epsilon)
        )
        
        # Path quality components
        price_path = df['close'].pct_change().rolling(period).std()
        time_above_entry = (
            (df['close'].diff().rolling(period).apply(lambda x: (x > 0).mean()))
        )
        
        # Combined quality score
        df[f'vol_adj_momentum_quality_{period}'] = (
            df[f'vol_adj_momentum_{period}'] * 
            (1 + time_above_entry) * 
            (1 / (1 + price_path))
        )
        
    return df

def add_slope_features(df, periods=[70, 45, 25, 15, 10, 5, 4], slope_period=2):
    """
    Add slope features to a dataframe for multiple periods.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing a 'close' column
    periods : list, optional
        List of periods for calculating slopes (default=[70, 45, 25, 15, 10, 5, 4])
    slope_period : int, optional
        Period for calculating linear regression slope (default=2)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added slope features
    """
    df_copy = df.copy()
    
    for period in periods:
        # Calculate slope directly on the close price
        slope_col = f'close_slope_{period}_{slope_period}'
        df_copy[slope_col] = talib.LINEARREG_SLOPE(df_copy['close'], timeperiod=period)
    
    return df_copy

def add_raw_slope_features(df, slope_periods=[30, 20, 10, 2], hma_period=50):
    """
    Add raw slope features with HMA smoothing.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing a 'close_raw' column
    slope_periods : list, optional
        List of periods for slope calculation (default=[30, 20, 10, 2])
    hma_period : int, optional
        Period for HMA calculation on slopes (default=50)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added slope and HMA features
    """
    df_copy = df.copy()
    
    for period in slope_periods:
        # Calculate raw slope
        slope_col = f'close_raw_slope_{period}'
        df_copy[slope_col] = talib.LINEARREG_SLOPE(df_copy['close_raw'], timeperiod=period)
        
        # Calculate HMA of the slope
        hma_col = f'close_raw_slope_{period}_hma_{hma_period}'
        df_copy[hma_col] = calculate_hma(df_copy[slope_col], period=hma_period)
    
    return df_copy




def add_long_term_hma_features_raw(df, periods=[350, 300, 250, 200, 150, 100, 75, 50, 40, 30, 25, 20, 15, 10, 5], 
                                  calculate_slopes=True, slope_period=2):
    """
    Add long-term HMA features for raw close prices, with optional slopes and positive indicators.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing a 'close_raw' column
    periods : list, optional
        List of periods for HMA calculation, ordered from largest to smallest
    calculate_slopes : bool, optional
        Whether to calculate slopes and positive indicators (default=True)
    slope_period : int, optional
        Period for calculating linear regression slope (default=2)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added features for raw close prices
    """
    df_copy = df.copy()
    
    for period in periods:
        # Calculate HMA for raw close
        hma_col = f'price_slope_hma_long_term_{period}_raw'
        df_copy[hma_col] = calculate_hma(df_copy['close_raw'].to_numpy(), period=period)
        
        if calculate_slopes:
            # Calculate slope of HMA
            slope_col = f'hma_long_term_slope_{period}_raw'
            df_copy[slope_col] = talib.LINEARREG_SLOPE(
                df_copy[hma_col].to_numpy(), 
                timeperiod=slope_period
            )
            
            # Calculate positive indicator with updated naming convention
            positive_col = f'hma_long_term_slope_{period}_positive_raw'
            df_copy[positive_col] = (df_copy[slope_col] > 0).astype(int)
    
    return df_copy





def add_long_term_hma_features(df, periods=[350, 300, 250, 200, 150, 100, 75, 50, 40, 30, 25, 20, 15, 10, 5], 
                              calculate_slopes=True, slope_period=2):
    """
    Add long-term HMA features, with optional slopes and positive indicators.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing a 'close' column
    periods : list, optional
        List of periods for HMA calculation, ordered from largest to smallest
    calculate_slopes : bool, optional
        Whether to calculate slopes and positive indicators (default=True)
    slope_period : int, optional
        Period for calculating linear regression slope (default=2)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added features
    """
    df_copy = df.copy()
    
    for period in periods:
        # Calculate HMA
        hma_col = f'price_slope_hma_long_term_{period}'
        df_copy[hma_col] = calculate_hma(df_copy['close'].to_numpy(), period=period)
        
        if calculate_slopes:
            # Calculate slope of HMA
            slope_col = f'hma_long_term_slope_{period}'
            df_copy[slope_col] = talib.LINEARREG_SLOPE(
                df_copy[hma_col].to_numpy(), 
                timeperiod=slope_period
            )
            
            # Calculate positive indicator
            positive_col = f'hma_long_term_slope_{period}_positive'
            df_copy[positive_col] = (df_copy[slope_col] > 0).astype(int)
    
    return df_copy



def add_long_term_slope_percentages(df, periods=[350, 300, 250, 200, 150, 100, 75, 50, 20]):
   """
   Add percentage of time HMA slopes have been positive over different lookback periods.
   
   Parameters:
   -----------
   df : pandas.DataFrame
       DataFrame containing 'hma_long_term_slope_*' columns
   periods : list, optional
       List of periods for trend calculation, ordered from largest to smallest
       
   Returns:
   --------
   pandas.DataFrame
       DataFrame with added trend percentage features
   """
   df_copy = df.copy()
   
   for period in periods:
       input_col = f'hma_long_term_slope_{period}'
       output_col = f'hma_long_term_slope_percent_positive_{period}'
       
       df_copy[output_col] = calculate_trend_vectorized(df_copy, input_col, period)
   
   return df_copy

def add_distance_metrics(df, epsilon=1e-9):
   """
   Add percentage distance metrics between different HMA periods and their slopes.
   
   Parameters:
   -----------
   df : pandas.DataFrame
       DataFrame containing HMA and slope percentage columns
   epsilon : float, optional
       Small constant to prevent division by zero (default=1e-9)
       
   Returns:
   --------
   pandas.DataFrame
       DataFrame with added distance metrics
   """
   df_copy = df.copy()
   
   # Percentage distances between slope positivity percentages
   metrics = [
       {
           'short_period': 50,
           'long_period': 350,
           'type': 'slope_percent_positive',
           'output_name': 'hma_long_term_slope_percent_positive_percentage_distance_50_350'
       },
       {
           'short_period': 20,
           'long_period': 350,
           'type': 'slope_percent_positive',
           'output_name': 'hma_long_term_slope_percent_positive_percentage_distance_20_350'
       },
       {
           'short_period': 10,
           'long_period': 75,
           'type': 'hma',
           'output_name': 'hma_long_term_distance_10_75'
       },
       {
           'short_period': 20,
           'long_period': 200,
           'type': 'hma',
           'output_name': 'hma_long_term_distance_20_200'
       }
   ]
   
   for metric in metrics:
       if metric['type'] == 'slope_percent_positive':
           short_col = f'hma_long_term_slope_percent_positive_{metric["short_period"]}'
           long_col = f'hma_long_term_slope_percent_positive_{metric["long_period"]}'
       else:  # type == 'hma'
           short_col = f'price_slope_hma_long_term_{metric["short_period"]}'
           long_col = f'price_slope_hma_long_term_{metric["long_period"]}'
           
       df_copy[metric['output_name']] = (
           df_copy[short_col] - df_copy[long_col]
       ) / (df_copy[long_col] + epsilon)
   
   return df_copy


def add_price_position_indicators(df, periods=[150, 75, 40, 30, 25, 20, 15, 10]):
   """
   Add binary indicators for when close price is below various HMA periods.
   
   This creates features showing when price is below different HMA periods, which can be 
   useful signals for XGBoost models. Using binary indicators (0/1) is often more effective
   than raw differences for tree-based models like XGBoost.
   
   Parameters:
   -----------
   df : pandas.DataFrame
       DataFrame containing 'close' and 'price_slope_hma_long_term_*' columns
   periods : list, optional
       List of HMA periods to check against price, ordered from largest to smallest
       
   Returns:
   --------
   pandas.DataFrame
       DataFrame with added binary position indicators
       
   Raises:
   -------
   ValueError
       If required columns are missing
   """
   df_copy = df.copy()
   
   # Validate required columns exist
   if 'close' not in df_copy.columns:
       raise ValueError("DataFrame must contain 'close' column")
       
   for period in periods:
       hma_col = f'price_slope_hma_long_term_{period}'
       if hma_col not in df_copy.columns:
           raise ValueError(f"Required column {hma_col} not found in dataframe")
       
       output_col = f'close_below_price_slope_hma_long_term_{period}'
       df_copy[output_col] = (df_copy['close'] < df_copy[hma_col]).astype(int)
   
   # Additional feature suggestions for XGBoost:
   # 1. Count of how many HMAs the price is below
   df_copy['close_below_hma_count'] = sum(
       (df_copy[f'close_below_price_slope_hma_long_term_{period}'] for period in periods)
   )
   
   # 2. Sequential HMA crossovers (is price below longer period but above shorter period?)
   for i in range(len(periods)-1):
       longer_period = periods[i]
       shorter_period = periods[i+1]
       col_name = f'close_between_hma_{shorter_period}_{longer_period}'
       df_copy[col_name] = (
           (df_copy[f'close_below_price_slope_hma_long_term_{longer_period}'] == 1) & 
           (df_copy[f'close_below_price_slope_hma_long_term_{shorter_period}'] == 0)
       ).astype(int)
   
   return df_copy


def add_price_position_indicators_raw(df, periods=[150, 75, 40, 30, 25, 20, 15, 10]):
    """
    Add binary indicators for when close_raw price is below various HMA periods.
    Feature names include '_raw' suffix to indicate they're based on raw price data.
    
    This creates features showing when price is below different HMA periods, which can be 
    useful signals for XGBoost models. Using binary indicators (0/1) is often more effective
    than raw differences for tree-based models like XGBoost.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'close_raw' and 'price_slope_hma_long_term_*' columns
    periods : list, optional
        List of HMA periods to check against price, ordered from largest to smallest
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added binary position indicators
        
    Raises:
    -------
    ValueError
        If required columns are missing
    """
    df_copy = df.copy()
    
    # Validate required columns exist
    if 'close_raw' not in df_copy.columns:
        raise ValueError("DataFrame must contain 'close_raw' column")
        
    for period in periods:
        hma_col = f'price_slope_hma_long_term_{period}_raw'
        if hma_col not in df_copy.columns:
            raise ValueError(f"Required column {hma_col} not found in dataframe")
        
        output_col = f'close_below_price_slope_hma_long_term_{period}_raw'
        df_copy[output_col] = (df_copy['close_raw'] < df_copy[hma_col]).astype(int)
    
    # Additional feature suggestions for XGBoost:
    # 1. Count of how many HMAs the price is below
    df_copy['close_below_hma_count_raw'] = sum(
        (df_copy[f'close_below_price_slope_hma_long_term_{period}_raw'] for period in periods)
    )
    
    # 2. Sequential HMA crossovers (is price below longer period but above shorter period?)
    for i in range(len(periods)-1):
        longer_period = periods[i]
        shorter_period = periods[i+1]
        col_name = f'close_between_hma_{shorter_period}_{longer_period}_raw'
        df_copy[col_name] = (
            (df_copy[f'close_below_price_slope_hma_long_term_{longer_period}_raw'] == 1) & 
            (df_copy[f'close_below_price_slope_hma_long_term_{shorter_period}_raw'] == 0)
        ).astype(int)
    
    return df_copy


def add_std_deviation_features(df, periods=[40, 30, 20], min_periods=None):
   """
   Add standard deviation-based features for price and HMA relationships.
   Enhanced for XGBoost modeling with additional derived features.
   
   Parameters:
   -----------
   df : pandas.DataFrame
       DataFrame containing 'close' and 'price_slope_hma_long_term_*' columns
   periods : list, optional
       List of periods for rolling standard deviation calculation
   min_periods : int, optional
       Minimum number of observations required for std calculation
       Defaults to period size if None
       
   Returns:
   --------
   pandas.DataFrame
       DataFrame with added standard deviation features
       
   Raises:
   -------
   ValueError
       If required columns are missing
   """
   df_copy = df.copy()
   
   # Input validation
   if 'close' not in df_copy.columns:
       raise ValueError("DataFrame must contain 'close' column")
   
   for period in periods:
       hma_col = f'price_slope_hma_long_term_{period}'
       if hma_col not in df_copy.columns:
           raise ValueError(f"Required column {hma_col} not found in dataframe")
           
       # Base std deviation
       std_col = f'close_std_{period}'
       df_copy[std_col] = df_copy['close'].rolling(
           window=period, 
           min_periods=min_periods or period
       ).std()
       
       # Standard deviations above/below HMA
       below_col = f'close_std_dev_below_hma_{period}'
       above_col = f'close_std_dev_above_hma_{period}'
       
       df_copy[below_col] = np.where(
           df_copy['close'] > df_copy[hma_col],
           0,
           (df_copy[hma_col] - df_copy['close']) / df_copy[std_col]
       )
       
       df_copy[above_col] = np.where(
           df_copy['close'] > df_copy[hma_col],
           (df_copy['close'] - df_copy[hma_col]) / df_copy[std_col],
           0
       )
       
       # Enhanced features for XGBoost
       
       # 1. Combined absolute distance in standard deviations
       df_copy[f'close_std_dev_total_distance_{period}'] = (
           df_copy[below_col] + df_copy[above_col]
       )
       
       # 2. Exponentially weighted version of std dev
       df_copy[f'close_std_ewm_{period}'] = df_copy['close'].ewm(
           span=period, 
           min_periods=min_periods or period
       ).std()
       
       # 3. Relative volatility (current std dev vs longer-term std dev)
       df_copy[f'close_std_relative_{period}'] = (
           df_copy[std_col] / 
           df_copy[std_col].rolling(window=period*2, min_periods=period).mean()
       )
       
       # 4. Volatility-adjusted price distance from HMA
       df_copy[f'close_vol_adj_distance_hma_{period}'] = (
           (df_copy['close'] - df_copy[hma_col]) / 
           (df_copy[std_col] * np.sqrt(period))
       )
       
       # 5. Binary indicators for extreme deviations
       for threshold in [1, 2, 3]:
           df_copy[f'close_std_dev_extreme_below_{period}_{threshold}'] = (
               df_copy[below_col] > threshold
           ).astype(int)
           
           df_copy[f'close_std_dev_extreme_above_{period}_{threshold}'] = (
               df_copy[above_col] > threshold
           ).astype(int)
       
       # 6. Rolling max of std deviations
       df_copy[f'close_std_dev_max_{period}'] = df_copy[f'close_std_dev_total_distance_{period}'].rolling(
           window=period,
           min_periods=min_periods or period
       ).max()
       
   # Cross-period features
   if len(periods) >= 2:
       # Ratio of short-term to long-term std dev
       short_period = min(periods)
       long_period = max(periods)
       df_copy['close_std_ratio_short_long'] = (
           df_copy[f'close_std_{short_period}'] / 
           df_copy[f'close_std_{long_period}']
       )
       
       # Difference in std dev distances
       df_copy['close_std_dev_distance_diff'] = (
           df_copy[f'close_std_dev_total_distance_{short_period}'] -
           df_copy[f'close_std_dev_total_distance_{long_period}']
       )
   
   return df_copy

def add_std_deviation_features_raw(df, periods=[40, 30, 20], min_periods=None):
    """
    Add standard deviation-based features for raw price and HMA relationships.
    Enhanced for XGBoost modeling with additional derived features.
    Uses close_raw price data with features named 'close_*_raw'.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'close_raw' and 'price_slope_hma_long_term_*' columns
    periods : list, optional
        List of periods for rolling standard deviation calculation
    min_periods : int, optional
        Minimum number of observations required for std calculation
        Defaults to period size if None
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added standard deviation features
        
    Raises:
    -------
    ValueError
        If required columns are missing
    """
    df_copy = df.copy()
    
    # Input validation
    if 'close_raw' not in df_copy.columns:
        raise ValueError("DataFrame must contain 'close_raw' column")
    
    for period in periods:
        hma_col = f'price_slope_hma_long_term_{period}_raw'
        if hma_col not in df_copy.columns:
            raise ValueError(f"Required column {hma_col} not found in dataframe")
            
        # Base std deviation
        std_col = f'close_std_{period}_raw'
        df_copy[std_col] = df_copy['close_raw'].rolling(
            window=period, 
            min_periods=min_periods or period
        ).std()
        
        # Standard deviations above/below HMA
        below_col = f'close_std_dev_below_hma_{period}_raw'
        above_col = f'close_std_dev_above_hma_{period}_raw'
        
        df_copy[below_col] = np.where(
            df_copy['close_raw'] > df_copy[hma_col],
            0,
            (df_copy[hma_col] - df_copy['close_raw']) / df_copy[std_col]
        )
        
        df_copy[above_col] = np.where(
            df_copy['close_raw'] > df_copy[hma_col],
            (df_copy['close_raw'] - df_copy[hma_col]) / df_copy[std_col],
            0
        )
        
        # Enhanced features for XGBoost
        
        # 1. Combined absolute distance in standard deviations
        df_copy[f'close_std_dev_total_distance_{period}_raw'] = (
            df_copy[below_col] + df_copy[above_col]
        )
        
        # 2. Exponentially weighted version of std dev
        df_copy[f'close_std_ewm_{period}_raw'] = df_copy['close_raw'].ewm(
            span=period, 
            min_periods=min_periods or period
        ).std()
        
        # 3. Relative volatility (current std dev vs longer-term std dev)
        df_copy[f'close_std_relative_{period}_raw'] = (
            df_copy[std_col] / 
            df_copy[std_col].rolling(window=period*2, min_periods=period).mean()
        )
        
        # 4. Volatility-adjusted price distance from HMA
        df_copy[f'close_vol_adj_distance_hma_{period}_raw'] = (
            (df_copy['close_raw'] - df_copy[hma_col]) / 
            (df_copy[std_col] * np.sqrt(period))
        )
        
        # 5. Binary indicators for extreme deviations
        for threshold in [1, 2, 3]:
            df_copy[f'close_std_dev_extreme_below_{period}_{threshold}_raw'] = (
                df_copy[below_col] > threshold
            ).astype(int)
            
            df_copy[f'close_std_dev_extreme_above_{period}_{threshold}_raw'] = (
                df_copy[above_col] > threshold
            ).astype(int)
        
        # 6. Rolling max of std deviations
        df_copy[f'close_std_dev_max_{period}_raw'] = df_copy[f'close_std_dev_total_distance_{period}_raw'].rolling(
            window=period,
            min_periods=min_periods or period
        ).max()
        
    # Cross-period features
    if len(periods) >= 2:
        # Ratio of short-term to long-term std dev
        short_period = min(periods)
        long_period = max(periods)
        df_copy['close_std_ratio_short_long_raw'] = (
            df_copy[f'close_std_{short_period}_raw'] / 
            df_copy[f'close_std_{long_period}_raw']
        )
        
        # Difference in std dev distances
        df_copy['close_std_dev_distance_diff_raw'] = (
            df_copy[f'close_std_dev_total_distance_{short_period}_raw'] -
            df_copy[f'close_std_dev_total_distance_{long_period}_raw']
        )
    
    return df_copy


def calculate_cci_features_raw(df, calculate_hma_func):
    """
    Calculate CCI-based features including HMAs and slopes.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'close_raw' column
    calculate_hma_func : function
        Function to calculate HMA that accepts (array, period) as parameters
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added CCI features
    """
    df = df.copy()
    
    # Calculate base CCI values for different timeperiods
    df['close_CCI_short_raw'] = talib.CCI(df['close_raw'], df['close_raw'], df['close_raw'], timeperiod=7)
    df['close_CCI_medium_raw'] = talib.CCI(df['close_raw'], df['close_raw'], df['close_raw'], timeperiod=14)
    df['close_CCI_long_raw'] = talib.CCI(df['close_raw'], df['close_raw'], df['close_raw'], timeperiod=28)
    
    # Calculate HMAs for short CCI
    hma_periods = [7, 10, 14, 18, 22]
    for period in hma_periods:
        df[f'close_CCI_short_raw_hma_{period}'] = calculate_hma_func(
            df['close_CCI_short_raw'].to_numpy(), period=period
        )
    
    # Calculate HMAs for medium CCI
    for period in hma_periods:
        df[f'close_CCI_medium_raw_hma_{period}'] = calculate_hma_func(
            df['close_CCI_medium_raw'].to_numpy(), period=period
        )
    
    # Calculate HMAs for long CCI
    for period in hma_periods:
        df[f'close_CCI_long_raw_hma_{period}'] = calculate_hma_func(
            df['close_CCI_long_raw'].to_numpy(), period=period
        )
    
    # Calculate slopes for short CCI HMAs
    for period in [14, 18, 22]:
        # 2-period slopes
        df[f'close_CCI_short_raw_hma_{period}_slope'] = talib.LINEARREG_SLOPE(
            df[f'close_CCI_short_raw_hma_{period}'], timeperiod=2
        )
        # 4-period slopes
        df[f'close_CCI_short_raw_hma_{period}_slope_4'] = talib.LINEARREG_SLOPE(
            df[f'close_CCI_short_raw_hma_{period}'], timeperiod=4
        )
    
    # Calculate slopes for medium CCI HMAs
    for period in [14, 18, 22]:
        df[f'close_CCI_medium_raw_hma_{period}_slope'] = talib.LINEARREG_SLOPE(
            df[f'close_CCI_medium_raw_hma_{period}'], timeperiod=2
        )
    
    # Calculate CCI 20 HMA
    df['close_CCI_20_raw_hma_20'] = calculate_hma_func(
        df['close_CCI_long_raw'].to_numpy(), period=20
    )
    
    # Create binary range features for CCI 20 HMA
    ranges = [
        (-200, -150),
        (-150, -100),
        (-100, -50),
        (-50, 0),
        (0, 50),
        (50, 100),
        (100, 150),
        (150, 200)
    ]
    
    def format_bound(value):
        return f"neg_{abs(value)}" if value < 0 else str(value)
    
    for lower, upper in ranges:
        lower_str = format_bound(lower)
        upper_str = format_bound(upper)
        column_name = f'close_CCI_20_raw_hma_20_between_{lower_str}_and_{upper_str}'
        df[column_name] = (
            (df['close_CCI_20_raw_hma_20'] > lower) &
            (df['close_CCI_20_raw_hma_20'] <= upper)
        ).astype(int)
    
    return df

def calculate_cci_features(df, calculate_hma_func):
    """
    Calculate CCI-based features including HMAs, slopes, and second derivatives.
    Uses standard high, low, close prices (non-raw version).
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'high', 'low', 'close' columns
    calculate_hma_func : function
        Function to calculate HMA that accepts (array, period) as parameters
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added CCI features
    """
    df = df.copy()
    
    # Base CCI calculations for different timeperiods
    periods = {
        'short': 7,
        'medium': 14,
        'long': 28,
        '40': 40,
        '60': 60
    }
    
    # Calculate base CCI values
    for name, period in periods.items():
        df[f'close_CCI_{name}'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=period)
    
    # Calculate HMAs for different CCI periods
    hma_periods = [14, 18, 22]  # All HMA periods we need
    cci_variants = ['short', 'medium', 'long']  # Base variants for all HMA periods
    extended_variants = ['40', '60']  # Variants that only use HMA-14
    
    # Calculate all HMAs
    for cci_var in cci_variants:
        for hma_period in hma_periods:
            df[f'close_CCI_{cci_var}_hma_{hma_period}'] = calculate_hma_func(
                df[f'close_CCI_{cci_var}'].to_numpy(), 
                period=hma_period
            )
    
    # Calculate HMAs for extended periods (40, 60) - only for HMA-14
    for cci_var in extended_variants:
        df[f'close_CCI_{cci_var}_hma_14'] = calculate_hma_func(
            df[f'close_CCI_{cci_var}'].to_numpy(), 
            period=14
        )
    
    # Calculate slopes
    # For short, medium, long CCI with all HMA periods
    for cci_var in cci_variants:
        for hma_period in hma_periods:
            col_name = f'close_CCI_{cci_var}_hma_{hma_period}'
            df[f'{col_name}_slope'] = talib.LINEARREG_SLOPE(df[col_name], timeperiod=2)
    
    # For 40 and 60 period CCI with HMA-14
    for cci_var in extended_variants:
        col_name = f'close_CCI_{cci_var}_hma_14'
        df[f'{col_name}_slope'] = talib.LINEARREG_SLOPE(df[col_name], timeperiod=2)
    
    # Calculate second derivatives (2nd_deriv)
    # For short and medium CCI with HMA-14
    for cci_var in ['short', 'medium']:
        base_col = f'close_CCI_{cci_var}_hma_14'
        df[f'{base_col}_2nd_deriv'] = talib.LINEARREG_SLOPE(
            df[f'{base_col}_slope'], timeperiod=2
        )
    
    # For long CCI with all HMA periods
    for hma_period in hma_periods:
        base_col = f'close_CCI_long_hma_{hma_period}'
        df[f'{base_col}_2nd_deriv'] = talib.LINEARREG_SLOPE(
            df[f'{base_col}_slope'], timeperiod=2
        )
    
    # For 40 and 60 period CCI with HMA-14
    for cci_var in extended_variants:
        base_col = f'close_CCI_{cci_var}_hma_14'
        df[f'{base_col}_2nd_deriv'] = talib.LINEARREG_SLOPE(
            df[f'{base_col}_slope'], timeperiod=2
        )
    
    return df


def calculate_technical_indicators_raw(df):
    """
    Calculate various technical indicators using raw price data.
    Includes ROC, RSI, PPO, and Stochastic oscillators.
    Also calculates 12-period EMAs for ROC and RSI features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'close_raw' column
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added technical indicator features
    """
    df = df.copy()
    
    # Rate of Change (ROC)
    roc_periods = {
        'short': 5,
        'medium': 14,
        'long': 30
    }
    for name, period in roc_periods.items():
        # Base ROC
        base_col = f'close_ROC_{name}_raw'
        df[base_col] = talib.ROC(df['close_raw'], timeperiod=period)
        # EMA of ROC
        df[f'{base_col}_EMA_12'] = talib.EMA(df[base_col], timeperiod=12)
    
    # Relative Strength Index (RSI)
    rsi_periods = {
        'short': 7,
        'medium': 14,
        'long': 28
    }
    for name, period in rsi_periods.items():
        # Base RSI
        base_col = f'close_RSI_{name}_raw'
        df[base_col] = talib.RSI(df['close_raw'], timeperiod=period)
        # EMA of RSI
        df[f'{base_col}_EMA_12'] = talib.EMA(df[base_col], timeperiod=12)
    
    # Percentage Price Oscillator (PPO)
    ppo_periods = [
        ('short', 6, 13),
        ('medium', 12, 26),
        ('long', 24, 52)
    ]
    for name, fast, slow in ppo_periods:
        df[f'close_PPO_{name}_raw'] = talib.PPO(
            df['close_raw'], 
            fastperiod=fast, 
            slowperiod=slow, 
            matype=0
        )
    
    # Stochastic Oscillator
    stoch_periods = {
        'short': 7,
        'medium': 14,
        'long': 28
    }
    for name, period in stoch_periods.items():
        k_col = f'close_Stoch_%K_{name}_raw'
        d_col = f'close_Stoch_%D_{name}_raw'
        df[k_col], df[d_col] = talib.STOCH(
            df['close_raw'],
            df['close_raw'],
            df['close_raw'],
            fastk_period=period,
            slowk_period=3,
            slowk_matype=0,
            slowd_period=3,
            slowd_matype=0
        )
    
    return df

def calculate_technical_indicators(df):
    """
    Calculate various technical indicators using standard price data.
    Includes ROC, RSI, PPO, and Stochastic oscillators.
    Also calculates 12-period EMAs for ROC and RSI features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'close', 'high', 'low' columns
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added technical indicator features
    """
    df = df.copy()
    
    # Input validation
    required_columns = ['close', 'high', 'low']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in dataframe")
    
    # Rate of Change (ROC)
    roc_periods = {
        'short': 5,
        'medium': 14,
        'long': 30
    }
    for name, period in roc_periods.items():
        # Base ROC
        base_col = f'close_ROC_{name}'
        df[base_col] = talib.ROC(df['close'], timeperiod=period)
        # EMA of ROC
        df[f'{base_col}_EMA_12'] = talib.EMA(df[base_col], timeperiod=12)
    
    # Relative Strength Index (RSI)
    rsi_periods = {
        'short': 7,
        'medium': 14,
        'long': 28
    }
    for name, period in rsi_periods.items():
        # Base RSI
        base_col = f'close_RSI_{name}'
        df[base_col] = talib.RSI(df['close'], timeperiod=period)
        # EMA of RSI
        df[f'{base_col}_EMA_12'] = talib.EMA(df[base_col], timeperiod=12)
    
    # Percentage Price Oscillator (PPO)
    ppo_periods = [
        ('short', 6, 13),
        ('medium', 12, 26),
        ('long', 24, 52)
    ]
    for name, fast, slow in ppo_periods:
        df[f'close_PPO_{name}'] = talib.PPO(
            df['close'], 
            fastperiod=fast, 
            slowperiod=slow, 
            matype=0
        )
    
    # Stochastic Oscillator
    stoch_periods = {
        'short': 7,
        'medium': 14,
        'long': 28
    }
    for name, period in stoch_periods.items():
        k_col = f'close_Stoch_%K_{name}'
        d_col = f'close_Stoch_%D_{name}'
        df[k_col], df[d_col] = talib.STOCH(
            df['high'],
            df['low'],
            df['close'],
            fastk_period=period,
            slowk_period=3,
            slowk_matype=0,
            slowd_period=3,
            slowd_matype=0
        )
    
    return df

import talib
import pandas as pd

def add_technical_indicators(df: pd.DataFrame, column_name: str, prefix: str = None) -> pd.DataFrame:
    """
    Add various technical indicators to a DataFrame for a given column.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing the data
    column_name : str
        Name of the column to calculate indicators for
    prefix : str, optional
        Prefix to add to the indicator column names. If None, uses column_name
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added technical indicators
    """
    # Create a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Use column_name as prefix if none provided
    prefix = prefix or column_name
    
    # ROC (Rate of Change) indicators
    roc_periods = {
        'short': 5,
        'medium': 14,
        'long': 30
    }
    
    for term, period in roc_periods.items():
        result_df[f'{prefix}_ROC_{term}'] = talib.ROC(
            result_df[column_name], 
            timeperiod=period
        )
    
    # RSI (Relative Strength Index) indicators
    rsi_periods = {
        'short': 7,
        'medium': 14,
        'long': 28
    }
    
    for term, period in rsi_periods.items():
        result_df[f'{prefix}_RSI_{term}'] = talib.RSI(
            result_df[column_name], 
            timeperiod=period
        )
    
    # PPO (Percentage Price Oscillator) indicators
    ppo_periods = {
        'short': (6, 13),
        'medium': (12, 26),
        'long': (24, 52)
    }
    
    for term, (fast, slow) in ppo_periods.items():
        result_df[f'{prefix}_PPO_{term}'] = talib.PPO(
            result_df[column_name], 
            fastperiod=fast, 
            slowperiod=slow, 
            matype=0
        )
    
    # Stochastic indicators
    stoch_periods = {
        'short': 7,
        'medium': 14,
        'long': 28
    }
    
    for term, period in stoch_periods.items():
        k, d = talib.STOCH(
            result_df[column_name],
            result_df[column_name],
            result_df[column_name],
            fastk_period=period,
            slowk_period=3,
            slowk_matype=0,
            slowd_period=3,
            slowd_matype=0
        )
        result_df[f'{prefix}_Stoch_%K_{term}'] = k
        result_df[f'{prefix}_Stoch_%D_{term}'] = d
    
    # CCI (Commodity Channel Index) indicators
    cci_periods = {
        'short': 7,
        'medium': 14,
        'long': 28
    }
    
    for term, period in cci_periods.items():
        result_df[f'{prefix}_CCI_{term}'] = talib.CCI(
            result_df[column_name],
            result_df[column_name],
            result_df[column_name],
            timeperiod=period
        )
    
    return result_df

import numpy as np
import pandas as pd

def misc_trend_buy_features(df: pd.DataFrame, 
                          trend_buy_col: str = 'trend_buy',
                          slope_col: str = 'slope_2_trend_medium_term_buy',
                          epsilon: float = 1e-9) -> pd.DataFrame:
    """
    Add HMA (Hull Moving Average) features and percentage distance calculations to a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing the data
    trend_buy_col : str, optional
        Name of the trend buy column (default: 'trend_buy')
    slope_col : str, optional
        Name of the slope column (default: 'slope_2_trend_medium_term_buy')
    epsilon : float, optional
        Small value to avoid division by zero (default: 1e-9)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added HMA features and percentage distances
    """
    # Create a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Calculate HMAs for trend_buy column
    trend_buy_periods = [100, 300, 500]
    for period in trend_buy_periods:
        column_name = f'{trend_buy_col}_hma_{period}'
        result_df[column_name] = calculate_hma(
            result_df[trend_buy_col].to_numpy(), 
            period=period
        )
    
    # Calculate HMAs for slope column
    slope_periods = [10, 30, 60, 150, 300]
    for period in slope_periods:
        column_name = f'{trend_buy_col}_{period}'  # Using trend_buy_col as prefix
        result_df[column_name] = calculate_hma(
            result_df[slope_col].to_numpy(), 
            period=period
        )
    
    # Calculate percentage distances
    distance_periods = [10, 150]
    for period in distance_periods:
        column_name = f'{trend_buy_col}_percentage_distance_{period}'
        result_df[column_name] = (
            result_df[trend_buy_col] - result_df[f'{trend_buy_col}_{period}']
        ) / (result_df[f'{trend_buy_col}_{period}'] + epsilon)
    
    return result_df

import pandas as pd
import numpy as np

def add_time_range_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time range indicator columns to a DataFrame with datetime index.
    Creates binary columns (1.0 or 0.0) for different trading hour ranges.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame with datetime index
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added time range indicator columns
    """
    # Create a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Add 9:30-10:00 indicator
    result_df['hours_930_10'] = np.where(
        ((result_df.index.hour == 9) & (result_df.index.minute >= 30)) | 
        (result_df.index.hour == 10),
        1.0, 0.0
    ).astype(float)
    
    # Add hourly indicators for 10:00 through 16:00
    hour_ranges = {
        'hours_1000_1100': 10,
        'hours_1100_1200': 11,
        'hours_1200_1300': 12,
        'hours_1300_1400': 13,
        'hours_1400_1500': 14,
        'hours_1500_1600': 15
    }
    
    for column_name, hour in hour_ranges.items():
        result_df[column_name] = np.where(
            result_df.index.hour == hour,
            1.0, 0.0
        ).astype(float)
    
    return result_df

import pandas as pd

def add_candle_pattern_features(df: pd.DataFrame, 
                              open_col: str = 'open', 
                              close_col: str = 'close') -> pd.DataFrame:
    """
    Add candlestick pattern features to a DataFrame.
    Creates binary columns (1 or 0) for different candlestick patterns.
    If either open_col or close_col contains 'raw', adds '_raw' suffix to feature names.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing the price data
    open_col : str, optional
        Name of the opening price column (default: 'open')
    close_col : str, optional
        Name of the closing price column (default: 'close')
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added candlestick pattern features
    """
    # Create a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Determine if we should use raw suffix
    use_raw_suffix = 'raw' in open_col.lower() or 'raw' in close_col.lower()
    suffix = '_raw' if use_raw_suffix else ''
    
    # Single candle patterns
    result_df[f'green_candle{suffix}'] = (result_df[close_col] > result_df[open_col]).astype(int)
    result_df[f'red_candle{suffix}'] = (result_df[close_col] < result_df[open_col]).astype(int)
    
    # Two consecutive candles patterns
    result_df[f'green_candle_2{suffix}'] = (
        (result_df[close_col] > result_df[open_col]) &
        (result_df[close_col].shift(1) > result_df[open_col].shift(1))
    ).astype(int)
    
    result_df[f'red_candle_2{suffix}'] = (
        (result_df[close_col] < result_df[open_col]) &
        (result_df[close_col].shift(1) < result_df[open_col].shift(1))
    ).astype(int)
    
    # Three consecutive candles patterns
    result_df[f'green_candle_3{suffix}'] = (
        (result_df[close_col] > result_df[open_col]) &
        (result_df[close_col].shift(1) > result_df[open_col].shift(1)) &
        (result_df[close_col].shift(2) > result_df[open_col].shift(2))
    ).astype(int)
    
    result_df[f'red_candle_3{suffix}'] = (
        (result_df[close_col] < result_df[open_col]) &
        (result_df[close_col].shift(1) < result_df[open_col].shift(1)) &
        (result_df[close_col].shift(2) < result_df[open_col].shift(2))
    ).astype(int)
    
    return result_df

import pandas as pd
import talib

def add_comprehensive_slope_features(df: pd.DataFrame, 
                      close_col: str = 'close',
                      close_raw_col: str = 'close_raw') -> pd.DataFrame:
    """
    Add slope-related features including KAMA calculations to a DataFrame.
    Creates features for both regular and raw close prices.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing the price data
    close_col : str, optional
        Name of the regular close price column (default: 'close')
    close_raw_col : str, optional
        Name of the raw close price column (default: 'close_raw')
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added slope and KAMA features
    """
    # Create a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Define the slope periods
    slope_periods = [2, 3, 4, 5, 10, 15, 20, 25, 50, 100, 200]
    
    # Calculate slope features for regular close price
    for period in slope_periods:
        # Regular slope features
        result_df[f'slope_{period}_positive'] = (
            talib.LINEARREG_SLOPE(result_df[close_col], timeperiod=period) > 0
        ).astype(int)
    
    # Calculate KAMA for regular close price
    result_df['KAMA_4'] = talib.KAMA(result_df[close_col], timeperiod=4)
    
    # Calculate KAMA slope features for regular close price
    for period in slope_periods:
        slope = talib.LINEARREG_SLOPE(result_df['KAMA_4'], timeperiod=period)
        result_df[f'kama_slope_{period}_positive'] = (slope > 0).astype(int)
    
    # Calculate slope features for raw close price
    for period in slope_periods:
        # Raw slope features
        result_df[f'slope_{period}_positive_raw'] = (
            talib.LINEARREG_SLOPE(result_df[close_raw_col], timeperiod=period) > 0
        ).astype(int)
    
    # Calculate KAMA for raw close price
    result_df['KAMA_4_raw'] = talib.KAMA(result_df[close_raw_col], timeperiod=4)
    
    # Calculate KAMA slope features for raw close price
    for period in slope_periods:
        slope = talib.LINEARREG_SLOPE(result_df['KAMA_4_raw'], timeperiod=period)
        result_df[f'kama_slope_{period}_positive_raw'] = (slope > 0).astype(int)
    
    return result_df

import pandas as pd
import talib
import numpy as np

def compute_trailing_hma_crossing_rate(df, short_window=5, long_window=20, trailing_window=300):
    """
    Calculates the trailing mean crossing rate of Hull Moving Averages (HMA) over a specified window.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the 'close' price column.
    - short_window (int): Period for the short HMA.
    - long_window (int): Period for the long HMA.
    - trailing_window (int): Number of previous time steps to calculate the mean crossing rate.

    Returns:
    - pd.Series: Trailing mean crossing rate.
    """
    
    # Ensure 'close' column exists
    if 'close' not in df.columns:
        raise ValueError("DataFrame must contain a 'close' column.")
    
    # Calculate Hull Moving Averages
    df['HMA_short'] = calculate_hma(df['close'], period=short_window)
    df['HMA_long'] = calculate_hma(df['close'], period=long_window)
    
    # Calculate the difference between short and long HMAs
    df['hma_diff'] = df['HMA_short'] - df['HMA_long']
    
    # Compute the sign of the differences
    df['sign_diff'] = np.sign(df['hma_diff'])
    
    # Compute the difference of the sign to identify crossings
    df['sign_changes'] = df['sign_diff'].diff().abs()
    
    # A crossing occurs when sign_changes equals 2 (from -1 to +1 or +1 to -1)
    df['crossing'] = df['sign_changes'] == 2
    
    # Calculate trailing mean crossing rate
    df['hma_crossing_rate'] = df['crossing'].rolling(window=trailing_window, min_periods=1).mean()
    
    # Clean up intermediate columns if desired
    df.drop(['HMA_short', 'HMA_long', 'hma_diff', 'sign_diff', 'sign_changes', 'crossing'], axis=1, inplace=True)
    
    return df['hma_crossing_rate']

def compute_hma_crossing_rate(time_series, short_window=5, long_window=20):
    # Convert input to float64 numpy array
    time_series = np.array(time_series, dtype=np.float64)
    
    # Calculate Hull Moving Averages for short and long windows
    short_hma = calculate_hma(time_series, period=short_window)
    long_hma = calculate_hma(time_series, period=long_window)
    
    # Calculate the difference between short and long HMAs
    hma_diff = short_hma - long_hma
    
    # Compute the sign of the differences
    sign_diff = np.sign(hma_diff)
    
    # Compute the difference of the sign to identify crossings
    sign_changes = np.diff(sign_diff)
    
    # Count the number of times the sign changes (crossings)
    crossings = np.where(sign_changes != 0)[0]
    
    # Calculate mean crossing rate
    mean_crossing_rate = len(crossings) / len(time_series) if len(time_series) > 0 else np.nan
    return mean_crossing_rate

def compute_trailing_coefficient_of_variation(time_series, trailing_window=300):
    """
    Calculates the trailing coefficient of variation over a specified window.

    Parameters:
    - time_series (pd.Series or np.ndarray): Input time series.
    - trailing_window (int): Number of previous time steps to calculate the coefficient of variation.

    Returns:
    - pd.Series: Trailing coefficient of variation.
    """
    rolling_mean = time_series.rolling(window=trailing_window, min_periods=1).mean()
    rolling_std = time_series.rolling(window=trailing_window, min_periods=1).std()
    cov = rolling_std / rolling_mean
    cov = cov.replace([np.inf, -np.inf], np.nan)  # Handle division by zero
    return cov

def calculate_crossing_rates_for_decile(df_with_signals, decile_results):
    """
    Calculate HMA crossing rates for all symbols in a decile
    
    Parameters:
    df_with_signals (pd.DataFrame): Raw dataframe containing close_raw prices
    decile_results (BacktestResults): Results object containing symbol metrics
    
    Returns:
    dict: Dictionary mapping symbols to their HMA crossing rates
    """
    crossing_rates = {}
    
    for symbol in decile_results.symbol_metrics['symbol']:
        # Get the symbol's data from the signals dataframe
        symbol_data = df_with_signals[df_with_signals['symbol'] == symbol]
        if not symbol_data.empty:
            # Calculate crossing rate from the close prices
            crossing_rate = compute_hma_crossing_rate(symbol_data['close_raw'].values)
            crossing_rates[symbol] = crossing_rate
        else:
            crossing_rates[symbol] = 0.0  # Default value if no data for this symbol
            
    return crossing_rates



def add_entropy_features(df: pd.DataFrame,
                        close_col: str = 'close',
                        high_col: str = 'high',
                        low_col: str = 'low',
                        trend_buy_col: str = 'trend_buy') -> pd.DataFrame:
    """
    Add entropy-related features including HMA, ADX, crossing rates, and coefficient of variation.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing the price data
    close_col : str, optional
        Name of the close price column (default: 'close')
    high_col : str, optional
        Name of the high price column (default: 'high')
    low_col : str, optional
        Name of the low price column (default: 'low')
    trend_buy_col : str, optional
        Name of the trend buy column (default: 'trend_buy')
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added entropy-related features
    """
    # Create a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Calculate HMA 20
    result_df['close_hma_20'] = calculate_hma(
        result_df[close_col].to_numpy(), 
        period=20
    )
    
    # Calculate ADX with different periods
    result_df['ADX_14'] = talib.ADX(
        result_df[high_col],
        result_df[low_col],
        result_df[close_col],
        timeperiod=14
    )
    
    result_df['ADX_30'] = talib.ADX(
        result_df[high_col],
        result_df[low_col],
        result_df[close_col],
        timeperiod=30
    )
    
    # Calculate trailing mean of ADX
    result_df['ADX_14_mean_300'] = result_df['ADX_14'].rolling(window=400).mean()
    result_df['ADX_30_mean_300'] = result_df['ADX_30'].rolling(window=400).mean()
    
    # Calculate HMA crossing rates
    result_df['hma_crossing_rate_5_20'] = compute_trailing_hma_crossing_rate(
        result_df, 
        short_window=5, 
        long_window=20, 
        trailing_window=450
    )
    
    result_df['hma_crossing_rate_15_40'] = compute_trailing_hma_crossing_rate(
        result_df, 
        short_window=15, 
        long_window=40, 
        trailing_window=450
    )
    
    # Calculate coefficient of variation
    result_df['cov_trailing_trend_buy_400'] = compute_trailing_coefficient_of_variation(
        result_df[trend_buy_col], 
        trailing_window=400
    )
    
    return result_df

