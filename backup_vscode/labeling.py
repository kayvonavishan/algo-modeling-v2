import numpy as np
import pandas as pd
from contextlib import contextmanager
from io import StringIO
import sys
from scipy.signal import savgol_filter
import talib
import pandas as pd
import numpy as np

def prepare_labels(df):
    """
    Prepares labels by encoding them using LabelEncoder while preserving special values (-100, -999).
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing a 'label' column to be encoded
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with encoded labels, where special values (-100, -999) are preserved
    le : sklearn.preprocessing.LabelEncoder
        Fitted LabelEncoder object that can be used for inverse transformation if needed
    """
    from sklearn.preprocessing import LabelEncoder
    
    # Create a copy to avoid modifying the original DataFrame
    df_encoded = df.copy()
    
    # Instantiate LabelEncoder
    le = LabelEncoder()
    
    # Create mask to exclude special values (-100, -999)
    mask = ~df_encoded['label'].isin([-100, -999])
    
    # Fit LabelEncoder on valid labels only
    le.fit(df_encoded.loc[mask, 'label'])
    
    # Transform valid labels using the fitted encoder
    temp_labels = df_encoded.loc[mask, 'label']
    df_encoded.loc[mask, 'label'] = le.transform(temp_labels).astype(df_encoded['label'].dtype)
    
    return df_encoded, le

# Now use the same label encoder on the prediction data
def encode_prediction_labels(df, le):
    """
    Encodes labels of a new DataFrame using a pre-fitted LabelEncoder.
    Preserves special values (-100, -999).
    
    Parameters:
    -----------
    df : pandas.DataFrame
        New DataFrame containing labels to be encoded
    le : sklearn.preprocessing.LabelEncoder
        Pre-fitted LabelEncoder from the training data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with encoded labels
    """
    # Create a copy to avoid modifying the original DataFrame
    df_encoded = df.copy()
    
    # Create mask to exclude special values
    mask = ~df_encoded['label'].isin([-100, -999])
    
    # Transform valid labels using the pre-fitted encoder
    temp_labels = df_encoded.loc[mask, 'label']
    df_encoded.loc[mask, 'label'] = le.transform(temp_labels).astype(df_encoded['label'].dtype)
    
    return df_encoded


def calculate_savgol_atr_percent(df, 
                               window_length,
                               polyorder,
                               atr_period=14,
                               sma_period=150,
                               use_intraday_atr=False,
                               calc_intraday_atr=None):
    """
    Calculate ATR percent change using Savitzky-Golay filtered prices.
    """
    # Create a copy of the input data to avoid modifications
    temp_df = pd.DataFrame(index=df.index)  # Ensure same index as input df
    
    # Ensure window_length is appropriate
    if window_length > len(df):
        window_length = len(df) if len(df) % 2 != 0 else len(df) - 1
    
    # Apply Savitzky-Golay filter to price data
    temp_df['high_prices_sg'] = savgol_filter(df['high_raw'], window_length=window_length, polyorder=polyorder)
    temp_df['low_prices_sg'] = savgol_filter(df['low_raw'], window_length=window_length, polyorder=polyorder)
    temp_df['close_prices_sg'] = savgol_filter(df['close_raw'], window_length=window_length, polyorder=polyorder)

    # Add the close_sav_gol column to the input DataFrame
    df['close_sav_gol'] = savgol_filter(df['close_raw'], window_length=window_length, polyorder=polyorder)
    
    # Calculate ATR based on the smoothed prices
    if use_intraday_atr and calc_intraday_atr is not None:
        atr = calc_intraday_atr(
            df=temp_df,
            col_high='high_prices_sg',
            col_low='low_prices_sg',
            col_close='close_prices_sg',
            atr_period=atr_period,
            debug=False
        )
    else:
        atr = pd.Series(
            talib.ATR(
                temp_df['high_prices_sg'].values,
                temp_df['low_prices_sg'].values,
                temp_df['close_prices_sg'].values,
                timeperiod=atr_period
            ),
            index=df.index
        )
    
    # Calculate SMA of smoothed close prices
    close_sma = pd.Series(
        talib.SMA(temp_df['close_prices_sg'], timeperiod=sma_period),
        index=df.index
    )
    
    # Calculate final ATR percent change
    # Create a Series with the same index as the input DataFrame
    atr_percent_change = pd.Series(np.nan, index=df.index)
    
    # Calculate only where both values are valid
    valid_data = ~(pd.isna(atr) | pd.isna(close_sma) | (close_sma == 0))
    atr_percent_change.loc[valid_data] = (atr[valid_data] / close_sma[valid_data]) * 100
    
    return atr_percent_change

class DebugOutputCollector:
    """
    Collects debug output and organizes it into sections for later rendering.
    """
    def __init__(self):
        self.sections = []
        self.current_section = []
        self.current_section_title = None
    
    def add_section(self, title):
        """Start a new section with the given title"""
        if self.current_section:
            self.sections.append((self.current_section_title, self.current_section))
        self.current_section = []
        self.current_section_title = title
    
    def write(self, text):
        """Add text to the current section"""
        self.current_section.append(text)
    
    def finalize(self):
        """Complete the current section if any"""
        if self.current_section:
            self.sections.append((self.current_section_title, self.current_section))
    
    @contextmanager
    def capture_output(self):
        """Context manager to capture stdout"""
        stdout = sys.stdout
        string_io = StringIO()
        sys.stdout = string_io
        try:
            yield
        finally:
            sys.stdout = stdout
            self.current_section.append(string_io.getvalue())




def create_date_mask(df_index, debug_start_date=None, debug_end_date=None):
    """
    Create a date mask for debugging, handling timezone-aware dates properly.
    
    Parameters:
    -----------
    df_index : pandas.DatetimeIndex
        The index of the DataFrame
    debug_start_date : datetime or str, optional
        Start date for debugging
    debug_end_date : datetime or str, optional
        End date for debugging
        
    Returns:
    --------
    pandas.Series
        Boolean mask for date filtering
    """
    # If no dates provided, include all dates
    if debug_start_date is None and debug_end_date is None:
        return pd.Series(True, index=df_index)
    
    # Create base mask
    date_mask = pd.Series(True, index=df_index)
    
    # Handle start date if provided
    if debug_start_date is not None:
        # Convert to Timestamp and localize timezone if needed
        start_ts = pd.Timestamp(debug_start_date)
        if df_index.tz is not None and start_ts.tz is None:
            start_ts = start_ts.tz_localize(df_index.tz)
        date_mask &= (df_index >= start_ts)
    
    # Handle end date if provided
    if debug_end_date is not None:
        # Convert to Timestamp and localize timezone if needed
        end_ts = pd.Timestamp(debug_end_date)
        if df_index.tz is not None and end_ts.tz is None:
            end_ts = end_ts.tz_localize(df_index.tz)
        date_mask &= (df_index <= end_ts)
        
    return date_mask



def print_sample_debug(idx, date, close_prices, atr_percent_changes, boundaries_positive, 
                    boundaries_negative, percent_changes, look_ahead_window, num_classes_per_side,
                    symbol=None, debug_symbol=None,  # Default to None
                    upper_coeff_smooth=3.0, lower_coeff_smooth=3.0,  # Added coefficients
                    quality_score=None, use_calculate_path_quality_score=False):
    """Extended debug print function to include quality score information when enabled"""
    # Only print debug info if current symbol matches debug symbol
    if symbol != debug_symbol:
        return
        
    print(f"\nAnalyzing {debug_symbol} sample at {date}:")
    print(f"Close price: {close_prices[idx]:.2f}")
    print(f"ATR %: {atr_percent_changes[idx]:.2f}")
    
    if use_calculate_path_quality_score and quality_score is not None:
        print(f"Path Quality Score: {quality_score:.3f}")
    
    print("\nBarrier Calculation:")
    print(f"1. Base ATR % = {atr_percent_changes[idx]:.2f}")
    print(f"2. Upper coefficient = {upper_coeff_smooth}")
    print(f"3. Lower coefficient = {lower_coeff_smooth}")
    print(f"4. Max upper barrier = ATR % * upper_coeff = {atr_percent_changes[idx]:.2f} * {upper_coeff_smooth} = {atr_percent_changes[idx] * upper_coeff_smooth:.2f}%")
    print(f"5. Max lower barrier = ATR % * lower_coeff = {atr_percent_changes[idx]:.2f} * {lower_coeff_smooth} = {atr_percent_changes[idx] * lower_coeff_smooth:.2f}%")
    print(f"6. Class barriers are calculated as fractions (1/{num_classes_per_side} to 1) of max barriers")
    
    print("\nBarriers:")
    for cls in range(num_classes_per_side):
        fraction = (cls + 1) / num_classes_per_side
        print(f"Class {cls+1} ({fraction:.2f} of max): {boundaries_positive[idx, cls]:.2f}% / {boundaries_negative[idx, cls]:.2f}%")
    
    print("\nFuture price changes:")
    for t in range(look_ahead_window):
        if not np.isnan(percent_changes[idx, t]):
            print(f"t+{t+1}: {percent_changes[idx, t]:.2f}%")
        else:
            print(f"t+{t+1}: NaN")

def calculate_composite_score(percent_changes, volatility_unit, quality_multiplier_strength=1.0):
    """
    quality_multiplier_strength controls how much effect the quality multiplier has:
    - 1.0 means normal effect (0.5 to 1.5 range)
    - 2.0 means doubled effect (0.0 to 2.0 range)
    - 0.5 means halved effect (0.75 to 1.25 range)
    
    Parameters:
    percent_changes: array of price changes
    volatility_unit: float, the atr_percent_change_for_labeling_sav_gol value
    quality_multiplier_strength: float, controls effect strength
    """
    # Remove any NaN values for calculations
    valid_changes = percent_changes[~np.isnan(percent_changes)]
    
    if len(valid_changes) == 0:
        return 0, {
            'path_volatility': 0,
            'time_above_entry': 0,
            'max_drawdown': 0,
            'scaled_drawdown': 0,  # Added this field
            'base_multiplier': 0,
            'quality_multiplier': 0,
            'final_change': 0.0,
            'cumulative_returns': [],
            'rolling_max': [],
            'drawdowns': [],
            'drawdown_penalty': 1.0,
            'alpha': 10.0,
            'volatility_unit': 0.0  # Added this field
        }
    
    # Get final price change to determine direction
    final_change = valid_changes[-1]
    direction = np.sign(final_change)
    
    # Calculate path quality components
    path_volatility = np.std(valid_changes) if len(valid_changes) > 1 else 0
    time_above_entry = np.mean(valid_changes > 0)
    
    # Calculate drawdown
    cumulative_returns = np.cumprod(1 + valid_changes / 100)
    rolling_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = np.abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0
    
    # Scale drawdown by volatility unit
    scaled_drawdown = max_drawdown / volatility_unit if volatility_unit > 0 else max_drawdown
    
    # Exponential penalty using scaled drawdown
    alpha = 250.0  # Tune as desired
    drawdown_penalty = np.exp(-alpha * scaled_drawdown)
    
    # Base multiplier (0.5..1.5 range)
    base_multiplier = 0.5 + np.clip(
        0.2 * (1 / (1 + path_volatility)) +
        0.2 * (2 * abs(time_above_entry - 0.5)) +
        0.6 * drawdown_penalty,
        0, 1
    )
    
    # Adjust multiplier range based on strength parameter, with minimum bound
    quality_multiplier = max(0.1, 1.0 + (base_multiplier - 1.0) * quality_multiplier_strength)
    
    # Modified composite score calculation to handle negative moves differently
    if final_change > 0:
        composite_score = final_change * quality_multiplier
    else:
        composite_score = final_change * (2 - quality_multiplier)
    
    # Collect debug information
    debug_info = {
        'path_volatility': path_volatility,
        'time_above_entry': time_above_entry,
        'max_drawdown': max_drawdown,
        'scaled_drawdown': scaled_drawdown,  # Added this field
        'base_multiplier': base_multiplier,
        'quality_multiplier': quality_multiplier,
        'final_change': final_change,
        'cumulative_returns': cumulative_returns.tolist(),
        'rolling_max': rolling_max.tolist(),
        'drawdowns': drawdowns.tolist(),
        'drawdown_penalty': drawdown_penalty,
        'alpha': alpha,
        'volatility_unit': volatility_unit  # Added this field
    }
    
    return composite_score, debug_info


def calculate_trajectory_score(percent_changes, volatility_unit):
    """
    Calculate trajectory score using split windows:
    - First half of window for direction
    - Full window for trajectory quality
    
    Parameters:
    -----------
    percent_changes: array of price changes
    volatility_unit: float, the ATR percent change value for scaling
    
    Returns:
    --------
    float: trajectory score (sign indicates direction, magnitude indicates strength)
    dict: debug information
    """
    # Remove any NaN values for calculations
    valid_changes = percent_changes[~np.isnan(percent_changes)]
    
    if len(valid_changes) == 0:
        return 0, {
            'path_volatility': 0,
            'time_above_entry_direction': 0.5,  # Initialize for empty case
            'time_above_entry_full': 0.5,
            'max_drawdown': 0,
            'scaled_drawdown': 0,
            'drawdown_penalty': 1.0,
            'weighted_sum': 0,
            'direction': 0,
            'trajectory_score': 0
        }
    
    # Calculate direction using first half of the window
    direction_window = valid_changes[:len(valid_changes)//2]
    
    # Initialize direction variables
    time_above_entry = 0.5  # Default neutral value
    direction = 0
    
    if len(direction_window) > 0:
        time_above_entry = np.mean(direction_window > 0)
        direction = 2 * (time_above_entry - 0.5)
        
        # If direction is neutral, use final cumulative return of direction window
        if abs(direction) < 1e-6:
            cumulative_return = np.prod(1 + direction_window/100) - 1
            direction = 1 if cumulative_return > 0 else -1
    else:
        direction = 0
    
    # Calculate quality components using full window
    path_volatility = np.std(valid_changes) if len(valid_changes) > 1 else 0
    time_above_entry_full = np.mean(valid_changes > 0)  # For quality calculation
    
    # Calculate drawdown using full window
    cumulative_returns = np.cumprod(1 + valid_changes/100)
    rolling_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = np.abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0
    scaled_drawdown = max_drawdown / volatility_unit if volatility_unit > 0 else max_drawdown
    
    # Calculate drawdown penalty
    alpha = 250.0
    drawdown_penalty = np.exp(-alpha * scaled_drawdown)
    
    # Calculate weighted sum using full window metrics
    weighted_sum = (
        0.2 * (1 / (1 + path_volatility)) +
        0.4 * (2 * abs(time_above_entry_full - 0.5)) +  # Using full window for quality
        0.4 * drawdown_penalty
    )
    
    # Calculate final trajectory score
    trajectory_score = direction * weighted_sum * 100
    
    debug_info = {
        'path_volatility': path_volatility,
        'time_above_entry_direction': time_above_entry,  # From direction window
        'time_above_entry_full': time_above_entry_full,  # From full window
        'max_drawdown': max_drawdown,
        'scaled_drawdown': scaled_drawdown,
        'drawdown_penalty': drawdown_penalty,
        'weighted_sum': weighted_sum,
        'direction': direction,
        'trajectory_score': trajectory_score
    }
    
    return trajectory_score, debug_info



def assign_classes_composite(composite_scores, num_classes_per_side):
    """
    Assign classes based on composite scores.
    """
    labels = np.zeros_like(composite_scores, dtype=int)
    
    # Handle zero scores
    zero_mask = (composite_scores == 0)
    labels[zero_mask] = -100  # Use same convention as original code for invalid/zero cases
    
    # Separate positive and negative scores
    pos_scores = composite_scores[composite_scores > 0]
    neg_scores = composite_scores[composite_scores < 0]
    
    if len(pos_scores) > 0:
        # Calculate percentile boundaries for positive side
        pos_boundaries = np.percentile(
            pos_scores, 
            np.linspace(0, 100, num_classes_per_side + 1)
        )
        
        # Assign positive classes
        for i in range(num_classes_per_side):
            mask = (composite_scores > pos_boundaries[i]) & \
                (composite_scores <= pos_boundaries[i + 1])
            labels[mask] = i + 1
    
    if len(neg_scores) > 0:
        # Calculate percentile boundaries for negative side
        neg_boundaries = np.percentile(
            abs(neg_scores), 
            np.linspace(0, 100, num_classes_per_side + 1)
        )
        
        # Assign negative classes
        for i in range(num_classes_per_side):
            mask = (composite_scores < -neg_boundaries[i]) & \
                (composite_scores >= -neg_boundaries[i + 1])
            labels[mask] = -(i + 1)
    
    return labels



def label_financial_data(
    ticker_df,
    symbol='AAPL',
    num_classes_per_side=3,
    look_ahead_window=5,
    upper_coeff_smooth=1.0,
    lower_coeff_smooth=1.0,
    use_next_day_prices=False,
    use_calculate_path_quality_score=False,
    use_trajectory_score=False,  
    quality_multiplier_strength=5.0,
    labeling_debugging_mode=False,
    debug_symbol=None,
    debug_start_date=None,
    debug_end_date=None
):
    """
    Label financial time series data based on future price movements.
    
    Parameters
    ----------
    ticker_df : pandas.DataFrame
        DataFrame containing financial data with required columns:
        - close_sav_gol: smoothed close prices
        - atr_percent_change_for_labeling_sav_gol: smoothed ATR percent changes
    num_classes_per_side : int, default 3
        Number of classes for both positive and negative movements
    look_ahead_window : int, default 5
        Number of future periods to consider
    upper_coeff_smooth : float, default 1.0
        Coefficient for upper barrier calculation
    lower_coeff_smooth : float, default 1.0
        Coefficient for lower barrier calculation
    use_next_day_prices : bool, default False
        Whether to use prices from the next trading day
    use_calculate_path_quality_score : bool, default False
        Whether to use path quality scoring instead of barrier-based classification
    quality_multiplier_strength : float, default 5.0
        Strength parameter for quality score calculation
    labeling_debugging_mode : bool, default False
        Enable detailed debugging output
    debug_symbol : str, optional
        Symbol to debug when in debugging mode
    debug_start_date : datetime, optional
        Start date for debugging period
    debug_end_date : datetime, optional
        End date for debugging period
    
    Returns
    -------
    pandas.DataFrame
        Original DataFrame with additional columns:
        - label: assigned class labels (-num_classes_per_side to num_classes_per_side)
        - time: extracted time from index (for statistics)
    """

    
    # Data existence checks
    if ticker_df is None or len(ticker_df) == 0:
        print("WARNING: ticker_df is empty or None!")
    else:
        print(f"\nData Overview:")
        print(f"Total rows in ticker_df: {len(ticker_df)}")
        print(f"Date range in data: {ticker_df.index.min()} to {ticker_df.index.max()}")
        
        # Check for data in debug date range
        if debug_start_date is not None and debug_end_date is not None:
            # Convert debug dates to timezone-aware timestamps matching the data
            tz = ticker_df.index.tz
            try:
                start_ts = pd.Timestamp(debug_start_date).tz_localize(tz)
                end_ts = pd.Timestamp(debug_end_date).tz_localize(tz)
                
                mask = (ticker_df.index >= start_ts) & (ticker_df.index <= end_ts)
                filtered_data = ticker_df[mask]
                print(f"\nData in debug date range:")
                print(f"Number of rows: {len(filtered_data)}")
                if len(filtered_data) > 0:
                    print(f"First timestamp: {filtered_data.index.min()}")
                    print(f"Last timestamp: {filtered_data.index.max()}")
                else:
                    print("WARNING: No data found in specified date range!")
            except Exception as e:
                print(f"Error checking date range: {str(e)}")
                
        # Check required columns
        required_columns = ['close', 'close_sav_gol', 'atr_percent_change_for_labeling_sav_gol']
        missing_columns = [col for col in required_columns if col not in ticker_df.columns]
        if missing_columns:
            print(f"\nWARNING: Missing required columns: {missing_columns}")
        else:
            print("\nAll required columns present")
            
    print("=" * 40 + "\n")


    #############################
    # DATA PREP 
    #############################
    debug_output = DebugOutputCollector()

    # Drop rows where 'close' is NaN without resetting the index
    ticker_df_adjusted = ticker_df.copy()
    ticker_df_adjusted = ticker_df_adjusted.dropna(subset=['close'])

    # Initialize parameters
    ticker_df_adjusted['label'] = -999

    # Set default value for use_calculate_path_quality_score if not defined
    if 'use_calculate_path_quality_score' not in locals():
        use_calculate_path_quality_score = False

    # Get debug indices if debugging
    if labeling_debugging_mode and symbol == debug_symbol:
        
        with debug_output.capture_output():
            debug_output.add_section("Initial Analysis")
            
            date_mask = create_date_mask(ticker_df_adjusted.index, debug_start_date, debug_end_date)
            debug_indices = np.where(date_mask)[0]
            if len(debug_indices) > 0:
                print(f"\nDebugging for {debug_symbol} in date range:",
                    f"{debug_start_date if debug_start_date is not None else 'start'} to",
                    f"{debug_end_date if debug_end_date is not None else 'end'}")
                print(f"Number of samples in debug range: {len(debug_indices)}")


    # Prepare data
    close_prices = ticker_df_adjusted['close_sav_gol'].to_numpy()
    atr_percent_changes = np.abs(
        ticker_df_adjusted['atr_percent_change_for_labeling_sav_gol'].to_numpy()
    )

    # Handle NaN values using recommended methods
    atr_percent_changes_series = pd.Series(atr_percent_changes)
    atr_percent_changes_filled = atr_percent_changes_series.ffill().bfill()
    atr_percent_changes = atr_percent_changes_filled.to_numpy()

    n = len(close_prices)

    # Calculate barriers for smoothed data
    upper_barrier_smooth = atr_percent_changes * upper_coeff_smooth
    lower_barrier_smooth = -atr_percent_changes * lower_coeff_smooth
    abs_lower_barrier_smooth = np.abs(lower_barrier_smooth)

    # Before calculating percent_changes, modify the shifted_prices calculation:
    shifted_prices = np.full((n, look_ahead_window), np.nan)

    for shift in range(1, look_ahead_window + 1):
        shifted = np.roll(close_prices, -shift)
        
        if not use_next_day_prices:
            # Get timestamps for current and shifted points
            current_times = pd.Series(ticker_df_adjusted.index)
            shifted_times = current_times.shift(-shift)
            
            # Create mask for same-day prices only
            same_day_mask = (
                current_times.dt.date == shifted_times.dt.date
            )
            
            # Adjust shifted array to match mask length
            shifted[:len(same_day_mask)][~same_day_mask] = np.nan
        
        shifted_prices[:, shift - 1] = shifted

    # Update the original end-of-array masking
    mask = (
        np.arange(n)[:, None] + np.arange(1, look_ahead_window + 1)
    ) > n - 1
    shifted_prices[mask] = np.nan

    # Compute percent changes
    percent_changes = (
        (shifted_prices - close_prices[:, None])
        / close_prices[:, None]
        * 100
    )

    # Compute class boundaries
    coefficients_positive = np.linspace(1/num_classes_per_side, 1, num_classes_per_side)
    coefficients_negative = np.linspace(1/num_classes_per_side, 1, num_classes_per_side)

    boundaries_positive = upper_barrier_smooth[:, None] * coefficients_positive[None, :]
    boundaries_negative = lower_barrier_smooth[:, None] * coefficients_negative[None, :]

    # Initialize labels
    labels = np.full(n, -999, dtype=int)
    max_class = num_classes_per_side

    #############################
    # IF USING PATH QUALITY SCORE 
    #############################

    if use_calculate_path_quality_score:
        print("Debug: Entering Path Quality Score Analysis section")
        debug_output.add_section("Path Quality Score Analysis")

        # Calculate composite scores for all samples
        composite_scores = np.zeros(n)
        debug_infos = [{} for _ in range(n)]  # Initialize list to store debug info per sample
        
        # Get volatility units for each sample
        volatility_units = ticker_df_adjusted['atr_percent_change_for_labeling_sav_gol'].to_numpy()
        
        for i in range(n):
            score, debug_info = calculate_composite_score(
                percent_changes[i, :],
                volatility_unit=volatility_units[i],
                quality_multiplier_strength=quality_multiplier_strength
            )
            composite_scores[i] = score
            debug_infos[i] = debug_info  # Store debug info for each sample

        # ADDING LOGIC TO HANDLE CASES WHERE SCORE IS VERY CLOSE TO 0 (ROUNDING ISSUE) 
        ZERO_THRESHOLD = 0.1  # Adjust this threshold as needed
        
        # Handle effectively zero scores
        tiny_score_mask = (np.abs(composite_scores) <= ZERO_THRESHOLD) & (composite_scores != 0)
        tiny_scores = composite_scores[tiny_score_mask]
        
        if np.any(tiny_score_mask):
            # Scale tiny scores to range [0.1, 0.2] while preserving sign
            scaled_scores = 0.1 + 0.1 * (np.abs(tiny_scores) / ZERO_THRESHOLD)
            composite_scores[tiny_score_mask] = np.sign(tiny_scores) * scaled_scores
        
        # Keep the exact zero handling as before
        exact_zero_mask = (composite_scores == 0)
        composite_scores[exact_zero_mask] = 0
        
        
        # Calculate boundaries using only first 60% of data
        train_size = int(n * 0.6)
        train_scores = composite_scores[:train_size]
        train_pos_scores = train_scores[train_scores > 0]
        train_neg_scores = train_scores[train_scores < 0]
        
        # Calculate boundaries from training data
        if len(train_pos_scores) > 0:
            pos_boundaries = np.percentile(
                train_pos_scores,
                np.linspace(0, 100, num_classes_per_side + 1)
            )
        else:
            pos_boundaries = np.array([0] * (num_classes_per_side + 1))
            
        if len(train_neg_scores) > 0:
            neg_boundaries = np.percentile(
                abs(train_neg_scores),
                np.linspace(0, 100, num_classes_per_side + 1)
            )
        else:
            neg_boundaries = np.array([0] * (num_classes_per_side + 1))
        
        # Print statistics if debugging
        if labeling_debugging_mode and symbol == debug_symbol:
            with debug_output.capture_output():
                print("\nComposite Score Statistics:")
                print(f"Total samples: {n}")
                print(f"Training samples used for boundaries: {train_size}")
                print(f"Mean composite score (all data): {np.mean(composite_scores):.3f}")
                print(f"Mean composite score (train only): {np.mean(train_scores):.3f}")
                print(f"Min composite score: {np.min(composite_scores):.3f}")
                print(f"Max composite score: {np.max(composite_scores):.3f}")
                
                print(f"\nTraining Data Distribution:")
                print(f"Positive scores in train: {len(train_pos_scores)}")
                print(f"Negative scores in train: {len(train_neg_scores)}")
                
                print("\nBoundary Calculation Explanation:")
                print("1. Using first 60% of data to determine boundaries")
                print("2. Scores are separated into positive and negative groups")
                print(f"3. For {num_classes_per_side} classes, creating {num_classes_per_side+1} boundaries")
                print(f"4. Percentiles used: {np.linspace(0, 100, num_classes_per_side + 1)}")
                
                if len(train_pos_scores) > 0:
                    print("\nPositive Score Boundaries (from training data):")
                    for i in range(num_classes_per_side):
                        n_samples = np.sum((train_pos_scores > pos_boundaries[i]) & 
                                        (train_pos_scores <= pos_boundaries[i+1]))
                        pct_samples = (n_samples / len(train_pos_scores)) * 100
                        print(f"Class {i+1}: {pos_boundaries[i]:.3f} to {pos_boundaries[i+1]:.3f}")
                        print(f"        Contains {n_samples} training samples ({pct_samples:.1f}% of positive training samples)")
                
                if len(train_neg_scores) > 0:
                    print("\nNegative Score Boundaries (from training data):")
                    for i in range(num_classes_per_side):
                        n_samples = np.sum((abs(train_neg_scores) > neg_boundaries[i]) & 
                                        (abs(train_neg_scores) <= neg_boundaries[i+1]))
                        pct_samples = (n_samples / len(train_neg_scores)) * 100
                        print(f"Class -{i+1}: -{neg_boundaries[i]:.3f} to -{neg_boundaries[i+1]:.3f}")
                        print(f"         Contains {n_samples} training samples ({pct_samples:.1f}% of negative training samples)")
        
        # Assign labels using the training-derived boundaries
        labels = np.full_like(composite_scores, -999, dtype=int)  # NEW - Initialize to -999
        
        # Handle zero scores
        zero_mask = (composite_scores == 0)
        labels[zero_mask] = -100
        
        # Assign positive classes using training boundaries
        for i in range(num_classes_per_side):
            mask = (composite_scores > pos_boundaries[i]) & \
                (composite_scores <= pos_boundaries[i + 1])
            labels[mask] = i + 1
        
        # Assign negative classes using training boundaries
        for i in range(num_classes_per_side):
            mask = (composite_scores < -neg_boundaries[i]) & \
                (composite_scores >= -neg_boundaries[i + 1])
            labels[mask] = -(i + 1)
        

        # Enhanced debugging for individual samples
        if labeling_debugging_mode and symbol == debug_symbol:
            with debug_output.capture_output():
                date_mask = create_date_mask(ticker_df_adjusted.index, debug_start_date, debug_end_date)
                for idx in np.where(date_mask)[0]:
                    date = ticker_df_adjusted.index[idx]
                    score = composite_scores[idx]
                    label = labels[idx]
                    is_training = idx < train_size
                    debug_info = debug_infos[idx]  # Retrieve debug info for the current sample
                    
                    # Extract necessary information from debug_info
                    final_change = debug_info.get('final_change', 0.0)
                    quality_multiplier = debug_info.get('quality_multiplier', 1.0)
                    path_volatility = debug_info.get('path_volatility', 0)
                    time_above_entry = debug_info.get('time_above_entry', 0)
                    max_drawdown = debug_info.get('max_drawdown', 0)
                    base_multiplier = debug_info.get('base_multiplier', 1.0)
                    cumulative_returns = debug_info.get('cumulative_returns', [])
                    rolling_max = debug_info.get('rolling_max', [])
                    drawdowns = debug_info.get('drawdowns', [])
                    
                    print(f"\n{'='*50}")
                    print(f"Sample Analysis at {date} ({'Training' if is_training else 'Test'} Sample)")
                    print(f"{'='*50}")
                    
                    print("\n1. Price Path Analysis:")
                    print("Future price changes:")
                    for t in range(look_ahead_window):
                        if not np.isnan(percent_changes[idx, t]):
                            print(f"t+{t+1}: {percent_changes[idx, t]:.2f}%")
                        else:
                            print(f"t+{t+1}: NaN")
                    
                    print("\n2. Quality Score Components:")
                    # Original stats
                    print(f"Path Volatility: {path_volatility:.3f}")
                    print(f"Time Above Entry: {time_above_entry:.1%}")
                    print(f"Maximum Drawdown: {max_drawdown:.1%}")
                    
                    print("\nDetailed Component Calculations:")
                    
                    # A. Volatility Component
                    print("A. Volatility Component:")
                    volatility_score = (1 / (1 + path_volatility))
                    print(f"   - Formula: 1 / (1 + {path_volatility:.3f})")
                    print(f"   - Score: {volatility_score:.3f}")
                    print(f"   - Weight: 0.2 * {volatility_score:.3f} = {0.2 * volatility_score:.3f}")
                    
                    # B. Time Above Entry Component
                    print("\nB. Time Above Entry Component:")
                    print(f"   - Raw percentage: {time_above_entry:.1%}  (percentage of time price spent above entry point)")
                    print(f"   - Distance from 50%: |{time_above_entry:.3f} - 0.5| = {abs(time_above_entry - 0.5):.3f}  (how far from neutral/random movement)")
                    consistency_score = 2 * abs(time_above_entry - 0.5)
                    print(f"   - Score: 2 * {abs(time_above_entry - 0.5):.3f} = {consistency_score:.3f}  (amplifying the consistency effect)")
                    print(f"   - Weight: 0.4 * {consistency_score:.3f} = {0.4 * consistency_score:.3f}  (this component is 40% of quality multiplier)")
                    
                    # C. Drawdown Component with Detailed Calculations
                    print("\nC. Drawdown Component:")
                    print("   - Calculation Steps:")
                    # Format arrays for readability
                    formatted_cum_returns = [f"{cr:.3f}" for cr in cumulative_returns]
                    formatted_rolling_max = [f"{rm:.3f}" for rm in rolling_max]
                    formatted_drawdowns = [f"{dd:.3f}" for dd in drawdowns]
                    
                    print(f"     1. Compute cumulative returns: {formatted_cum_returns}")
                    print(f"     2. Calculate rolling maximum of cumulative returns: {formatted_rolling_max}")
                    print(f"     3. Determine drawdowns: {formatted_drawdowns}")
                    
                    volatility_unit = debug_info.get('volatility_unit', 0.0)
                    scaled_drawdown = debug_info.get('scaled_drawdown', 0.0)
                    
                    print(f"   - Raw max drawdown: {max_drawdown:.3f}")
                    print(f"   - Volatility unit: {volatility_unit:.3f}")
                    print(f"   - Scaled drawdown (max_drawdown / volatility_unit): {scaled_drawdown:.3f}")
                    
                    # -----------------------------
                    # EXPONENTIAL LOGIC WITH VOLATILITY SCALING
                    # -----------------------------
                    drawdown_penalty = debug_info.get('drawdown_penalty', 1.0)
                    alpha = debug_info.get('alpha', 10.0)
                    
                    print(f"\n   - (Exponential) Penalty Calculation:")
                    print(f"     Formula: e^(-alpha * scaled_drawdown)")
                    print(f"     => alpha = {alpha:.2f}, scaled_drawdown = {scaled_drawdown:.3f}")
                    print(f"     => Penalty = e^(-{alpha:.2f} * {scaled_drawdown:.3f}) = {drawdown_penalty:.3f}")
                    
                    # Assign exponential penalty as the drawdown score
                    drawdown_score = drawdown_penalty
                    print(f"   - (Exponential) Score: {drawdown_score:.3f}")
                    print(f"   - Weight: 0.4 * {drawdown_score:.3f} = {0.4 * drawdown_score:.3f}")
                    
                    # -----------------------------
                    # Update Weighted Sum
                    # -----------------------------
                    weighted_sum = (0.2 * volatility_score + 
                                0.4 * consistency_score +  # Fixed from 0.2 to 0.4 to match above
                                0.4 * drawdown_score)
                    
                    print("\nQuality Multiplier Calculation:")
                    print(f"1. Sum of weighted components: {weighted_sum:.3f}")
                    print(f"   - Volatility:  0.2 * {volatility_score:.3f} = {0.2 * volatility_score:.3f}")
                    print(f"   - Consistency: 0.4 * {consistency_score:.3f} = {0.4 * consistency_score:.3f}")  # Updated weight
                    print(f"   - Drawdown:    0.4 * {drawdown_score:.3f} = {0.4 * drawdown_score:.3f}")
                    
                    print(f"2. Base multiplier (0.5 to 1.5 range): {base_multiplier:.3f}")
                    print("   Explanation:")
                    print("     base_multiplier = 0.5 + np.clip(sum_of_weighted_components, 0, 1)")
                    print(f"     => 0.5 + np.clip({weighted_sum:.3f}, 0, 1) = 0.5 + {min(weighted_sum, 1.0):.3f}")
                    print(f"     => {base_multiplier:.3f}")
                    
                    print(f"3. Strength-adjusted multiplier (minimum 0.1): {quality_multiplier:.3f} (using strength={quality_multiplier_strength:.1f})")
                    print("   Explanation:")
                    print("     quality_multiplier = max(0.1, 1.0 + (base_multiplier - 1.0) * strength_multiplier_strength)")
                    intermediate_calculation = (base_multiplier - 1.0) * quality_multiplier_strength
                    print(f"     => max(0.1, 1.0 + ({base_multiplier - 1.0:.3f}) * {quality_multiplier_strength:.1f})")
                    print(f"     => max(0.1, 1.0 + {intermediate_calculation:.3f})")
                    print(f"     => {quality_multiplier:.3f}")
                    
                    print(f"4. Final Score: {final_change:.2f}% * {quality_multiplier:.3f} = {score:.3f}")
                    
                    print("\n3. Final Score Calculation:")
                    print(f"Final Price Change: {final_change:.2f}%")
                    print(f"Quality Multiplier: {quality_multiplier:.3f}")
                    print(f"Composite Score = {final_change:.2f} * {quality_multiplier:.3f} = {score:.3f}")
                    
                    print("\n4. Label Assignment:")
                    print(f"Assigned Label: {label}")
                    if label == -100:
                        print("This is an INVALID/ZERO movement")
                    elif label == -999:
                        print("This sample hasn't been assigned a valid label yet")
                    elif score > 0:
                        boundary_low = pos_boundaries[abs(label)-1]
                        boundary_high = pos_boundaries[abs(label)]
                        print(f"Why? Positive composite score {score:.3f} fell between boundaries {boundary_low:.3f} and {boundary_high:.3f}")
                    elif score < 0:
                        boundary_low = neg_boundaries[abs(label)-1]
                        boundary_high = neg_boundaries[abs(label)]
                        print(f"Why? Negative composite score {score:.3f} fell between boundaries -{boundary_high:.3f} and -{boundary_low:.3f}")
                    
                    if label > 0:
                        print(f"This is a POSITIVE movement with {label} strength out of {num_classes_per_side}")
                    elif label < 0 and label not in [-100, -999]:
                        print(f"This is a NEGATIVE movement with {abs(label)} strength out of {num_classes_per_side}")
                    else:
                        print("This is an INVALID/ZERO movement")
                    
                    print(f"\n{'='*50}")
            
        

    #############################
    # IF USING TRAJECTORY SCORE
    #############################
    elif use_trajectory_score:
        print("Debug: Entering Trajectory Score Analysis section")
        debug_output.add_section("Trajectory Score Analysis")

        # Calculate trajectory scores for all samples
        trajectory_scores = np.zeros(n)
        debug_infos = [{} for _ in range(n)]
        
        # Get volatility units for each sample
        volatility_units = ticker_df_adjusted['atr_percent_change_for_labeling_sav_gol'].to_numpy()
        
        for i in range(n):
            score, debug_info = calculate_trajectory_score(
                percent_changes[i, :],
                volatility_unit=volatility_units[i]
            )
            trajectory_scores[i] = score
            debug_infos[i] = debug_info

        # Calculate boundaries using first 60% of data
        train_size = int(n * 0.6)
        train_scores = trajectory_scores[:train_size]
        train_pos_scores = train_scores[train_scores > 0]
        train_neg_scores = train_scores[train_scores < 0]
        
        # Calculate boundaries from training data
        if len(train_pos_scores) > 0:
            pos_boundaries = np.percentile(
                train_pos_scores,
                np.linspace(0, 100, num_classes_per_side + 1)
            )
        else:
            pos_boundaries = np.array([0] * (num_classes_per_side + 1))
            
        if len(train_neg_scores) > 0:
            neg_boundaries = np.percentile(
                abs(train_neg_scores),
                np.linspace(0, 100, num_classes_per_side + 1)
            )
        else:
            neg_boundaries = np.array([0] * (num_classes_per_side + 1))

        # Initialize labels array
        labels = np.full(n, -999, dtype=int)
        
        # Handle zero/invalid scores
        zero_mask = (np.abs(trajectory_scores) < 0.1)
        labels[zero_mask] = -100
        
        # Assign positive classes
        for i in range(num_classes_per_side):
            mask = (trajectory_scores > pos_boundaries[i]) & \
                   (trajectory_scores <= pos_boundaries[i + 1])
            labels[mask] = i + 1
        
        # Assign negative classes
        for i in range(num_classes_per_side):
            mask = (trajectory_scores < -neg_boundaries[i]) & \
                   (trajectory_scores >= -neg_boundaries[i + 1])
            labels[mask] = -(i + 1)

        # Add debugging output
        if labeling_debugging_mode and symbol == debug_symbol:
            with debug_output.capture_output():
                print("\nTrajectory Score Statistics:")
                print(f"Total samples: {n}")
                print(f"Training samples: {train_size}")
                print(f"Mean trajectory score: {np.mean(trajectory_scores):.3f}")
                print(f"Min score: {np.min(trajectory_scores):.3f}")
                print(f"Max score: {np.max(trajectory_scores):.3f}")
                
                # Print sample analysis for debug range
                date_mask = create_date_mask(ticker_df_adjusted.index, debug_start_date, debug_end_date)
                for idx in np.where(date_mask)[0]:
                    debug_info = debug_infos[idx]
                    date = ticker_df_adjusted.index[idx]
                    score = trajectory_scores[idx]
                    label = labels[idx]
                    
                    print(f"\n{'='*50}")
                    print(f"Sample Analysis at {date}")
                    print(f"{'='*50}")
                    
                    print("\nComponent Scores:")
                    print(f"Path Volatility: {debug_info['path_volatility']:.3f}")
                    print(f"Time Above Entry (Direction Window): {debug_info['time_above_entry_direction']:.1%}")
                    print(f"Time Above Entry (Full Window): {debug_info['time_above_entry_full']:.1%}")
                    print(f"Drawdown Penalty: {debug_info['drawdown_penalty']:.3f}")
                    
                    print("\nScore Calculation:")
                    print(f"Weighted Sum: {debug_info['weighted_sum']:.3f}")
                    print(f"Direction: {debug_info['direction']:.3f}")
                    print(f"Final Score: {score:.3f}")
                    print(f"Assigned Label: {label}")


    #############################
    # NOT USING PATH QUALITY SCORE 
    #############################
    else:
        print("Debug: Entering Barrier-Based Analysis section")
        debug_output.add_section("Barrier-Based Analysis")
        print("Debug: Added Barrier-Based section") 

        # Get max class barriers
        pos_boundary_max = boundaries_positive[:, -1]  # Max class barrier
        neg_boundary_max = boundaries_negative[:, -1]  # Max negative class barrier
        
        # Check for max class barrier hits
        pos_hit_max = percent_changes >= pos_boundary_max[:, None]
        neg_hit_max = percent_changes <= neg_boundary_max[:, None]
        
        # Assign max class labels where appropriate
        pos_hit_max_any = np.any(pos_hit_max, axis=1)
        neg_hit_max_any = np.any(neg_hit_max, axis=1)
        
        labels[pos_hit_max_any] = max_class
        labels[neg_hit_max_any] = -max_class

        # Debug section for max class hits
        if labeling_debugging_mode and symbol == debug_symbol:
            with debug_output.capture_output():
                # Initialize date_mask
                if debug_start_date is None and debug_end_date is None:
                    # If no dates provided, include all dates
                    date_mask = pd.Series(True, index=ticker_df_adjusted.index)
                else:
                    # Convert debug dates to match index timezone if they're not None
                    if debug_start_date is not None:
                        debug_start_date = pd.Timestamp(debug_start_date).tz_localize(ticker_df_adjusted.index.tz)
                    if debug_end_date is not None:
                        debug_end_date = pd.Timestamp(debug_end_date).tz_localize(ticker_df_adjusted.index.tz)
                    
                    # Create the date mask based on provided dates
                    date_mask = pd.Series(True, index=ticker_df_adjusted.index)
                    if debug_start_date is not None:
                        date_mask &= (ticker_df_adjusted.index >= debug_start_date)
                    if debug_end_date is not None:
                        date_mask &= (ticker_df_adjusted.index <= debug_end_date)
                                
                # Debug positive max hits
                debug_pos = pos_hit_max_any & date_mask
                if np.any(debug_pos):
                    for idx in np.where(debug_pos)[0]:
                        date = ticker_df_adjusted.index[idx]
                        print(f"\nMax positive class hit at {date}:")
                        print_sample_debug(idx, date, close_prices, atr_percent_changes, 
                                        boundaries_positive, boundaries_negative, percent_changes,
                                        look_ahead_window, num_classes_per_side,
                                        symbol=symbol, debug_symbol=debug_symbol,
                                        upper_coeff_smooth=upper_coeff_smooth,
                                        lower_coeff_smooth=lower_coeff_smooth)
                        print(f"Assigned Label: {max_class}")
                
                # Debug negative max hits
                debug_neg = neg_hit_max_any & date_mask
                if np.any(debug_neg):
                    for idx in np.where(debug_neg)[0]:
                        date = ticker_df_adjusted.index[idx]
                        print(f"\nMax negative class hit at {date}:")
                        print_sample_debug(idx, date, close_prices, atr_percent_changes, 
                                        boundaries_positive, boundaries_negative, percent_changes,
                                        look_ahead_window, num_classes_per_side,
                                        symbol=symbol, debug_symbol=debug_symbol,
                                        upper_coeff_smooth=upper_coeff_smooth,
                                        lower_coeff_smooth=lower_coeff_smooth)
                        print(f"Assigned Label: {-max_class}")
            
        # For remaining samples, use final price change
        remaining_mask = labels == -999
        
        if np.any(remaining_mask):
            # Get the last valid price change for each sample
            last_valid_changes = np.zeros(n)
            for i in range(n):
                valid_changes = percent_changes[i, ~np.isnan(percent_changes[i, :])]
                if len(valid_changes) > 0:
                    last_valid_changes[i] = valid_changes[-1]
                else:
                    last_valid_changes[i] = np.nan
            
            # Iterate through classes (largest to smallest) for remaining samples
            for cls in range(max_class - 1, 0, -1):  # Exclude max_class as it's already handled
                pos_boundary = boundaries_positive[:, cls-1]
                neg_boundary = boundaries_negative[:, cls-1]
                
                # Check which samples exceed current barriers
                pos_samples = (last_valid_changes >= pos_boundary) & remaining_mask
                neg_samples = (last_valid_changes <= neg_boundary) & remaining_mask
                
                # Assign labels and update remaining mask
                labels[pos_samples] = cls
                labels[neg_samples] = -cls
                remaining_mask = labels == -999
                
                if labeling_debugging_mode and symbol == debug_symbol:
                    with debug_output.capture_output():
                        date_mask = create_date_mask(ticker_df_adjusted.index, debug_start_date, debug_end_date)
                        
                        debug_pos = pos_samples & date_mask
                        debug_neg = neg_samples & date_mask
                        
                        if np.any(debug_pos):
                            for idx in np.where(debug_pos)[0]:
                                date = ticker_df_adjusted.index[idx]
                                print(f"\nClass {cls} positive assignment at {date}:")
                                print_sample_debug(idx, date, close_prices, atr_percent_changes, 
                                                boundaries_positive, boundaries_negative, percent_changes,
                                                look_ahead_window, num_classes_per_side,
                                                symbol=symbol, debug_symbol=debug_symbol,
                                                upper_coeff_smooth=upper_coeff_smooth,
                                                lower_coeff_smooth=lower_coeff_smooth)
                                print(f"Final change: {last_valid_changes[idx]:.2f}%")
                                print(f"Assigned Label: {cls}")
                        
                        if np.any(debug_neg):
                            for idx in np.where(debug_neg)[0]:
                                date = ticker_df_adjusted.index[idx]
                                print(f"\nClass {cls} negative assignment at {date}:")
                                print_sample_debug(idx, date, close_prices, atr_percent_changes, 
                                                boundaries_positive, boundaries_negative, percent_changes,
                                                look_ahead_window, num_classes_per_side,
                                                symbol=symbol, debug_symbol=debug_symbol,
                                                upper_coeff_smooth=upper_coeff_smooth,
                                                lower_coeff_smooth=lower_coeff_smooth)
                                print(f"Final change: {last_valid_changes[idx]:.2f}%")
                                print(f"Assigned Label: {-cls}")

        # Assign smallest class to any remaining samples with valid final changes
        remaining_samples = remaining_mask & ~np.isnan(last_valid_changes)
        
        labels[remaining_samples & (last_valid_changes > 0)] = 1
        labels[remaining_samples & (last_valid_changes < 0)] = -1
            
        if labeling_debugging_mode and symbol == debug_symbol:
            with debug_output.capture_output():
                date_mask = create_date_mask(ticker_df_adjusted.index, debug_start_date, debug_end_date)
                debug_remaining = remaining_samples & date_mask
                
                if np.any(debug_remaining):
                    for idx in np.where(debug_remaining)[0]:
                        date = ticker_df_adjusted.index[idx]
                        print(f"\nSmallest class assignment at {date}:")
                        print_sample_debug(idx, date, close_prices, atr_percent_changes, 
                                        boundaries_positive, boundaries_negative, percent_changes,
                                        look_ahead_window, num_classes_per_side,
                                        symbol=symbol, debug_symbol=debug_symbol,
                                        upper_coeff_smooth=upper_coeff_smooth,
                                        lower_coeff_smooth=lower_coeff_smooth)
                        print(f"Final change: {last_valid_changes[idx]:.2f}%")
                        print(f"Assigned Label: {1 if last_valid_changes[idx] > 0 else -1}")

        # Handle remaining invalid samples (those with NaN final changes)
        labels[remaining_mask & np.isnan(last_valid_changes)] = -100


    #############################
    # UPDATE THE DATAFRAME 
    #############################

    # Add this at the end of the labeling section, just before updating the DataFrame
    if labeling_debugging_mode and symbol == debug_symbol:
        with debug_output.capture_output():
            if np.any(labels == 0):
                zero_mask = labels == 0
                zero_indices = np.where(zero_mask)[0]
                print("\nFound zero labels at indices:", zero_indices)
                print("Dates with zero labels:")
                for idx in zero_indices:
                    date = ticker_df_adjusted.index[idx]
                    print(f"Date: {date}")
                    if use_calculate_path_quality_score:
                        print(f"Composite score: {composite_scores[idx]}")
                    print("Future price changes:", percent_changes[idx])

    # Update the DataFrame with the final labels
    ticker_df_adjusted['label'] = labels

    if labeling_debugging_mode and symbol == debug_symbol:
        with debug_output.capture_output():
            date_mask = create_date_mask(ticker_df_adjusted.index, debug_start_date, debug_end_date)
            debug_labels = labels[date_mask]
            print("\nFinal Label Distribution in Debug Range:")
            unique_labels, counts = np.unique(debug_labels, return_counts=True)
            for label, count in zip(unique_labels, counts):
                print(f"  Label {label}: {count}")

            # Print detailed sample information if using composite scores
            if use_calculate_path_quality_score:
                for idx in np.where(date_mask)[0]:
                    date = ticker_df_adjusted.index[idx]
                    print(f"\nAnalyzing {debug_symbol} sample at {date}:")
                    print(f"Close price: {close_prices[idx]:.2f}")
                    print(f"Composite score: {composite_scores[idx]:.3f}")
                    print(f"Assigned Label: {labels[idx]}")
                    print("\nFuture price changes:")
                    for t in range(look_ahead_window):
                        if not np.isnan(percent_changes[idx, t]):
                            print(f"t+{t+1}: {percent_changes[idx, t]:.2f}%")
                        else:
                            print(f"t+{t+1}: NaN")

    #############################
    # LABEL STATS
    #############################
    if labeling_debugging_mode and symbol == debug_symbol:  # Add symbol check
        
        debug_output.add_section("Label Statistics")

        # Extract time from datetime index
        ticker_df_adjusted['time'] = ticker_df_adjusted.index.time

        # Group by time and calculate stats
        time_stats = ticker_df_adjusted.groupby('time').agg({
            'label': lambda x: (x.isin([-100, -999])).sum()
        }).rename(columns={'label': 'invalid_count'})

        # Add total count per time
        time_stats['total_count'] = ticker_df_adjusted.groupby('time').size()

        # Calculate percentages
        time_stats['invalid_percent'] = (time_stats['invalid_count'] / time_stats['total_count'] * 100)

        # Sort by time
        time_stats = time_stats.sort_index()

        # Print results within the debug section
        with debug_output.capture_output():
            print("\nInvalid Label (-100 or -999) Distribution by Time:")
            print("Time | Invalid % | Invalid Count | Total Count")
            print("-" * 50)

            for time, row in time_stats.iterrows():
                print(f"{time} | {row['invalid_percent']:.2f}% | {row['invalid_count']} | {row['total_count']}")

            # Overall statistics
            total_invalid = time_stats['invalid_count'].sum()
            total_samples = time_stats['total_count'].sum()
            overall_percent = (total_invalid / total_samples * 100)

            print("\nOverall Statistics:")
            print(f"Total Invalid Labels: {total_invalid}")
            print(f"Total Samples: {total_samples}")
            print(f"Overall Invalid Percentage: {overall_percent:.2f}%")

    else:  # Still calculate stats, but don't add to debug output
        # Extract time from datetime index
        ticker_df_adjusted['time'] = ticker_df_adjusted.index.time

        # Group by time and calculate stats
        time_stats = ticker_df_adjusted.groupby('time').agg({
            'label': lambda x: (x.isin([-100, -999])).sum()
        }).rename(columns={'label': 'invalid_count'})

        time_stats['total_count'] = ticker_df_adjusted.groupby('time').size()
        time_stats['invalid_percent'] = (time_stats['invalid_count'] / time_stats['total_count'] * 100)

    # Make sure we finalize before returning
    debug_output.finalize()

    return ticker_df_adjusted, debug_output
                        


