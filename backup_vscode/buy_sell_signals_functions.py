
from algo_feature_engineering.features.ma import (
    moving_average_features_normalized,
    moving_average_features_slopes,
    moving_average_features_percent_diff,
    calculate_hma
)


def process_single_symbol(
    symbol_df,
    hma_period=15,  
    feature_suffix="", 
    slope_period=2,
    classes_for_spread=5,
    buy_signal_n_classes=5,
    sell_signal_n_classes=5,
    debug=False,
    use_class_spread=True,
    # NEW LOGIC
    use_adjusted_hma=False,
    hma_adjust_ma_period=50,
    hma_adjust=10,
    # PARAMETER for cutoff
    buy_time_cutoff="13:00",
    apply_930_logic=True,
    sell_at_end_of_day=True,
    morning_hma_adjustment=0,
    is_model_type_short=False
):
    """
    Process a single symbol's data with HMA, slope, and generate buy/sell signals.
    
    If use_class_spread=True, we perform the class-spread-based buy/sell logic.
    Otherwise, we apply a top/bottom crossing strategy.
    
    If use_adjusted_hma=True, the HMA period is adjusted based on whether the
    `close_raw` is above/below a simple moving average (SMA) of period
    hma_adjust_ma_period. The amount of adjustment is set by hma_adjust.

    buy_time_cutoff is a string (e.g., "13:00") indicating the hour/minute cutoff
    for generating a buy signal in the class-spread strategy.
    """
    import numpy as np
    import pandas as pd
    import talib

    # Modify column names to include suffix
    def add_suffix(col_name):
        return f"{col_name}{feature_suffix}" if feature_suffix else col_name

    
    # -------------------------------------------------------------------------
    # Parse the buy_time_cutoff
    # -------------------------------------------------------------------------
    try:
        buy_hour, buy_minute = [int(x) for x in buy_time_cutoff.split(":")]
    except:
        buy_hour, buy_minute = (14, 0)
        print(f"Warning: Could not parse buy_time_cutoff='{buy_time_cutoff}', defaulting to 14:00.")

    # If sell_at_end_of_day is False, extend buy time to end of day
    if not sell_at_end_of_day:
        buy_hour, buy_minute = (23, 59)
        if debug:
            print("sell_at_end_of_day is False, extending buy time cutoff to 23:59")

    symbol_df = symbol_df.copy()
    
    # Debug helper
    debug_df = symbol_df.head(50) if debug else None

    # -------------------------------------------------------------------------
    # 1) Find and validate prediction_raw_class columns
    # -------------------------------------------------------------------------
    prediction_raw_class_columns = [
        col for col in symbol_df.columns if col.startswith('prediction_raw_class_')
    ]
    
    if not prediction_raw_class_columns:
        print("Warning: No prediction_raw_class columns found in the DataFrame")
        symbol_df[add_suffix('predicted_class_moving_avg')] = np.nan
        symbol_df[add_suffix('buy_final')] = 0
        symbol_df[add_suffix('sell')] = 0
        return symbol_df

    if debug:
        print("\n[DEBUG] Found prediction class columns:", prediction_raw_class_columns)

    # -------------------------------------------------------------------------
    # 2) Calculate HMA (with optional adjusted HMA) and Slope
    # -------------------------------------------------------------------------
    successful_hma_columns = []

    # Add morning session check
    def get_morning_adjusted_period(timestamp, base_period):
        """Helper to adjust HMA period during morning session"""
        if (timestamp.hour == 9 and timestamp.minute >= 30) or \
           (timestamp.hour == 10 and timestamp.minute <= 30):
            return max(4, base_period - morning_hma_adjustment)  # Ensure period doesn't go below 1
        return base_period


    if use_adjusted_hma:
        symbol_df['close_raw_ma_for_adjustment'] = talib.SMA(
            symbol_df['close_raw'].to_numpy(dtype=np.float64),
            timeperiod=hma_adjust_ma_period
        )

        symbol_df['close_raw_above_ma'] = (
            symbol_df['close_raw'] > symbol_df['close_raw_ma_for_adjustment']
        ).astype(int)

        for col in prediction_raw_class_columns:
            try:
                # Get morning-adjusted period based on first timestamp for base and adjusted periods
                base_period = get_morning_adjusted_period(
                    symbol_df.index[0],  # Use first timestamp
                    hma_period
                )
                adjusted_period = get_morning_adjusted_period(
                    symbol_df.index[0],  # Use first timestamp
                    hma_period + hma_adjust
                )

                # Calculate original and adjusted HMAs with their respective periods
                original_hma = calculate_hma(
                    symbol_df[col].to_numpy(),
                    period=int(base_period)
                )
                adjusted_hma = calculate_hma(
                    symbol_df[col].to_numpy(),
                    period=int(adjusted_period)
                )

                # Select HMA based on MA condition
                if not is_model_type_short:
                    # For non-short models, use the original mapping
                    symbol_df[add_suffix(f'HMA_{col}')] = np.where(
                        symbol_df['close_raw_above_ma'] == 1,
                        adjusted_hma,
                        original_hma
                    )
                else:
                    # For short models, reverse the mapping of the chosen columns
                    symbol_df[add_suffix(f'HMA_{col}')] = np.where(
                        symbol_df['close_raw_above_ma'] == 1,
                        original_hma,
                        adjusted_hma
                    )

                hma_values = symbol_df[add_suffix(f'HMA_{col}')].astype(float).values
                symbol_df[add_suffix(f'Slope_HMA_{col}')] = talib.LINEARREG_SLOPE(
                    hma_values, timeperiod=slope_period
                )
                successful_hma_columns.append(col)

            except Exception as e:
                print(f"Warning: Error processing column {col}: {str(e)}")
                continue
    else:
        for col in prediction_raw_class_columns:
            try:
                # Get morning-adjusted period based on first timestamp
                adjusted_period = get_morning_adjusted_period(
                    symbol_df.index[0],  # Use first timestamp
                    hma_period
                )

                # Calculate HMA with morning-adjusted period
                symbol_df[add_suffix(f'HMA_{col}')] = calculate_hma(
                    symbol_df[col].to_numpy(),
                    period=int(adjusted_period)
                )

                hma_values = symbol_df[add_suffix(f'HMA_{col}')].astype(float).values
                symbol_df[add_suffix(f'Slope_HMA_{col}')] = talib.LINEARREG_SLOPE(
                    hma_values, timeperiod=slope_period
                )
                successful_hma_columns.append(col)
            except Exception as e:
                print(f"Warning: Error processing column {col}: {str(e)}")
                continue

    if not successful_hma_columns:
        print("Warning: No HMA calculations were successful")
        symbol_df[add_suffix('predicted_class_moving_avg')] = np.nan
        symbol_df[add_suffix('buy_final')] = 0
        symbol_df[add_suffix('sell')] = 0
        return symbol_df

    # -------------------------------------------------------------------------
    # 3) Calculate predicted_class_moving_avg
    # -------------------------------------------------------------------------
    try:
        # Add suffix when creating the list of HMA column names
        hma_prediction_raw_class_cols = [add_suffix(f'HMA_{col}') for col in successful_hma_columns]
        if hma_prediction_raw_class_cols:
            class_numbers = [int(col.split('_')[-1]) for col in successful_hma_columns]
            # Use the suffixed column names when accessing the DataFrame
            hma_df = symbol_df[hma_prediction_raw_class_cols]

            mask_all_nan = hma_df.isna().all(axis=1)
            hma_df_filled = hma_df.fillna(-np.inf).infer_objects()
            idxmax = hma_df_filled.idxmax(axis=1)
            idxmax[mask_all_nan] = np.nan

            # Use suffix for class mapping keys
            class_mapping = {
                add_suffix(f'HMA_prediction_raw_class_{num}'): num for num in class_numbers
            }
            # Use suffix for predicted class column
            symbol_df[add_suffix('predicted_class_moving_avg')] = idxmax.map(class_mapping)
    except Exception as e:
        print(f"Warning: Error calculating predicted_class_moving_avg: {str(e)}")
        symbol_df[add_suffix('predicted_class_moving_avg')] = np.nan

    if debug:
        print("\n[DEBUG] Checking predicted_class_moving_avg calculation (first 50 rows):")
        print(symbol_df[add_suffix('predicted_class_moving_avg')].head())

    # -------------------------------------------------------------------------
    # 4) Ensure datetime index
    # -------------------------------------------------------------------------
    if not pd.api.types.is_datetime64_any_dtype(symbol_df.index):
        symbol_df.index = pd.to_datetime(symbol_df.index)

    # -------------------------------------------------------------------------
    # 5) Strategy Selection
    # -------------------------------------------------------------------------
    if use_class_spread:
        try:
            sorted_class_numbers = sorted(class_numbers)
            if len(sorted_class_numbers) % 2 != 0:
                raise ValueError("The number of classes must be even.")
            if classes_for_spread * 2 > len(sorted_class_numbers):
                raise ValueError(
                    f"classes_for_spread={classes_for_spread} is too large "
                    f"for total_classes={len(sorted_class_numbers)}."
                )

            half = len(sorted_class_numbers) // 2
            smallest_classes = sorted_class_numbers[:half]
            largest_classes = sorted_class_numbers[half:]

            symbol_df[add_suffix('smallest_spread_classes_used')] = ",".join(map(str, smallest_classes[:classes_for_spread]))
            symbol_df[add_suffix('largest_spread_classes_used')] = ",".join(map(str, largest_classes[-classes_for_spread:]))

            smallest_spread_cols = [
                add_suffix(f'HMA_prediction_raw_class_{num}') for num in smallest_classes[:classes_for_spread]
            ]
            largest_spread_cols = [
                add_suffix(f'HMA_prediction_raw_class_{num}') for num in largest_classes[-classes_for_spread:]
            ]

            symbol_df[add_suffix('smallest_spread_mean')] = symbol_df[smallest_spread_cols].mean(axis=1)
            symbol_df[add_suffix('largest_spread_mean')] = symbol_df[largest_spread_cols].mean(axis=1)

            symbol_df[add_suffix('class_spread')] = (
                symbol_df[add_suffix('largest_spread_mean')] - 
                symbol_df[add_suffix('smallest_spread_mean')]
            )

            # BUY LOGIC (including OR time == 9:30 & spread > 0)
            symbol_df[add_suffix('buy_final')] = np.where(
                (
                    (
                        (symbol_df[add_suffix('class_spread')].shift(1) <= 0.00) &
                        (symbol_df[add_suffix('class_spread')] > 0.00)
                    ) &
                    (
                        (symbol_df.index.hour < buy_hour) |
                        ((symbol_df.index.hour == buy_hour) & (symbol_df.index.minute == buy_minute))
                    )
                )
                |
                (
                    (apply_930_logic) &  # <--- We add this condition
                    (symbol_df.index.hour == 9) &
                    (symbol_df.index.minute == 30) &
                    (symbol_df[add_suffix('class_spread')] > 0.00)
                ),
                1,
                0
            )

            # Modified sell logic to include Friday end-of-day rule
            regular_end_of_day_condition = (
                (symbol_df.index.hour == 15) & 
                (symbol_df.index.minute == 45)
            ) if sell_at_end_of_day else False

            # New Friday end-of-day condition (will be enforced regardless of sell_at_end_of_day flag)
            friday_end_of_day_condition = (
                (symbol_df.index.hour == 15) & 
                (symbol_df.index.minute == 45) &
                (symbol_df.index.weekday == 4)  # 4 represents Friday in pandas
            )

            symbol_df[add_suffix('sell')] = np.where(
                (
                    (symbol_df[add_suffix('class_spread')].shift(1) >= 0.00) &
                    (symbol_df[add_suffix('class_spread')] < 0.00)
                ) | regular_end_of_day_condition | friday_end_of_day_condition,
                1,
                0
            )

        except Exception as e:
            print(f"Warning: Error in class spread strategy: {str(e)}")
            symbol_df[add_suffix('buy_final')] = 0
            symbol_df[add_suffix('sell')] = 0

    else:
        # Top/Bottom Crossing Strategy
        try:
            sorted_class_numbers = sorted(class_numbers)

            lowest_classes = sorted_class_numbers[:sell_signal_n_classes]
            highest_classes = sorted_class_numbers[-buy_signal_n_classes:]

            # BUY LOGIC (including OR time == 9:30 & in highest_classes)
            symbol_df[add_suffix('buy_final')] = np.where(
                (
                    (symbol_df[add_suffix('predicted_class_moving_avg')].shift(1).isin(lowest_classes)) &
                    (symbol_df[add_suffix('predicted_class_moving_avg')].isin(highest_classes))
                )
                |
                (
                    (apply_930_logic) &  # <--- We add this condition
                    (symbol_df.index.hour == 9) &
                    (symbol_df.index.minute == 30) &
                    (symbol_df[add_suffix('predicted_class_moving_avg')].isin(highest_classes))
                ),
                1, 
                0
            )

            # Modified sell logic to include both regular and Friday end-of-day rules
            regular_end_of_day_condition = (
                (symbol_df.index.hour == 15) & 
                (symbol_df.index.minute == 45)
            ) if sell_at_end_of_day else False

            # New Friday end-of-day condition (enforced regardless of sell_at_end_of_day flag)
            friday_end_of_day_condition = (
                (symbol_df.index.hour == 15) & 
                (symbol_df.index.minute == 45) &
                (symbol_df.index.weekday == 4)  # 4 represents Friday in pandas
            )

            symbol_df[add_suffix('sell')] = np.where(
                (
                    (symbol_df[add_suffix('predicted_class_moving_avg')].shift(1).isin(highest_classes)) &
                    (symbol_df[add_suffix('predicted_class_moving_avg')].isin(lowest_classes))
                ) | regular_end_of_day_condition | friday_end_of_day_condition,
                1,
                0
            )

        except Exception as e:
            print(f"Warning: Error in top/bottom crossing strategy: {str(e)}")
            symbol_df[add_suffix('buy_final')] = 0
            symbol_df[add_suffix('sell')] = 0

    # -------------------------------------------------------------------------
    # 6) Debug print for last 50 rows buy/sell logic
    # -------------------------------------------------------------------------
    if debug:
        debug_df_tail = symbol_df.tail(50)
        
        if use_class_spread:
            # Debug info for Class Spread Strategy
            print("\n[DEBUG] Last 50 rows - buy_final & sell signals (Class Spread):")
            columns_to_print = [
                'smallest_spread_classes_used',
                'largest_spread_classes_used',
                add_suffix('smallest_spread_mean'),
                add_suffix('largest_spread_mean'),
                add_suffix('class_spread'),
                add_suffix('buy_final'),
                add_suffix('sell')
            ]
        else:
            # Debug info for Top/Bottom Crossing Strategy
            print("\n[DEBUG] Last 50 rows - buy_final & sell signals (Top/Bottom Crossing):")
            columns_to_print = [
                add_suffix('predicted_class_moving_avg'),
                add_suffix('buy_final'),
                add_suffix('sell')
            ]
        
        columns_to_print = [c for c in columns_to_print if c in debug_df_tail.columns]
        print(debug_df_tail[columns_to_print])

    return symbol_df


def process_all_symbols(
    df, 
    hma_period,
    feature_suffix="",
    classes_for_spread=5, 
    buy_signal_n_classes=5, 
    sell_signal_n_classes=5, 
    debug=False,
    use_class_spread=True,
    # NEW LOGIC
    use_adjusted_hma=False,
    hma_adjust_ma_period=50,
    hma_adjust=10,
    # PARAMETER for cutoff
    buy_time_cutoff="13:00",
    apply_930_logic=True,
    sell_at_end_of_day=True,
    morning_hma_adjustment=0,
    is_model_type_short=False
):
    """
    Process multiple symbols from a single DataFrame by applying process_single_symbol
    to each unique symbol.

    Parameters:
    - df: pd.DataFrame
    - classes_for_spread: int, number of classes used for the class_spread calculation.
    - buy_signal_n_classes: int, top classes to consider for buy signal.
    - sell_signal_n_classes: int, bottom classes to consider for sell signal.
    - debug: bool, if True prints debug info from process_single_symbol.
    - use_class_spread: bool, if True uses class_spread-based strategy; if False uses top/bottom crossing strategy.
    - use_adjusted_hma: bool, if True applies an adjusted HMA period based on price vs. a moving average
    - hma_adjust_ma_period: int, period used for the SMA to decide if HMA is adjusted
    - hma_adjust: int, how much to add to hma_period when above the MA
    - buy_time_cutoff: str, e.g. "13:00" to set the hour/minute limit for buy signals in class-spread strategy
    - sell_at_end_of_day: bool, if True enforces selling at 15:45, if False only sells based on strategy signals

    Returns:
    - pd.DataFrame, concatenated result for all symbols.
    """
    import pandas as pd
    symbols = df['symbol'].unique()
    processed_dfs = []

    for symbol in symbols:
        symbol_df = df[df['symbol'] == symbol].copy()
        
        processed_symbol_df = process_single_symbol(
            symbol_df,
            hma_period=hma_period,  
            feature_suffix=feature_suffix,
            slope_period=2,
            classes_for_spread=classes_for_spread,
            buy_signal_n_classes=buy_signal_n_classes,
            sell_signal_n_classes=sell_signal_n_classes,
            debug=debug,
            use_class_spread=use_class_spread,
            use_adjusted_hma=use_adjusted_hma,
            hma_adjust_ma_period=hma_adjust_ma_period,
            hma_adjust=hma_adjust,
            buy_time_cutoff=buy_time_cutoff,
            apply_930_logic=apply_930_logic,
            sell_at_end_of_day=sell_at_end_of_day,
            morning_hma_adjustment=morning_hma_adjustment,
            is_model_type_short=is_model_type_short
        )
        processed_dfs.append(processed_symbol_df)

    result_df = pd.concat(processed_dfs)
    return result_df





