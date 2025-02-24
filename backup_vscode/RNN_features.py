import numpy as np


######################
# FUNCTION TO LOAD SAVED MODEL AND ADD PREDICTIONS TO DATAFRAME
######################

def predict_sequences(model, df, columns_to_include, seq_length, batch_size, new_column_name, scaler=None, should_scale=False):
    import torch
    import numpy as np
    import pandas as pd 
    
    # Pre-filter data and convert to contiguous array for faster access
    data_values = df[columns_to_include].values.astype(np.float32, copy=False)
    
    if should_scale and scaler is not None:
        data_values = scaler.transform(df[columns_to_include]).astype(np.float32)
        
    data_values = np.ascontiguousarray(data_values)
    
    num_sequences = len(data_values) - seq_length + 1
    if num_sequences <= 0:
        raise ValueError("Sequence length is too long for the data")
    
    # Optimize batch size based on data size
    batch_size = min(batch_size, 2048)  # Prevent excessive memory usage
    
    # Pre-allocate predictions array
    predictions = np.empty(num_sequences, dtype=np.float32)
    seq_indices = np.arange(seq_length)
    
    # Create fixed CUDA tensors
    with torch.no_grad():
        for i in range(0, num_sequences, batch_size):
            end_idx = min(i + batch_size, num_sequences)
            batch_size_actual = end_idx - i
            
            # Optimize indexing operations
            batch_sequences = np.lib.stride_tricks.as_strided(
                data_values[i:end_idx + seq_length - 1],
                shape=(batch_size_actual, seq_length, data_values.shape[1]),
                strides=(data_values.strides[0], data_values.strides[0], data_values.strides[1])
            ).astype(np.float32)
            
            # Optimize tensor transfer
            batch_tensor = torch.from_numpy(batch_sequences).cuda(non_blocking=True)
            batch_predictions = model(batch_tensor)
            predictions[i:end_idx] = batch_predictions.cpu().numpy().flatten()
            
            del batch_tensor, batch_predictions
            
            # Less frequent cleanup
            if i % (batch_size * 20) == 0:
                torch.cuda.empty_cache()

    if should_scale and scaler is not None:
        dummy_array = np.zeros((len(predictions), len(columns_to_include)), dtype=np.float32)
        dummy_df = pd.DataFrame(dummy_array, columns=columns_to_include)
        target_column_idx = columns_to_include.index(columns_to_include[-1])
        dummy_df.iloc[:, target_column_idx] = predictions
        predictions = scaler.inverse_transform(dummy_df)[:, target_column_idx].astype(np.float32)
    
    # Optimize DataFrame assignment
    result = pd.Series([None] * len(df), index=df.index, dtype=np.float32)
    result.iloc[seq_length-1:seq_length-1+len(predictions)] = predictions
    df[new_column_name] = result
    
    return df



######################
# composite
######################

import numpy as np
import pandas as pd

def create_logical_group_composites(df: pd.DataFrame,
                                    threshold: float = 1.5) -> pd.DataFrame:
    """
    Create three logical groupings of composite forecast signals, each with:
      1) A composite (weighted average) forecast
      2) Acceleration (first difference)
      3) An extreme flag, based on a threshold

    Args:
        df (pd.DataFrame):
            The input DataFrame containing your individual forecast columns.
        threshold (float, optional):
            Threshold to determine an “extreme” regime for each composite.
            Default = 1.5

    Returns:
        pd.DataFrame:
            The input DataFrame with new columns added for each composite group:
              <group_name>_composite_forecast
              <group_name>_composite_accel
              <group_name>_composite_extreme
    """

    # --------------------------------------------------------------------
    # 1. Define your logical groupings here.
    #    Adjust these lists to match the relevant forecast columns in your DataFrame.
    # --------------------------------------------------------------------
    forecast_groups = {
        # Example: VWAP z-score forecasts
        'vwap': [
            'vwap_zscore_10_15min_t6_prediction',
            'vwap_zscore_10_15min_t7_prediction',
            'vwap_zscore_10_15min_t8_prediction',
            'vwap_zscore_10_15min_t9_prediction',
        ],
        # Example: RSI forecasts
        'rsi': [
            'RSI_7_hma_15_15min_t1_prediction',
            'RSI_7_hma_15_15min_t2_prediction',
            'RSI_7_hma_15_15min_t3_prediction',
            'RSI_7_hma_15_15min_t4_prediction',
        ],
        # Example: CCI forecasts
        'cci': [
            'close_raw_CCI_hma_14_t1_prediction',
            'close_raw_CCI_hma_14_t2_prediction',
            'close_raw_CCI_hma_14_t3_prediction',
        ],
    }

    # --------------------------------------------------------------------
    # 2. Create new columns for each forecast group
    # --------------------------------------------------------------------
    for group_name, columns in forecast_groups.items():
        # Filter only those columns that actually exist in df
        valid_cols = [col for col in columns if col in df.columns]
        if not valid_cols:
            # If the DataFrame doesn’t have any of the columns, skip
            continue

        # Equal weights by default; you can customize or pass them in
        weight = 1.0 / len(valid_cols)

        # Composite forecast
        composite_col = f"{group_name}_composite_forecast"
        df[composite_col] = 0.0
        for col in valid_cols:
            df[composite_col] += df[col] * weight

        # Acceleration = first difference of the composite
        accel_col = f"{group_name}_composite_accel"
        df[accel_col] = df[composite_col].diff()

        # Extreme Regime Indicator
        extreme_col = f"{group_name}_composite_extreme"
        df[extreme_col] = (
            (df[composite_col] > threshold) |
            (df[composite_col] < -threshold)
        ).astype(int)

    return df

######################
# RETURNS
######################

def predict_trailing_return_20_zscore_ema_15min_t1(dataframe, model_trailing_return_20_zscore_ema_15min_t1, scaler_trailing_return_20_zscore_ema_15min_t1):
    """
    Predicts trailing return values using the t1 model and adds prediction, comparison,
    acceleration, and extreme regime indicator columns.
    """
    columns_to_include = [
        'trailing_return_5_zscore',
        'trailing_return_5_zscore_ema',
        'trailing_return_10_zscore',
        'trailing_return_10_zscore_ema',
        'trailing_return_20_zscore',
        'trailing_return_20_zscore_ema',
        'trailing_return_30_zscore',
        'trailing_return_30_zscore_ema'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'trailing_return_20_zscore_ema'
    
    dataframe = predict_sequences(
        model_trailing_return_20_zscore_ema_15min_t1,
        dataframe,
        columns_to_include,
        seq_length,
        1000,
        'trailing_return_20_zscore_ema_15min_t1_prediction',
        should_scale=should_scale
    )
    
    dataframe['trailing_return_20_zscore_ema_15min_t1_diff'] = (
        dataframe['trailing_return_20_zscore_ema_15min_t1_prediction'] - dataframe[original_feature]
    )
    dataframe['trailing_return_20_zscore_ema_15min_t1_diff_positive'] = (
        dataframe['trailing_return_20_zscore_ema_15min_t1_diff'] > 0
    ).astype(int)
    
    dataframe['trailing_return_20_zscore_ema_15min_t1_accel'] = dataframe['trailing_return_20_zscore_ema_15min_t1_prediction'].diff()
    
    return dataframe

def predict_trailing_return_20_zscore_ema_15min_t2(dataframe, model_trailing_return_20_zscore_ema_15min_t2, scaler_trailing_return_20_zscore_ema_15min_t2):
    """
    Predicts trailing return values using the t2 model and adds prediction, comparison,
    acceleration, and extreme regime indicator columns.
    """
    columns_to_include = [
        'trailing_return_5_zscore',
        'trailing_return_5_zscore_ema',
        'trailing_return_10_zscore',
        'trailing_return_10_zscore_ema',
        'trailing_return_20_zscore',
        'trailing_return_20_zscore_ema',
        'trailing_return_30_zscore',
        'trailing_return_30_zscore_ema'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'trailing_return_20_zscore_ema'
    
    dataframe = predict_sequences(
        model_trailing_return_20_zscore_ema_15min_t2,
        dataframe,
        columns_to_include,
        seq_length,
        1000,
        'trailing_return_20_zscore_ema_15min_t2_prediction',
        should_scale=should_scale
    )
    
    dataframe['trailing_return_20_zscore_ema_15min_t2_diff'] = (
        dataframe['trailing_return_20_zscore_ema_15min_t2_prediction'] - dataframe[original_feature]
    )
    dataframe['trailing_return_20_zscore_ema_15min_t2_diff_positive'] = (
        dataframe['trailing_return_20_zscore_ema_15min_t2_diff'] > 0
    ).astype(int)
    
    dataframe['trailing_return_20_zscore_ema_15min_t2_accel'] = dataframe['trailing_return_20_zscore_ema_15min_t2_prediction'].diff()
    
    return dataframe

def predict_trailing_return_20_zscore_ema_15min_t3(dataframe, model_trailing_return_20_zscore_ema_15min_t3, scaler_trailing_return_20_zscore_ema_15min_t3):
    """
    Predicts trailing return values using the t3 model and adds prediction, comparison,
    acceleration, and extreme regime indicator columns.
    """
    columns_to_include = [
        'trailing_return_5_zscore',
        'trailing_return_5_zscore_ema',
        'trailing_return_10_zscore',
        'trailing_return_10_zscore_ema',
        'trailing_return_20_zscore',
        'trailing_return_20_zscore_ema',
        'trailing_return_30_zscore',
        'trailing_return_30_zscore_ema'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'trailing_return_20_zscore_ema'
    
    dataframe = predict_sequences(
        model_trailing_return_20_zscore_ema_15min_t3,
        dataframe,
        columns_to_include,
        seq_length,
        1000,
        'trailing_return_20_zscore_ema_15min_t3_prediction',
        should_scale=should_scale
    )
    
    dataframe['trailing_return_20_zscore_ema_15min_t3_diff'] = (
        dataframe['trailing_return_20_zscore_ema_15min_t3_prediction'] - dataframe[original_feature]
    )
    dataframe['trailing_return_20_zscore_ema_15min_t3_diff_positive'] = (
        dataframe['trailing_return_20_zscore_ema_15min_t3_diff'] > 0
    ).astype(int)
    
    dataframe['trailing_return_20_zscore_ema_15min_t3_accel'] = dataframe['trailing_return_20_zscore_ema_15min_t3_prediction'].diff()
    
    return dataframe

def predict_trailing_return_20_zscore_ema_15min_t4(dataframe, model_trailing_return_20_zscore_ema_15min_t4, scaler_trailing_return_20_zscore_ema_15min_t4):
    """
    Predicts trailing return values using the t4 model and adds prediction, comparison,
    acceleration, and extreme regime indicator columns.
    """
    columns_to_include = [
        'trailing_return_5_zscore',
        'trailing_return_5_zscore_ema',
        'trailing_return_10_zscore',
        'trailing_return_10_zscore_ema',
        'trailing_return_20_zscore',
        'trailing_return_20_zscore_ema',
        'trailing_return_30_zscore',
        'trailing_return_30_zscore_ema'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'trailing_return_20_zscore_ema'
    
    dataframe = predict_sequences(
        model_trailing_return_20_zscore_ema_15min_t4,
        dataframe,
        columns_to_include,
        seq_length,
        1000,
        'trailing_return_20_zscore_ema_15min_t4_prediction',
        should_scale=should_scale
    )
    
    dataframe['trailing_return_20_zscore_ema_15min_t4_diff'] = (
        dataframe['trailing_return_20_zscore_ema_15min_t4_prediction'] - dataframe[original_feature]
    )
    dataframe['trailing_return_20_zscore_ema_15min_t4_diff_positive'] = (
        dataframe['trailing_return_20_zscore_ema_15min_t4_diff'] > 0
    ).astype(int)
    
    dataframe['trailing_return_20_zscore_ema_15min_t4_accel'] = dataframe['trailing_return_20_zscore_ema_15min_t4_prediction'].diff()
    
    return dataframe

def predict_trailing_return_20_zscore_ema_15min_t5(dataframe, model_trailing_return_20_zscore_ema_15min_t5, scaler_trailing_return_20_zscore_ema_15min_t5):
    """
    Predicts trailing return values using the t5 model and adds prediction, comparison,
    acceleration, and extreme regime indicator columns. 
    Calculates the EMA (period = 5) of the prediction and uses it for subsequent calculations.
    """
    columns_to_include = [
        'trailing_return_5_zscore',
        'trailing_return_5_zscore_ema',
        'trailing_return_10_zscore',
        'trailing_return_10_zscore_ema',
        'trailing_return_20_zscore',
        'trailing_return_20_zscore_ema',
        'trailing_return_30_zscore',
        'trailing_return_30_zscore_ema'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'trailing_return_20_zscore_ema'
    prediction_column = 'trailing_return_20_zscore_ema_15min_t5_prediction'
    ema_column = 'trailing_return_20_zscore_ema_15min_t5_prediction_ema5'
    
    # Predict sequences and add the new prediction column
    dataframe = predict_sequences(
        model_trailing_return_20_zscore_ema_15min_t5,
        dataframe,
        columns_to_include,
        seq_length,
        1000,
        prediction_column,
        should_scale=should_scale
    )

    # Compute EMA (period = 5) for the prediction column
    dataframe[ema_column] = dataframe[prediction_column].ewm(span=5, adjust=False).mean()

    # Use the EMA column for subsequent calculations
    dataframe['trailing_return_20_zscore_ema_15min_t5_diff'] = dataframe[ema_column] - dataframe[original_feature]
    dataframe['trailing_return_20_zscore_ema_15min_t5_diff_positive'] = (dataframe['trailing_return_20_zscore_ema_15min_t5_diff'] > 0).astype(int)
    dataframe['trailing_return_20_zscore_ema_15min_t5_accel'] = dataframe[ema_column].diff()

    return dataframe



######################
# PERCENTILES
######################

def predict_close_raw_percentile_20_15min_t1(dataframe, model_close_raw_percentile_20_15min_t1, scaler_close_raw_percentile_20_15min_t1):
    """
    Predicts close raw percentile values using the t1 model and adds prediction, comparison,
    acceleration, and extreme regime indicator columns.
    """
    columns_to_include = [
        'close_raw_percentile_200',
        'close_raw_percentile_100',
        'close_raw_percentile_50',
        'close_raw_percentile_20',
        'close_raw_percentile_10',
        'close_raw_percentile_5'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'close_raw_percentile_20'
    
    dataframe = predict_sequences(
        model_close_raw_percentile_20_15min_t1,
        dataframe,
        columns_to_include,
        seq_length,
        1000,
        'close_raw_percentile_20_15min_t1_prediction',
        should_scale=should_scale
    )
    
    dataframe['close_raw_percentile_20_15min_t1_diff'] = (
        dataframe['close_raw_percentile_20_15min_t1_prediction'] - dataframe[original_feature]
    )
    dataframe['close_raw_percentile_20_15min_t1_diff_positive'] = (
        dataframe['close_raw_percentile_20_15min_t1_diff'] > 0
    ).astype(int)
    
    dataframe['close_raw_percentile_20_15min_t1_accel'] = dataframe['close_raw_percentile_20_15min_t1_prediction'].diff()
    
    return dataframe

def predict_close_raw_percentile_20_15min_t2(dataframe, model_close_raw_percentile_20_15min_t2, scaler_close_raw_percentile_20_15min_t2):
    """
    Predicts close raw percentile values using the t2 model and adds prediction, comparison,
    acceleration, and extreme regime indicator columns.
    """
    columns_to_include = [
        'close_raw_percentile_200',
        'close_raw_percentile_100',
        'close_raw_percentile_50',
        'close_raw_percentile_20',
        'close_raw_percentile_10',
        'close_raw_percentile_5'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'close_raw_percentile_20'
    
    dataframe = predict_sequences(
        model_close_raw_percentile_20_15min_t2,
        dataframe,
        columns_to_include,
        seq_length,
        1000,
        'close_raw_percentile_20_15min_t2_prediction',
        should_scale=should_scale
    )
    
    dataframe['close_raw_percentile_20_15min_t2_diff'] = (
        dataframe['close_raw_percentile_20_15min_t2_prediction'] - dataframe[original_feature]
    )
    dataframe['close_raw_percentile_20_15min_t2_diff_positive'] = (
        dataframe['close_raw_percentile_20_15min_t2_diff'] > 0
    ).astype(int)
    
    dataframe['close_raw_percentile_20_15min_t2_accel'] = dataframe['close_raw_percentile_20_15min_t2_prediction'].diff()
    
    return dataframe

def predict_close_raw_percentile_50_15min_t1(dataframe, model_close_raw_percentile_50_15min_t1, scaler_close_raw_percentile_50_15min_t1):
    """
    Predicts close raw percentile values using the t1 model (50 percentile version) and adds prediction, 
    comparison, and acceleration indicator columns.
    """
    columns_to_include = [
        'close_raw_percentile_200',
        'close_raw_percentile_100',
        'close_raw_percentile_50',
        'close_raw_percentile_20',
        'close_raw_percentile_10',
        'close_raw_percentile_5'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'close_raw_percentile_50'
    
    dataframe = predict_sequences(
        model_close_raw_percentile_50_15min_t1,
        dataframe,
        columns_to_include,
        seq_length,
        1000,
        'close_raw_percentile_50_15min_t1_prediction',
        should_scale=should_scale
    )
    
    dataframe['close_raw_percentile_50_15min_t1_diff'] = (
        dataframe['close_raw_percentile_50_15min_t1_prediction'] - dataframe[original_feature]
    )
    dataframe['close_raw_percentile_50_15min_t1_diff_positive'] = (
        dataframe['close_raw_percentile_50_15min_t1_diff'] > 0
    ).astype(int)
    
    dataframe['close_raw_percentile_50_15min_t1_accel'] = dataframe['close_raw_percentile_50_15min_t1_prediction'].diff()
    
    return dataframe

def predict_close_raw_percentile_50_15min_t2(dataframe, model_close_raw_percentile_50_15min_t2, scaler_close_raw_percentile_50_15min_t2):
    """
    Predicts close raw percentile values using the t2 model (50 percentile version) and adds prediction, 
    comparison, and acceleration indicator columns.
    """
    columns_to_include = [
        'close_raw_percentile_200',
        'close_raw_percentile_100',
        'close_raw_percentile_50',
        'close_raw_percentile_20',
        'close_raw_percentile_10',
        'close_raw_percentile_5'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'close_raw_percentile_50'
    
    dataframe = predict_sequences(
        model_close_raw_percentile_50_15min_t2,
        dataframe,
        columns_to_include,
        seq_length,
        1000,
        'close_raw_percentile_50_15min_t2_prediction',
        should_scale=should_scale
    )
    
    dataframe['close_raw_percentile_50_15min_t2_diff'] = (
        dataframe['close_raw_percentile_50_15min_t2_prediction'] - dataframe[original_feature]
    )
    dataframe['close_raw_percentile_50_15min_t2_diff_positive'] = (
        dataframe['close_raw_percentile_50_15min_t2_diff'] > 0
    ).astype(int)
    
    dataframe['close_raw_percentile_50_15min_t2_accel'] = dataframe['close_raw_percentile_50_15min_t2_prediction'].diff()
    
    return dataframe

def predict_close_raw_percentile_50_15min_t3(dataframe, model_close_raw_percentile_50_15min_t3, scaler_close_raw_percentile_50_15min_t3):
    """
    Predicts close raw percentile values using the t3 model (50 percentile version) and adds prediction, 
    comparison, and acceleration indicator columns.
    """
    columns_to_include = [
        'close_raw_percentile_200',
        'close_raw_percentile_100',
        'close_raw_percentile_50',
        'close_raw_percentile_20',
        'close_raw_percentile_10',
        'close_raw_percentile_5'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'close_raw_percentile_50'
    
    dataframe = predict_sequences(
        model_close_raw_percentile_50_15min_t3,
        dataframe,
        columns_to_include,
        seq_length,
        1000,
        'close_raw_percentile_50_15min_t3_prediction',
        should_scale=should_scale
    )
    
    dataframe['close_raw_percentile_50_15min_t3_diff'] = (
        dataframe['close_raw_percentile_50_15min_t3_prediction'] - dataframe[original_feature]
    )
    dataframe['close_raw_percentile_50_15min_t3_diff_positive'] = (
        dataframe['close_raw_percentile_50_15min_t3_diff'] > 0
    ).astype(int)
    
    dataframe['close_raw_percentile_50_15min_t3_accel'] = dataframe['close_raw_percentile_50_15min_t3_prediction'].diff()
    
    return dataframe





######################
# VWAP
######################

# def predict_vwap_zscore_10_15min_t1(dataframe, model_vwap_zscore_10_15min_t1, scaler_vwap_zscore_10_15min_t1):
#     """
#     Predicts VWAP ZScore values using the t1 model and adds prediction and comparison columns.
#     """
#     columns_to_include = [
#         'hma_vwap_zscore_5', 
#         'hma_vwap_zscore_5_slope',
#         'hma_vwap_zscore_10', 
#         'hma_vwap_zscore_10_slope',
#         'hma_vwap_zscore_15', 
#         'hma_vwap_zscore_15_slope',
#         'hma_vwap_zscore_20', 
#         'hma_vwap_zscore_20_slope'
#     ]
    
#     seq_length = 7
#     should_scale = False
#     original_feature = 'hma_vwap_zscore_10'
    
#     dataframe = predict_sequences(
#         model_vwap_zscore_10_15min_t1,
#         dataframe,
#         columns_to_include,
#         seq_length,
#         1000,
#         'vwap_zscore_10_15min_t1_prediction',
#         should_scale=should_scale
#     )
    
#     dataframe['vwap_zscore_10_15min_t1_diff'] = (
#         dataframe['vwap_zscore_10_15min_t1_prediction'] - dataframe[original_feature]
#     )
#     dataframe['vwap_zscore_10_15min_t1_diff_positive'] = (
#         dataframe['vwap_zscore_10_15min_t1_diff'] > 0
#     ).astype(int)
    
#     return dataframe


# def predict_vwap_zscore_10_15min_t2(dataframe, model_vwap_zscore_10_15min_t2, scaler_vwap_zscore_10_15min_t2):
#     """
#     Predicts VWAP ZScore values using the t2 model and adds prediction and comparison columns.
#     """
#     columns_to_include = [
#         'hma_vwap_zscore_5', 
#         'hma_vwap_zscore_5_slope',
#         'hma_vwap_zscore_10', 
#         'hma_vwap_zscore_10_slope',
#         'hma_vwap_zscore_15', 
#         'hma_vwap_zscore_15_slope',
#         'hma_vwap_zscore_20', 
#         'hma_vwap_zscore_20_slope'
#     ]
    
#     seq_length = 7
#     should_scale = False
#     original_feature = 'hma_vwap_zscore_10'
    
#     dataframe = predict_sequences(
#         model_vwap_zscore_10_15min_t2,
#         dataframe,
#         columns_to_include,
#         seq_length,
#         1000,
#         'vwap_zscore_10_15min_t2_prediction',
#         should_scale=should_scale
#     )
    
#     dataframe['vwap_zscore_10_15min_t2_diff'] = (
#         dataframe['vwap_zscore_10_15min_t2_prediction'] - dataframe[original_feature]
#     )
#     dataframe['vwap_zscore_10_15min_t2_diff_positive'] = (
#         dataframe['vwap_zscore_10_15min_t2_diff'] > 0
#     ).astype(int)
    
#     return dataframe


# def predict_vwap_zscore_10_15min_t3(dataframe, model_vwap_zscore_10_15min_t3, scaler_vwap_zscore_10_15min_t3):
#     """
#     Predicts VWAP ZScore values using the t3 model and adds prediction and comparison columns.
#     """
#     columns_to_include = [
#         'hma_vwap_zscore_5', 
#         'hma_vwap_zscore_5_slope',
#         'hma_vwap_zscore_10', 
#         'hma_vwap_zscore_10_slope',
#         'hma_vwap_zscore_15', 
#         'hma_vwap_zscore_15_slope',
#         'hma_vwap_zscore_20', 
#         'hma_vwap_zscore_20_slope'
#     ]
    
#     seq_length = 7
#     should_scale = False
#     original_feature = 'hma_vwap_zscore_10'
    
#     dataframe = predict_sequences(
#         model_vwap_zscore_10_15min_t3,
#         dataframe,
#         columns_to_include,
#         seq_length,
#         1000,
#         'vwap_zscore_10_15min_t3_prediction',
#         should_scale=should_scale
#     )
    
#     dataframe['vwap_zscore_10_15min_t3_diff'] = (
#         dataframe['vwap_zscore_10_15min_t3_prediction'] - dataframe[original_feature]
#     )
#     dataframe['vwap_zscore_10_15min_t3_diff_positive'] = (
#         dataframe['vwap_zscore_10_15min_t3_diff'] > 0
#     ).astype(int)
    
#     return dataframe


# def predict_vwap_zscore_10_15min_t4(dataframe, model_vwap_zscore_10_15min_t4, scaler_vwap_zscore_10_15min_t4):
#     """
#     Predicts VWAP ZScore values using the t4 model and adds prediction and comparison columns.
#     """
#     columns_to_include = [
#         'hma_vwap_zscore_5', 
#         'hma_vwap_zscore_5_slope',
#         'hma_vwap_zscore_10', 
#         'hma_vwap_zscore_10_slope',
#         'hma_vwap_zscore_15', 
#         'hma_vwap_zscore_15_slope',
#         'hma_vwap_zscore_20', 
#         'hma_vwap_zscore_20_slope'
#     ]
    
#     seq_length = 7
#     should_scale = False
#     original_feature = 'hma_vwap_zscore_10'
    
#     dataframe = predict_sequences(
#         model_vwap_zscore_10_15min_t4,
#         dataframe,
#         columns_to_include,
#         seq_length,
#         1000,
#         'vwap_zscore_10_15min_t4_prediction',
#         should_scale=should_scale
#     )
    
#     dataframe['vwap_zscore_10_15min_t4_diff'] = (
#         dataframe['vwap_zscore_10_15min_t4_prediction'] - dataframe[original_feature]
#     )
#     dataframe['vwap_zscore_10_15min_t4_diff_positive'] = (
#         dataframe['vwap_zscore_10_15min_t4_diff'] > 0
#     ).astype(int)
    
#     return dataframe


# def predict_vwap_zscore_10_15min_t5(dataframe, model_vwap_zscore_10_15min_t5, scaler_vwap_zscore_10_15min_t5):
#     """
#     Predicts VWAP ZScore values using the t5 model and adds prediction and comparison columns.
#     """
#     columns_to_include = [
#         'hma_vwap_zscore_5', 
#         'hma_vwap_zscore_5_slope',
#         'hma_vwap_zscore_10', 
#         'hma_vwap_zscore_10_slope',
#         'hma_vwap_zscore_15', 
#         'hma_vwap_zscore_15_slope',
#         'hma_vwap_zscore_20', 
#         'hma_vwap_zscore_20_slope'
#     ]
    
#     seq_length = 7
#     should_scale = False
#     original_feature = 'hma_vwap_zscore_10'
    
#     dataframe = predict_sequences(
#         model_vwap_zscore_10_15min_t5,
#         dataframe,
#         columns_to_include,
#         seq_length,
#         1000,
#         'vwap_zscore_10_15min_t5_prediction',
#         should_scale=should_scale
#     )
    
#     dataframe['vwap_zscore_10_15min_t5_diff'] = (
#         dataframe['vwap_zscore_10_15min_t5_prediction'] - dataframe[original_feature]
#     )
#     dataframe['vwap_zscore_10_15min_t5_diff_positive'] = (
#         dataframe['vwap_zscore_10_15min_t5_diff'] > 0
#     ).astype(int)
    
#     return dataframe


def predict_vwap_zscore_10_15min_t6(dataframe, model_vwap_zscore_10_15min_t6, scaler_vwap_zscore_10_15min_t6):
    """
    Predicts VWAP ZScore values using the t6 model and adds prediction, comparison,
    acceleration, and extreme regime indicator columns.
    """
    columns_to_include = [
        'hma_vwap_zscore_5', 
        'hma_vwap_zscore_5_slope',
        'hma_vwap_zscore_10', 
        'hma_vwap_zscore_10_slope',
        'hma_vwap_zscore_15', 
        'hma_vwap_zscore_15_slope',
        'hma_vwap_zscore_20', 
        'hma_vwap_zscore_20_slope'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'hma_vwap_zscore_10'
    
    dataframe = predict_sequences(
        model_vwap_zscore_10_15min_t6,
        dataframe,
        columns_to_include,
        seq_length,
        1000,
        'vwap_zscore_10_15min_t6_prediction',
        should_scale=should_scale
    )
    
    dataframe['vwap_zscore_10_15min_t6_diff'] = (
        dataframe['vwap_zscore_10_15min_t6_prediction'] - dataframe[original_feature]
    )
    dataframe['vwap_zscore_10_15min_t6_diff_positive'] = (
        dataframe['vwap_zscore_10_15min_t6_diff'] > 0
    ).astype(int)
    
    # New Feature 1: VWAP ZScore Acceleration (first difference of the prediction)
    dataframe['vwap_zscore_10_15min_t6_accel'] = dataframe['vwap_zscore_10_15min_t6_prediction'].diff()
    
    # New Feature 2: Extreme Regime Indicator (binary flag if forecast exceeds threshold)
    threshold = 1.5
    dataframe['vwap_zscore_10_15min_t6_extreme'] = (
        (dataframe['vwap_zscore_10_15min_t6_prediction'] > threshold) |
        (dataframe['vwap_zscore_10_15min_t6_prediction'] < -threshold)
    ).astype(int)
    
    return dataframe


def predict_vwap_zscore_10_15min_t7(dataframe, model_vwap_zscore_10_15min_t7, scaler_vwap_zscore_10_15min_t7):
    """
    Predicts VWAP ZScore values using the t7 model and adds prediction, comparison,
    acceleration, and extreme regime indicator columns.
    """
    columns_to_include = [
        'hma_vwap_zscore_5', 
        'hma_vwap_zscore_5_slope',
        'hma_vwap_zscore_10', 
        'hma_vwap_zscore_10_slope',
        'hma_vwap_zscore_15', 
        'hma_vwap_zscore_15_slope',
        'hma_vwap_zscore_20', 
        'hma_vwap_zscore_20_slope'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'hma_vwap_zscore_10'
    
    dataframe = predict_sequences(
        model_vwap_zscore_10_15min_t7,
        dataframe,
        columns_to_include,
        seq_length,
        1000,
        'vwap_zscore_10_15min_t7_prediction',
        should_scale=should_scale
    )
    
    dataframe['vwap_zscore_10_15min_t7_diff'] = (
        dataframe['vwap_zscore_10_15min_t7_prediction'] - dataframe[original_feature]
    )
    dataframe['vwap_zscore_10_15min_t7_diff_positive'] = (
        dataframe['vwap_zscore_10_15min_t7_diff'] > 0
    ).astype(int)
    
    dataframe['vwap_zscore_10_15min_t7_accel'] = dataframe['vwap_zscore_10_15min_t7_prediction'].diff()
    
    threshold = 1.5
    dataframe['vwap_zscore_10_15min_t7_extreme'] = (
        (dataframe['vwap_zscore_10_15min_t7_prediction'] > threshold) |
        (dataframe['vwap_zscore_10_15min_t7_prediction'] < -threshold)
    ).astype(int)
    
    return dataframe


def predict_vwap_zscore_10_15min_t8(dataframe, model_vwap_zscore_10_15min_t8, scaler_vwap_zscore_10_15min_t8):
    """
    Predicts VWAP ZScore values using the t8 model and adds prediction, comparison,
    acceleration, and extreme regime indicator columns.
    """
    columns_to_include = [
        'hma_vwap_zscore_5', 
        'hma_vwap_zscore_5_slope',
        'hma_vwap_zscore_10', 
        'hma_vwap_zscore_10_slope',
        'hma_vwap_zscore_15', 
        'hma_vwap_zscore_15_slope',
        'hma_vwap_zscore_20', 
        'hma_vwap_zscore_20_slope'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'hma_vwap_zscore_10'
    
    dataframe = predict_sequences(
        model_vwap_zscore_10_15min_t8,
        dataframe,
        columns_to_include,
        seq_length,
        1000,
        'vwap_zscore_10_15min_t8_prediction',
        should_scale=should_scale
    )
    
    dataframe['vwap_zscore_10_15min_t8_diff'] = (
        dataframe['vwap_zscore_10_15min_t8_prediction'] - dataframe[original_feature]
    )
    dataframe['vwap_zscore_10_15min_t8_diff_positive'] = (
        dataframe['vwap_zscore_10_15min_t8_diff'] > 0
    ).astype(int)
    
    dataframe['vwap_zscore_10_15min_t8_accel'] = dataframe['vwap_zscore_10_15min_t8_prediction'].diff()
    
    threshold = 1.5
    dataframe['vwap_zscore_10_15min_t8_extreme'] = (
        (dataframe['vwap_zscore_10_15min_t8_prediction'] > threshold) |
        (dataframe['vwap_zscore_10_15min_t8_prediction'] < -threshold)
    ).astype(int)
    
    return dataframe


def predict_vwap_zscore_10_15min_t9(dataframe, model_vwap_zscore_10_15min_t9, scaler_vwap_zscore_10_15min_t9):
    """
    Predicts VWAP ZScore values using the t9 model and adds prediction, comparison,
    acceleration, and extreme regime indicator columns.
    """
    columns_to_include = [
        'hma_vwap_zscore_5', 
        'hma_vwap_zscore_5_slope',
        'hma_vwap_zscore_10', 
        'hma_vwap_zscore_10_slope',
        'hma_vwap_zscore_15', 
        'hma_vwap_zscore_15_slope',
        'hma_vwap_zscore_20', 
        'hma_vwap_zscore_20_slope'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'hma_vwap_zscore_10'
    
    dataframe = predict_sequences(
        model_vwap_zscore_10_15min_t9,
        dataframe,
        columns_to_include,
        seq_length,
        1000,
        'vwap_zscore_10_15min_t9_prediction',
        should_scale=should_scale
    )
    
    dataframe['vwap_zscore_10_15min_t9_diff'] = (
        dataframe['vwap_zscore_10_15min_t9_prediction'] - dataframe[original_feature]
    )
    dataframe['vwap_zscore_10_15min_t9_diff_positive'] = (
        dataframe['vwap_zscore_10_15min_t9_diff'] > 0
    ).astype(int)
    
    dataframe['vwap_zscore_10_15min_t9_accel'] = dataframe['vwap_zscore_10_15min_t9_prediction'].diff()
    
    threshold = 1.5
    dataframe['vwap_zscore_10_15min_t9_extreme'] = (
        (dataframe['vwap_zscore_10_15min_t9_prediction'] > threshold) |
        (dataframe['vwap_zscore_10_15min_t9_prediction'] < -threshold)
    ).astype(int)
    
    return dataframe











######################
# SQZ MOM
######################


def predict_sqz_momentum_slope_30_15min_t1(dataframe, model_sqz_momentum_slope_30_15min_t1, scaler_sqz_momentum_slope_30_15min_t1):
    """
    Predicts squeeze momentum slope values using the t1 model and adds prediction and comparison columns.
    
    Args:
        dataframe (pd.DataFrame): Input dataframe containing required columns
        model_sqz_momentum_slope_30_15min_t1: The pre-trained model
        scaler_sqz_momentum_slope_30_15min_t1: The pre-trained scaler
    
    Returns:
        pd.DataFrame: DataFrame with added prediction and analysis columns
    """
    columns_to_include = [
        'sqz_momentum_20',
        'sqz_momentum_slope_20',
        'sqz_momentum_30',
        'sqz_momentum_slope_30',
        'sqz_momentum_40',
        'sqz_momentum_slope_40'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'sqz_momentum_slope_30'
    
    # Make predictions
    dataframe = predict_sequences(
        model_sqz_momentum_slope_30_15min_t1,
        dataframe,
        columns_to_include,
        seq_length,
        1000,
        'sqz_momentum_slope_30_15min_t1_prediction',
        should_scale=should_scale
    )
    
    # Calculate difference between predicted and actual values
    dataframe['sqz_momentum_slope_30_15min_t1_diff'] = (
        dataframe['sqz_momentum_slope_30_15min_t1_prediction'] - 
        dataframe[original_feature]
    )
    
    # Create binary columns for differences and predictions
    dataframe['sqz_momentum_slope_30_15min_t1_prediction_positive'] = (
        dataframe['sqz_momentum_slope_30_15min_t1_prediction'] > 0
    ).astype(int)
    
    dataframe['sqz_momentum_slope_30_15min_t1_diff_positive'] = (
        dataframe['sqz_momentum_slope_30_15min_t1_diff'] > 0
    ).astype(int)
    
    return dataframe

def predict_sqz_momentum_slope_30_15min_t2(dataframe, model_sqz_momentum_slope_30_15min_t2, scaler_sqz_momentum_slope_30_15min_t2):
    """
    Predicts squeeze momentum slope values using the t2 model and adds prediction and comparison columns.
    
    Args:
        dataframe (pd.DataFrame): Input dataframe containing required columns
        model_sqz_momentum_slope_30_15min_t2: The pre-trained model
        scaler_sqz_momentum_slope_30_15min_t2: The pre-trained scaler
    
    Returns:
        pd.DataFrame: DataFrame with added prediction and analysis columns
    """
    columns_to_include = [
        'sqz_momentum_20',
        'sqz_momentum_slope_20',
        'sqz_momentum_30',
        'sqz_momentum_slope_30',
        'sqz_momentum_40',
        'sqz_momentum_slope_40'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'sqz_momentum_slope_30'
    
    # Make predictions
    dataframe = predict_sequences(
        model_sqz_momentum_slope_30_15min_t2,
        dataframe,
        columns_to_include,
        seq_length,
        1000,
        'sqz_momentum_slope_30_15min_t2_prediction',
        should_scale=should_scale
    )
    
    # Calculate difference between predicted and actual values
    dataframe['sqz_momentum_slope_30_15min_t2_diff'] = (
        dataframe['sqz_momentum_slope_30_15min_t2_prediction'] - 
        dataframe[original_feature]
    )
    
    # Create binary columns for differences and predictions
    dataframe['sqz_momentum_slope_30_15min_t2_prediction_positive'] = (
        dataframe['sqz_momentum_slope_30_15min_t2_prediction'] > 0
    ).astype(int)
    
    dataframe['sqz_momentum_slope_30_15min_t2_diff_positive'] = (
        dataframe['sqz_momentum_slope_30_15min_t2_diff'] > 0
    ).astype(int)
    
    return dataframe

def predict_sqz_momentum_slope_30_15min_t3(dataframe, model_sqz_momentum_slope_30_15min_t3, scaler_sqz_momentum_slope_30_15min_t3):
    """
    Predicts squeeze momentum slope values using the t3 model and adds prediction and comparison columns.
    
    Args:
        dataframe (pd.DataFrame): Input dataframe containing required columns
        model_sqz_momentum_slope_30_15min_t3: The pre-trained model
        scaler_sqz_momentum_slope_30_15min_t3: The pre-trained scaler
    
    Returns:
        pd.DataFrame: DataFrame with added prediction and analysis columns
    """
    columns_to_include = [
        'sqz_momentum_20',
        'sqz_momentum_slope_20',
        'sqz_momentum_30',
        'sqz_momentum_slope_30',
        'sqz_momentum_40',
        'sqz_momentum_slope_40'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'sqz_momentum_slope_30'
    
    # Make predictions
    dataframe = predict_sequences(
        model_sqz_momentum_slope_30_15min_t3,
        dataframe,
        columns_to_include,
        seq_length,
        1000,
        'sqz_momentum_slope_30_15min_t3_prediction',
        should_scale=should_scale
    )
    
    # Calculate difference between predicted and actual values
    dataframe['sqz_momentum_slope_30_15min_t3_diff'] = (
        dataframe['sqz_momentum_slope_30_15min_t3_prediction'] - 
        dataframe[original_feature]
    )
    
    # Create binary columns for differences and predictions
    dataframe['sqz_momentum_slope_30_15min_t3_prediction_positive'] = (
        dataframe['sqz_momentum_slope_30_15min_t3_prediction'] > 0
    ).astype(int)
    
    dataframe['sqz_momentum_slope_30_15min_t3_diff_positive'] = (
        dataframe['sqz_momentum_slope_30_15min_t3_diff'] > 0
    ).astype(int)
    
    return dataframe

def predict_sqz_momentum_slope_30_15min_t4(dataframe, model_sqz_momentum_slope_30_15min_t4, scaler_sqz_momentum_slope_30_15min_t4):
    """
    Predicts squeeze momentum slope values using the t4 model and adds prediction and comparison columns.
    
    Args:
        dataframe (pd.DataFrame): Input dataframe containing required columns
        model_sqz_momentum_slope_30_15min_t4: The pre-trained model
        scaler_sqz_momentum_slope_30_15min_t4: The pre-trained scaler
    
    Returns:
        pd.DataFrame: DataFrame with added prediction and analysis columns
    """
    columns_to_include = [
        'sqz_momentum_20',
        'sqz_momentum_slope_20',
        'sqz_momentum_30',
        'sqz_momentum_slope_30',
        'sqz_momentum_40',
        'sqz_momentum_slope_40'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'sqz_momentum_slope_30'
    
    # Make predictions
    dataframe = predict_sequences(
        model_sqz_momentum_slope_30_15min_t4,
        dataframe,
        columns_to_include,
        seq_length,
        1000,
        'sqz_momentum_slope_30_15min_t4_prediction',
        should_scale=should_scale
    )
    
    # Calculate difference between predicted and actual values
    dataframe['sqz_momentum_slope_30_15min_t4_diff'] = (
        dataframe['sqz_momentum_slope_30_15min_t4_prediction'] - 
        dataframe[original_feature]
    )
    
    # Create binary columns for differences and predictions
    dataframe['sqz_momentum_slope_30_15min_t4_prediction_positive'] = (
        dataframe['sqz_momentum_slope_30_15min_t4_prediction'] > 0
    ).astype(int)
    
    dataframe['sqz_momentum_slope_30_15min_t4_diff_positive'] = (
        dataframe['sqz_momentum_slope_30_15min_t4_diff'] > 0
    ).astype(int)
    
    return dataframe

def predict_sqz_momentum_slope_30_15min_t5(dataframe, model_sqz_momentum_slope_30_15min_t5, scaler_sqz_momentum_slope_30_15min_t5):
    """
    Predicts squeeze momentum slope values using the t5 model and adds prediction and comparison columns.
    
    Args:
        dataframe (pd.DataFrame): Input dataframe containing required columns
        model_sqz_momentum_slope_30_15min_t5: The pre-trained model
        scaler_sqz_momentum_slope_30_15min_t5: The pre-trained scaler
    
    Returns:
        pd.DataFrame: DataFrame with added prediction and analysis columns
    """
    columns_to_include = [
        'sqz_momentum_20',
        'sqz_momentum_slope_20',
        'sqz_momentum_30',
        'sqz_momentum_slope_30',
        'sqz_momentum_40',
        'sqz_momentum_slope_40'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'sqz_momentum_slope_30'
    
    # Make predictions
    dataframe = predict_sequences(
        model_sqz_momentum_slope_30_15min_t5,
        dataframe,
        columns_to_include,
        seq_length,
        1000,
        'sqz_momentum_slope_30_15min_t5_prediction',
        should_scale=should_scale
    )
    
    # Calculate difference between predicted and actual values
    dataframe['sqz_momentum_slope_30_15min_t5_diff'] = (
        dataframe['sqz_momentum_slope_30_15min_t5_prediction'] - 
        dataframe[original_feature]
    )
    
    # Create binary columns for differences and predictions
    dataframe['sqz_momentum_slope_30_15min_t5_prediction_positive'] = (
        dataframe['sqz_momentum_slope_30_15min_t5_prediction'] > 0
    ).astype(int)
    
    dataframe['sqz_momentum_slope_30_15min_t5_diff_positive'] = (
        dataframe['sqz_momentum_slope_30_15min_t5_diff'] > 0
    ).astype(int)
    
    return dataframe

def predict_sqz_momentum_slope_30_15min_t6(dataframe, model_sqz_momentum_slope_30_15min_t6, scaler_sqz_momentum_slope_30_15min_t6):
    """
    Predicts squeeze momentum slope values using the t6 model and adds prediction and comparison columns.
    
    Args:
        dataframe (pd.DataFrame): Input dataframe containing required columns
        model_sqz_momentum_slope_30_15min_t6: The pre-trained model
        scaler_sqz_momentum_slope_30_15min_t6: The pre-trained scaler
    
    Returns:
        pd.DataFrame: DataFrame with added prediction and analysis columns
    """
    columns_to_include = [
        'sqz_momentum_20',
        'sqz_momentum_slope_20',
        'sqz_momentum_30',
        'sqz_momentum_slope_30',
        'sqz_momentum_40',
        'sqz_momentum_slope_40'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'sqz_momentum_slope_30'
    
    # Make predictions
    dataframe = predict_sequences(
        model_sqz_momentum_slope_30_15min_t6,
        dataframe,
        columns_to_include,
        seq_length,
        1000,
        'sqz_momentum_slope_30_15min_t6_prediction',
        should_scale=should_scale
    )
    
    # Calculate difference between predicted and actual values
    dataframe['sqz_momentum_slope_30_15min_t6_diff'] = (
        dataframe['sqz_momentum_slope_30_15min_t6_prediction'] - 
        dataframe[original_feature]
    )
    
    # Create binary columns for differences and predictions
    dataframe['sqz_momentum_slope_30_15min_t6_prediction_positive'] = (
        dataframe['sqz_momentum_slope_30_15min_t6_prediction'] > 0
    ).astype(int)
    
    dataframe['sqz_momentum_slope_30_15min_t6_diff_positive'] = (
        dataframe['sqz_momentum_slope_30_15min_t6_diff'] > 0
    ).astype(int)
    
    return dataframe

def predict_sqz_momentum_slope_30_15min_t7(dataframe, model_sqz_momentum_slope_30_15min_t7, scaler_sqz_momentum_slope_30_15min_t7):
    """
    Predicts squeeze momentum slope values using the t7 model and adds prediction and comparison columns.
    
    Args:
        dataframe (pd.DataFrame): Input dataframe containing required columns
        model_sqz_momentum_slope_30_15min_t7: The pre-trained model
        scaler_sqz_momentum_slope_30_15min_t7: The pre-trained scaler
    
    Returns:
        pd.DataFrame: DataFrame with added prediction and analysis columns
    """
    columns_to_include = [
        'sqz_momentum_20',
        'sqz_momentum_slope_20',
        'sqz_momentum_30',
        'sqz_momentum_slope_30',
        'sqz_momentum_40',
        'sqz_momentum_slope_40'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'sqz_momentum_slope_30'
    
    # Make predictions
    dataframe = predict_sequences(
        model_sqz_momentum_slope_30_15min_t7,
        dataframe,
        columns_to_include,
        seq_length,
        1000,
        'sqz_momentum_slope_30_15min_t7_prediction',
        should_scale=should_scale
    )
    
    # Calculate difference between predicted and actual values
    dataframe['sqz_momentum_slope_30_15min_t7_diff'] = (
        dataframe['sqz_momentum_slope_30_15min_t7_prediction'] - 
        dataframe[original_feature]
    )
    
    # Create binary columns for differences and predictions
    dataframe['sqz_momentum_slope_30_15min_t7_prediction_positive'] = (
        dataframe['sqz_momentum_slope_30_15min_t7_prediction'] > 0
    ).astype(int)
    
    dataframe['sqz_momentum_slope_30_15min_t7_diff_positive'] = (
        dataframe['sqz_momentum_slope_30_15min_t7_diff'] > 0
    ).astype(int)
    
    return dataframe

def predict_sqz_momentum_slope_30_15min_t8(dataframe, model_sqz_momentum_slope_30_15min_t8, scaler_sqz_momentum_slope_30_15min_t8):
    """
    Predicts squeeze momentum slope values using the t8 model and adds prediction and comparison columns.
    
    Args:
        dataframe (pd.DataFrame): Input dataframe containing required columns
        model_sqz_momentum_slope_30_15min_t8: The pre-trained model
        scaler_sqz_momentum_slope_30_15min_t8: The pre-trained scaler
    
    Returns:
        pd.DataFrame: DataFrame with added prediction and analysis columns
    """
    columns_to_include = [
        'sqz_momentum_20',
        'sqz_momentum_slope_20',
        'sqz_momentum_30',
        'sqz_momentum_slope_30',
        'sqz_momentum_40',
        'sqz_momentum_slope_40'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'sqz_momentum_slope_30'
    
    # Make predictions
    dataframe = predict_sequences(
        model_sqz_momentum_slope_30_15min_t8,
        dataframe,
        columns_to_include,
        seq_length,
        1000,
        'sqz_momentum_slope_30_15min_t8_prediction',
        should_scale=should_scale
    )
    
    # Calculate difference between predicted and actual values
    dataframe['sqz_momentum_slope_30_15min_t8_diff'] = (
        dataframe['sqz_momentum_slope_30_15min_t8_prediction'] - 
        dataframe[original_feature]
    )
    
    # Create binary columns for differences and predictions
    dataframe['sqz_momentum_slope_30_15min_t8_prediction_positive'] = (
        dataframe['sqz_momentum_slope_30_15min_t8_prediction'] > 0
    ).astype(int)
    
    dataframe['sqz_momentum_slope_30_15min_t8_diff_positive'] = (
        dataframe['sqz_momentum_slope_30_15min_t8_diff'] > 0
    ).astype(int)
    
    return dataframe







def predict_sqz_momentum_30_15min_t1(dataframe, model_sqz_momentum_30_15min_t1, scaler_sqz_momentum_30_15min_t1):
    """
    Predicts squeeze momentum values using the t1 model and adds prediction and comparison columns.
    
    Args:
        dataframe (pd.DataFrame): Input dataframe containing required columns
        model_sqz_momentum_30_15min_t1: The pre-trained model
        scaler_sqz_momentum_30_15min_t1: The pre-trained scaler
    
    Returns:
        pd.DataFrame: DataFrame with added prediction and analysis columns
    """
    columns_to_include = [
        'sqz_momentum_20',
        'sqz_momentum_slope_20',
        'sqz_momentum_30',
        'sqz_momentum_slope_30',
        'sqz_momentum_40',
        'sqz_momentum_slope_40'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'sqz_momentum_30'
    
    # Make predictions
    dataframe = predict_sequences(
        model_sqz_momentum_30_15min_t1,
        dataframe,
        columns_to_include,
        seq_length,
        1000,
        'sqz_momentum_30_15min_t1_prediction',
        should_scale=should_scale
    )
    
    # Calculate difference between predicted and actual values
    dataframe['sqz_momentum_30_15min_t1_diff'] = (
        dataframe['sqz_momentum_30_15min_t1_prediction'] - 
        dataframe[original_feature]
    )
    
    # Create binary columns for differences and predictions
    dataframe['sqz_momentum_30_15min_t1_prediction_positive'] = (
        dataframe['sqz_momentum_30_15min_t1_prediction'] > 0
    ).astype(int)
    
    dataframe['sqz_momentum_30_15min_t1_diff_positive'] = (
        dataframe['sqz_momentum_30_15min_t1_diff'] > 0
    ).astype(int)
    
    return dataframe

def predict_sqz_momentum_30_15min_t2(dataframe, model_sqz_momentum_30_15min_t2, scaler_sqz_momentum_30_15min_t2):
    """
    Predicts squeeze momentum values using the t2 model and adds prediction and comparison columns.
    
    Args:
        dataframe (pd.DataFrame): Input dataframe containing required columns
        model_sqz_momentum_30_15min_t2: The pre-trained model
        scaler_sqz_momentum_30_15min_t2: The pre-trained scaler
    
    Returns:
        pd.DataFrame: DataFrame with added prediction and analysis columns
    """
    columns_to_include = [
        'sqz_momentum_20',
        'sqz_momentum_slope_20',
        'sqz_momentum_30',
        'sqz_momentum_slope_30',
        'sqz_momentum_40',
        'sqz_momentum_slope_40'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'sqz_momentum_30'
    
    # Make predictions
    dataframe = predict_sequences(
        model_sqz_momentum_30_15min_t2,
        dataframe,
        columns_to_include,
        seq_length,
        1000,
        'sqz_momentum_30_15min_t2_prediction',
        should_scale=should_scale
    )
    
    # Calculate difference between predicted and actual values
    dataframe['sqz_momentum_30_15min_t2_diff'] = (
        dataframe['sqz_momentum_30_15min_t2_prediction'] - 
        dataframe[original_feature]
    )
    
    # Create binary columns for differences and predictions
    dataframe['sqz_momentum_30_15min_t2_prediction_positive'] = (
        dataframe['sqz_momentum_30_15min_t2_prediction'] > 0
    ).astype(int)
    
    dataframe['sqz_momentum_30_15min_t2_diff_positive'] = (
        dataframe['sqz_momentum_30_15min_t2_diff'] > 0
    ).astype(int)
    
    return dataframe

def predict_sqz_momentum_30_15min_t3(dataframe, model_sqz_momentum_30_15min_t3, scaler_sqz_momentum_30_15min_t3):
    """
    Predicts squeeze momentum values using the t3 model and adds prediction and comparison columns.
    
    Args:
        dataframe (pd.DataFrame): Input dataframe containing required columns
        model_sqz_momentum_30_15min_t3: The pre-trained model
        scaler_sqz_momentum_30_15min_t3: The pre-trained scaler
    
    Returns:
        pd.DataFrame: DataFrame with added prediction and analysis columns
    """
    columns_to_include = [
        'sqz_momentum_20',
        'sqz_momentum_slope_20',
        'sqz_momentum_30',
        'sqz_momentum_slope_30',
        'sqz_momentum_40',
        'sqz_momentum_slope_40'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'sqz_momentum_30'
    
    # Make predictions
    dataframe = predict_sequences(
        model_sqz_momentum_30_15min_t3,
        dataframe,
        columns_to_include,
        seq_length,
        1000,
        'sqz_momentum_30_15min_t3_prediction',
        should_scale=should_scale
    )
    
    # Calculate difference between predicted and actual values
    dataframe['sqz_momentum_30_15min_t3_diff'] = (
        dataframe['sqz_momentum_30_15min_t3_prediction'] - 
        dataframe[original_feature]
    )
    
    # Create binary columns for differences and predictions
    dataframe['sqz_momentum_30_15min_t3_prediction_positive'] = (
        dataframe['sqz_momentum_30_15min_t3_prediction'] > 0
    ).astype(int)
    
    dataframe['sqz_momentum_30_15min_t3_diff_positive'] = (
        dataframe['sqz_momentum_30_15min_t3_diff'] > 0
    ).astype(int)
    
    return dataframe

def predict_sqz_momentum_30_15min_t4(dataframe, model_sqz_momentum_30_15min_t4, scaler_sqz_momentum_30_15min_t4):
    """
    Predicts squeeze momentum values using the t4 model and adds prediction and comparison columns.
    
    Args:
        dataframe (pd.DataFrame): Input dataframe containing required columns
        model_sqz_momentum_30_15min_t4: The pre-trained model
        scaler_sqz_momentum_30_15min_t4: The pre-trained scaler
    
    Returns:
        pd.DataFrame: DataFrame with added prediction and analysis columns
    """
    columns_to_include = [
        'sqz_momentum_20',
        'sqz_momentum_slope_20',
        'sqz_momentum_30',
        'sqz_momentum_slope_30',
        'sqz_momentum_40',
        'sqz_momentum_slope_40'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'sqz_momentum_30'
    
    # Make predictions
    dataframe = predict_sequences(
        model_sqz_momentum_30_15min_t4,
        dataframe,
        columns_to_include,
        seq_length,
        1000,
        'sqz_momentum_30_15min_t4_prediction',
        should_scale=should_scale
    )
    
    # Calculate difference between predicted and actual values
    dataframe['sqz_momentum_30_15min_t4_diff'] = (
        dataframe['sqz_momentum_30_15min_t4_prediction'] - 
        dataframe[original_feature]
    )
    
    # Create binary columns for differences and predictions
    dataframe['sqz_momentum_30_15min_t4_prediction_positive'] = (
        dataframe['sqz_momentum_30_15min_t4_prediction'] > 0
    ).astype(int)
    
    dataframe['sqz_momentum_30_15min_t4_diff_positive'] = (
        dataframe['sqz_momentum_30_15min_t4_diff'] > 0
    ).astype(int)
    
    return dataframe

def predict_sqz_momentum_30_15min_t5(dataframe, model_sqz_momentum_30_15min_t5, scaler_sqz_momentum_30_15min_t5):
    """
    Predicts squeeze momentum values using the t5 model and adds prediction and comparison columns.
    
    Args:
        dataframe (pd.DataFrame): Input dataframe containing required columns
        model_sqz_momentum_30_15min_t5: The pre-trained model
        scaler_sqz_momentum_30_15min_t5: The pre-trained scaler
    
    Returns:
        pd.DataFrame: DataFrame with added prediction and analysis columns
    """
    columns_to_include = [
        'sqz_momentum_20',
        'sqz_momentum_slope_20',
        'sqz_momentum_30',
        'sqz_momentum_slope_30',
        'sqz_momentum_40',
        'sqz_momentum_slope_40'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'sqz_momentum_30'
    
    # Make predictions
    dataframe = predict_sequences(
        model_sqz_momentum_30_15min_t5,
        dataframe,
        columns_to_include,
        seq_length,
        1000,
        'sqz_momentum_30_15min_t5_prediction',
        should_scale=should_scale
    )
    
    # Calculate difference between predicted and actual values
    dataframe['sqz_momentum_30_15min_t5_diff'] = (
        dataframe['sqz_momentum_30_15min_t5_prediction'] - 
        dataframe[original_feature]
    )
    
    # Create binary columns for differences and predictions
    dataframe['sqz_momentum_30_15min_t5_prediction_positive'] = (
        dataframe['sqz_momentum_30_15min_t5_prediction'] > 0
    ).astype(int)
    
    dataframe['sqz_momentum_30_15min_t5_diff_positive'] = (
        dataframe['sqz_momentum_30_15min_t5_diff'] > 0
    ).astype(int)
    
    return dataframe


######################
# RSI
######################


def predict_RSI_7_hma_15_15min_t1(dataframe, model_RSI_7_hma_15_15min_t1, scaler_RSI_7_hma_15_15min_t1):
    """
    Predicts RSI HMA values using the t1 model and adds prediction and comparison columns.
    """
    columns_to_include = [
        'close_RSI_7_raw',
        'close_RSI_14_raw',
        'close_RSI_28_raw',
        'close_RSI_7_hma_15',
        'close_RSI_14_hma_15',
        'close_RSI_28_hma_15'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'close_RSI_7_hma_15'
    
    dataframe = predict_sequences(
        model_RSI_7_hma_15_15min_t1,
        dataframe,
        columns_to_include,
        seq_length,
        1000,
        'RSI_7_hma_15_15min_t1_prediction',
        should_scale=should_scale
    )
    
    dataframe['RSI_7_hma_15_15min_t1_diff'] = (
        dataframe['RSI_7_hma_15_15min_t1_prediction'] - 
        dataframe[original_feature]
    )
    
    dataframe['RSI_7_hma_15_15min_t1_diff_positive'] = (
        dataframe['RSI_7_hma_15_15min_t1_diff'] > 0
    ).astype(int)
    
    return dataframe

def predict_RSI_7_hma_15_15min_t2(dataframe, model_RSI_7_hma_15_15min_t2, scaler_RSI_7_hma_15_15min_t2):
    """
    Predicts RSI HMA values using the t2 model and adds prediction and comparison columns.
    """
    columns_to_include = [
        'close_RSI_7_raw',
        'close_RSI_14_raw',
        'close_RSI_28_raw',
        'close_RSI_7_hma_15',
        'close_RSI_14_hma_15',
        'close_RSI_28_hma_15'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'close_RSI_7_hma_15'
    
    dataframe = predict_sequences(
        model_RSI_7_hma_15_15min_t2,
        dataframe,
        columns_to_include,
        seq_length,
        1000,
        'RSI_7_hma_15_15min_t2_prediction',
        should_scale=should_scale
    )
    
    dataframe['RSI_7_hma_15_15min_t2_diff'] = (
        dataframe['RSI_7_hma_15_15min_t2_prediction'] - 
        dataframe[original_feature]
    )
    
    dataframe['RSI_7_hma_15_15min_t2_diff_positive'] = (
        dataframe['RSI_7_hma_15_15min_t2_diff'] > 0
    ).astype(int)
    
    return dataframe

def predict_RSI_7_hma_15_15min_t3(dataframe, model_RSI_7_hma_15_15min_t3, scaler_RSI_7_hma_15_15min_t3):
    """
    Predicts RSI HMA values using the t3 model and adds prediction and comparison columns.
    """
    columns_to_include = [
        'close_RSI_7_raw',
        'close_RSI_14_raw',
        'close_RSI_28_raw',
        'close_RSI_7_hma_15',
        'close_RSI_14_hma_15',
        'close_RSI_28_hma_15'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'close_RSI_7_hma_15'
    
    dataframe = predict_sequences(
        model_RSI_7_hma_15_15min_t3,
        dataframe,
        columns_to_include,
        seq_length,
        1000,
        'RSI_7_hma_15_15min_t3_prediction',
        should_scale=should_scale
    )
    
    dataframe['RSI_7_hma_15_15min_t3_diff'] = (
        dataframe['RSI_7_hma_15_15min_t3_prediction'] - 
        dataframe[original_feature]
    )
    
    dataframe['RSI_7_hma_15_15min_t3_diff_positive'] = (
        dataframe['RSI_7_hma_15_15min_t3_diff'] > 0
    ).astype(int)
    
    return dataframe

def predict_RSI_7_hma_15_15min_t4(dataframe, model_RSI_7_hma_15_15min_t4, scaler_RSI_7_hma_15_15min_t4):
    """
    Predicts RSI HMA values using the t4 model and adds prediction and comparison columns.
    """
    columns_to_include = [
        'close_RSI_7_raw',
        'close_RSI_14_raw',
        'close_RSI_28_raw',
        'close_RSI_7_hma_15',
        'close_RSI_14_hma_15',
        'close_RSI_28_hma_15'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'close_RSI_7_hma_15'
    
    dataframe = predict_sequences(
        model_RSI_7_hma_15_15min_t4,
        dataframe,
        columns_to_include,
        seq_length,
        1000,
        'RSI_7_hma_15_15min_t4_prediction',
        should_scale=should_scale
    )
    
    dataframe['RSI_7_hma_15_15min_t4_diff'] = (
        dataframe['RSI_7_hma_15_15min_t4_prediction'] - 
        dataframe[original_feature]
    )
    
    dataframe['RSI_7_hma_15_15min_t4_diff_positive'] = (
        dataframe['RSI_7_hma_15_15min_t4_diff'] > 0
    ).astype(int)
    
    return dataframe

def predict_RSI_14_hma_15_15min_t1(dataframe, model_RSI_14_hma_15_15min_t1, scaler_RSI_14_hma_15_15min_t1):
    """
    Predicts RSI-14 HMA values using the t1 model and adds prediction and comparison columns.
    """
    columns_to_include = [
        'close_RSI_7_raw',
        'close_RSI_14_raw',
        'close_RSI_28_raw',
        'close_RSI_7_hma_15',
        'close_RSI_14_hma_15',
        'close_RSI_28_hma_15'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'close_RSI_14_hma_15'
    
    dataframe = predict_sequences(
        model_RSI_14_hma_15_15min_t1,
        dataframe,
        columns_to_include,
        seq_length,
        1000,
        'RSI_14_hma_15_15min_t1_prediction',
        should_scale=should_scale
    )
    
    dataframe['RSI_14_hma_15_15min_t1_diff'] = (
        dataframe['RSI_14_hma_15_15min_t1_prediction'] - 
        dataframe[original_feature]
    )
    
    dataframe['RSI_14_hma_15_15min_t1_diff_positive'] = (
        dataframe['RSI_14_hma_15_15min_t1_diff'] > 0
    ).astype(int)
    
    return dataframe

def predict_RSI_14_hma_15_15min_t2(dataframe, model_RSI_14_hma_15_15min_t2, scaler_RSI_14_hma_15_15min_t2):
    """
    Predicts RSI-14 HMA values using the t2 model and adds prediction and comparison columns.
    """
    columns_to_include = [
        'close_RSI_7_raw',
        'close_RSI_14_raw',
        'close_RSI_28_raw',
        'close_RSI_7_hma_15',
        'close_RSI_14_hma_15',
        'close_RSI_28_hma_15'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'close_RSI_14_hma_15'
    
    dataframe = predict_sequences(
        model_RSI_14_hma_15_15min_t2,
        dataframe,
        columns_to_include,
        seq_length,
        1000,
        'RSI_14_hma_15_15min_t2_prediction',
        should_scale=should_scale
    )
    
    dataframe['RSI_14_hma_15_15min_t2_diff'] = (
        dataframe['RSI_14_hma_15_15min_t2_prediction'] - 
        dataframe[original_feature]
    )
    
    dataframe['RSI_14_hma_15_15min_t2_diff_positive'] = (
        dataframe['RSI_14_hma_15_15min_t2_diff'] > 0
    ).astype(int)
    
    return dataframe

def predict_RSI_14_hma_15_15min_t3(dataframe, model_RSI_14_hma_15_15min_t3, scaler_RSI_14_hma_15_15min_t3):
    """
    Predicts RSI-14 HMA values using the t3 model and adds prediction and comparison columns.
    """
    columns_to_include = [
        'close_RSI_7_raw',
        'close_RSI_14_raw',
        'close_RSI_28_raw',
        'close_RSI_7_hma_15',
        'close_RSI_14_hma_15',
        'close_RSI_28_hma_15'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'close_RSI_14_hma_15'
    
    dataframe = predict_sequences(
        model_RSI_14_hma_15_15min_t3,
        dataframe,
        columns_to_include,
        seq_length,
        1000,
        'RSI_14_hma_15_15min_t3_prediction',
        should_scale=should_scale
    )
    
    dataframe['RSI_14_hma_15_15min_t3_diff'] = (
        dataframe['RSI_14_hma_15_15min_t3_prediction'] - 
        dataframe[original_feature]
    )
    
    dataframe['RSI_14_hma_15_15min_t3_diff_positive'] = (
        dataframe['RSI_14_hma_15_15min_t3_diff'] > 0
    ).astype(int)
    
    return dataframe

def predict_RSI_14_hma_15_15min_t4(dataframe, model_RSI_14_hma_15_15min_t4, scaler_RSI_14_hma_15_15min_t4):
    """
    Predicts RSI-14 HMA values using the t4 model and adds prediction and comparison columns.
    """
    columns_to_include = [
        'close_RSI_7_raw',
        'close_RSI_14_raw',
        'close_RSI_28_raw',
        'close_RSI_7_hma_15',
        'close_RSI_14_hma_15',
        'close_RSI_28_hma_15'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'close_RSI_14_hma_15'
    
    dataframe = predict_sequences(
        model_RSI_14_hma_15_15min_t4,
        dataframe,
        columns_to_include,
        seq_length,
        1000,
        'RSI_14_hma_15_15min_t4_prediction',
        should_scale=should_scale
    )
    
    dataframe['RSI_14_hma_15_15min_t4_diff'] = (
        dataframe['RSI_14_hma_15_15min_t4_prediction'] - 
        dataframe[original_feature]
    )
    
    dataframe['RSI_14_hma_15_15min_t4_diff_positive'] = (
        dataframe['RSI_14_hma_15_15min_t4_diff'] > 0
    ).astype(int)
    
    return dataframe

def predict_RSI_28_hma_15_15min_t1(dataframe, model_RSI_28_hma_15_15min_t1, scaler_RSI_28_hma_15_15min_t1):
    """
    Predicts RSI-28 HMA values using the t1 model and adds prediction and comparison columns.
    """
    columns_to_include = [
        'close_RSI_7_raw',
        'close_RSI_28_raw',
        'close_RSI_28_raw',
        'close_RSI_7_hma_15',
        'close_RSI_28_hma_15',
        'close_RSI_28_hma_15'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'close_RSI_28_hma_15'
    
    dataframe = predict_sequences(
        model_RSI_28_hma_15_15min_t1,
        dataframe,
        columns_to_include,
        seq_length,
        1000,
        'RSI_28_hma_15_15min_t1_prediction',
        should_scale=should_scale
    )
    
    dataframe['RSI_28_hma_15_15min_t1_diff'] = (
        dataframe['RSI_28_hma_15_15min_t1_prediction'] - 
        dataframe[original_feature]
    )
    
    dataframe['RSI_28_hma_15_15min_t1_diff_positive'] = (
        dataframe['RSI_28_hma_15_15min_t1_diff'] > 0
    ).astype(int)

    # Add acceleration feature
    dataframe['RSI_28_hma_15_15min_t1_accel'] = dataframe['RSI_28_hma_15_15min_t1_prediction'].diff()
    
    return dataframe

def predict_RSI_28_hma_15_15min_t2(dataframe, model_RSI_28_hma_15_15min_t2, scaler_RSI_28_hma_15_15min_t2):
    """
    Predicts RSI-28 HMA values using the t2 model and adds prediction and comparison columns.
    """
    columns_to_include = [
        'close_RSI_7_raw',
        'close_RSI_28_raw',
        'close_RSI_28_raw',
        'close_RSI_7_hma_15',
        'close_RSI_28_hma_15',
        'close_RSI_28_hma_15'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'close_RSI_28_hma_15'
    
    dataframe = predict_sequences(
        model_RSI_28_hma_15_15min_t2,
        dataframe,
        columns_to_include,
        seq_length,
        1000,
        'RSI_28_hma_15_15min_t2_prediction',
        should_scale=should_scale
    )
    
    dataframe['RSI_28_hma_15_15min_t2_diff'] = (
        dataframe['RSI_28_hma_15_15min_t2_prediction'] - 
        dataframe[original_feature]
    )
    
    dataframe['RSI_28_hma_15_15min_t2_diff_positive'] = (
        dataframe['RSI_28_hma_15_15min_t2_diff'] > 0
    ).astype(int)

    # Add acceleration feature
    dataframe['RSI_28_hma_15_15min_t2_accel'] = dataframe['RSI_28_hma_15_15min_t2_prediction'].diff()
    
    return dataframe

def predict_RSI_28_hma_15_15min_t3(dataframe, model_RSI_28_hma_15_15min_t3, scaler_RSI_28_hma_15_15min_t3):
    """
    Predicts RSI-28 HMA values using the t3 model and adds prediction and comparison columns.
    """
    columns_to_include = [
        'close_RSI_7_raw',
        'close_RSI_28_raw',
        'close_RSI_28_raw',
        'close_RSI_7_hma_15',
        'close_RSI_28_hma_15',
        'close_RSI_28_hma_15'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'close_RSI_28_hma_15'
    
    dataframe = predict_sequences(
        model_RSI_28_hma_15_15min_t3,
        dataframe,
        columns_to_include,
        seq_length,
        1000,
        'RSI_28_hma_15_15min_t3_prediction',
        should_scale=should_scale
    )
    
    dataframe['RSI_28_hma_15_15min_t3_diff'] = (
        dataframe['RSI_28_hma_15_15min_t3_prediction'] - 
        dataframe[original_feature]
    )
    
    dataframe['RSI_28_hma_15_15min_t3_diff_positive'] = (
        dataframe['RSI_28_hma_15_15min_t3_diff'] > 0
    ).astype(int)

    # Add acceleration feature
    dataframe['RSI_28_hma_15_15min_t3_accel'] = dataframe['RSI_28_hma_15_15min_t3_prediction'].diff()
    
    return dataframe

def predict_RSI_28_hma_15_15min_t4(dataframe, model_RSI_28_hma_15_15min_t4, scaler_RSI_28_hma_15_15min_t4):
    """
    Predicts RSI-28 HMA values using the t4 model and adds prediction and comparison columns.
    """
    columns_to_include = [
        'close_RSI_7_raw',
        'close_RSI_28_raw',
        'close_RSI_28_raw',
        'close_RSI_7_hma_15',
        'close_RSI_28_hma_15',
        'close_RSI_28_hma_15'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'close_RSI_28_hma_15'
    
    dataframe = predict_sequences(
        model_RSI_28_hma_15_15min_t4,
        dataframe,
        columns_to_include,
        seq_length,
        1000,
        'RSI_28_hma_15_15min_t4_prediction',
        should_scale=should_scale
    )
    
    dataframe['RSI_28_hma_15_15min_t4_diff'] = (
        dataframe['RSI_28_hma_15_15min_t4_prediction'] - 
        dataframe[original_feature]
    )
    
    dataframe['RSI_28_hma_15_15min_t4_diff_positive'] = (
        dataframe['RSI_28_hma_15_15min_t4_diff'] > 0
    ).astype(int)

    # Add acceleration feature
    dataframe['RSI_28_hma_15_15min_t4_accel'] = dataframe['RSI_28_hma_15_15min_t4_prediction'].diff()
    
    return dataframe







######################
# zscore 
######################


def predict_hma15_5_zscore_t1(ticker_df_adjusted, model_hma15_5_zscore_t1, scaler_hma15_5_zscore_t1):
    """
    Predicts HMA 15 zscore values using the t1 model and adds prediction and comparison columns.
    
    Args:
        ticker_df_adjusted (pd.DataFrame): Input dataframe containing required columns
        model_hma15_5_zscore_t1: The pre-trained model
        scaler_hma15_5_zscore_t1: The pre-trained scaler
    
    Returns:
        pd.DataFrame: DataFrame with added prediction and analysis columns
    """
    columns_to_include = [
        'close_slope_15_5_zscore',
        'close_slope_15_2_zscore',
        'close_slope_25_2_zscore',
        'close_slope_25_5_zscore',
        'close_slope_35_2_zscore'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'close_slope_15_5_zscore'
    
    # Make predictions
    ticker_df_adjusted = predict_sequences(
        model_hma15_5_zscore_t1,
        ticker_df_adjusted,
        columns_to_include,
        seq_length,
        1000,
        'close_slope_15_5_zscore_t1_prediction',
        should_scale=should_scale
    )
    
    # Calculate difference between predicted and actual values
    ticker_df_adjusted['close_slope_15_5_zscore_t1_diff'] = (
        ticker_df_adjusted['close_slope_15_5_zscore_t1_prediction'] - 
        ticker_df_adjusted[original_feature]
    )
    
    # Create binary columns for differences and predictions
    ticker_df_adjusted['close_slope_15_5_zscore_t1_diff_positive'] = (
        ticker_df_adjusted['close_slope_15_5_zscore_t1_diff'] > 0
    ).astype(int)
    
    ticker_df_adjusted['close_slope_15_5_zscore_t1_prediction_positive'] = (
        ticker_df_adjusted['close_slope_15_5_zscore_t1_prediction'] > 0
    ).astype(int)

    # Add acceleration feature (first difference of the prediction)
    ticker_df_adjusted['close_slope_15_5_zscore_t1_accel'] = ticker_df_adjusted['close_slope_15_5_zscore_t1_prediction'].diff()


    # 1) Calculate a 150-bar rolling standard deviation of the ORIGINAL feature
    ticker_df_adjusted['close_slope_15_5_zscore_std_150'] = (
        ticker_df_adjusted[original_feature].rolling(window=150).std()
    )

    # 2) Calculate how many standard deviations above zero the prediction is (if > 0)
    ticker_df_adjusted['std_dev_above_zero_15_5_zscore_t1'] = np.where(
        ticker_df_adjusted['close_slope_15_5_zscore_t1_prediction'] > 0,
        ticker_df_adjusted['close_slope_15_5_zscore_t1_prediction']
        / ticker_df_adjusted['close_slope_15_5_zscore_std_150'],
        0
    )

    # 3) Calculate how many standard deviations below zero the prediction is (if < 0)
    ticker_df_adjusted['std_dev_below_zero_15_5_zscore_t1'] = np.where(
        ticker_df_adjusted['close_slope_15_5_zscore_t1_prediction'] < 0,
        ticker_df_adjusted['close_slope_15_5_zscore_t1_prediction']
        / ticker_df_adjusted['close_slope_15_5_zscore_std_150'],
        0
    )

    
    return ticker_df_adjusted

def predict_hma15_5_zscore_t2(ticker_df_adjusted, model_hma15_5_zscore_t2, scaler_hma15_5_zscore_t2):
    """
    Predicts HMA 15 zscore values using the t2 model and adds prediction and comparison columns.
    
    Args:
        ticker_df_adjusted (pd.DataFrame): Input dataframe containing required columns
        model_hma15_5_zscore_t2: The pre-trained model
        scaler_hma15_5_zscore_t2: The pre-trained scaler
    
    Returns:
        pd.DataFrame: DataFrame with added prediction and analysis columns
    """
    columns_to_include = [
        'close_slope_15_5_zscore',
        'close_slope_15_2_zscore',
        'close_slope_25_2_zscore',
        'close_slope_25_5_zscore',
        'close_slope_35_2_zscore'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'close_slope_15_5_zscore'
    
    # Make predictions
    ticker_df_adjusted = predict_sequences(
        model_hma15_5_zscore_t2,
        ticker_df_adjusted,
        columns_to_include,
        seq_length,
        1000,
        'close_slope_15_5_zscore_t2_prediction',
        should_scale=should_scale
    )
    
    # Calculate difference between predicted and actual values
    ticker_df_adjusted['close_slope_15_5_zscore_t2_diff'] = (
        ticker_df_adjusted['close_slope_15_5_zscore_t2_prediction'] - 
        ticker_df_adjusted[original_feature]
    )
    
    # Create binary columns for differences and predictions
    ticker_df_adjusted['close_slope_15_5_zscore_t2_diff_positive'] = (
        ticker_df_adjusted['close_slope_15_5_zscore_t2_diff'] > 0
    ).astype(int)
    
    ticker_df_adjusted['close_slope_15_5_zscore_t2_prediction_positive'] = (
        ticker_df_adjusted['close_slope_15_5_zscore_t2_prediction'] > 0
    ).astype(int)
    
    # Add acceleration feature (first difference of the prediction)
    ticker_df_adjusted['close_slope_15_5_zscore_t2_accel'] = ticker_df_adjusted['close_slope_15_5_zscore_t2_prediction'].diff()

    import numpy as np

    # 1) Calculate a 150-bar rolling standard deviation of the ORIGINAL feature
    ticker_df_adjusted['close_slope_15_5_zscore_std_150_t2'] = (
        ticker_df_adjusted[original_feature].rolling(window=150).std()
    )

    # 2) Number of standard deviations above zero (only if prediction > 0)
    ticker_df_adjusted['std_dev_above_zero_15_5_zscore_t2'] = np.where(
        ticker_df_adjusted['close_slope_15_5_zscore_t2_prediction'] > 0,
        ticker_df_adjusted['close_slope_15_5_zscore_t2_prediction']
        / ticker_df_adjusted['close_slope_15_5_zscore_std_150_t2'],
        0
    )

    # 3) Number of standard deviations below zero (only if prediction < 0)
    ticker_df_adjusted['std_dev_below_zero_15_5_zscore_t2'] = np.where(
        ticker_df_adjusted['close_slope_15_5_zscore_t2_prediction'] < 0,
        ticker_df_adjusted['close_slope_15_5_zscore_t2_prediction']
        / ticker_df_adjusted['close_slope_15_5_zscore_std_150_t2'],
        0
    )

    return ticker_df_adjusted


def predict_hma15_5_zscore_t3(ticker_df_adjusted, model_hma15_5_zscore_t3, scaler_hma15_5_zscore_t3):
    """
    Predicts HMA 15 zscore values using the t3 model and adds prediction and comparison columns.
    
    Args:
        ticker_df_adjusted (pd.DataFrame): Input dataframe containing required columns
        model_hma15_5_zscore_t3: The pre-trained model
        scaler_hma15_5_zscore_t3: The pre-trained scaler
    
    Returns:
        pd.DataFrame: DataFrame with added prediction and analysis columns
    """
    columns_to_include = [
        'close_slope_15_5_zscore',
        'close_slope_15_2_zscore',
        'close_slope_25_2_zscore',
        'close_slope_25_5_zscore',
        'close_slope_35_2_zscore'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'close_slope_15_5_zscore'
    
    # Make predictions
    ticker_df_adjusted = predict_sequences(
        model_hma15_5_zscore_t3,
        ticker_df_adjusted,
        columns_to_include,
        seq_length,
        1000,
        'close_slope_15_5_zscore_t3_prediction',
        should_scale=should_scale
    )
    
    # Calculate difference between predicted and actual values
    ticker_df_adjusted['close_slope_15_5_zscore_t3_diff'] = (
        ticker_df_adjusted['close_slope_15_5_zscore_t3_prediction'] - 
        ticker_df_adjusted[original_feature]
    )
    
    # Create binary columns for differences and predictions
    ticker_df_adjusted['close_slope_15_5_zscore_t3_diff_positive'] = (
        ticker_df_adjusted['close_slope_15_5_zscore_t3_diff'] > 0
    ).astype(int)
    
    ticker_df_adjusted['close_slope_15_5_zscore_t3_prediction_positive'] = (
        ticker_df_adjusted['close_slope_15_5_zscore_t3_prediction'] > 0
    ).astype(int)
    
    # Add acceleration feature (first difference of the prediction)
    ticker_df_adjusted['close_slope_15_5_zscore_t3_accel'] = ticker_df_adjusted['close_slope_15_5_zscore_t3_prediction'].diff()


    # 1) Calculate a 150-bar rolling standard deviation of the ORIGINAL feature
    ticker_df_adjusted['close_slope_15_5_zscore_std_150_t3'] = (
        ticker_df_adjusted[original_feature].rolling(window=150).std()
    )

    # 2) Calculate how many standard deviations above zero the prediction is (if > 0)
    ticker_df_adjusted['std_dev_above_zero_15_5_zscore_t3'] = np.where(
        ticker_df_adjusted['close_slope_15_5_zscore_t3_prediction'] > 0,
        ticker_df_adjusted['close_slope_15_5_zscore_t3_prediction']
        / ticker_df_adjusted['close_slope_15_5_zscore_std_150_t3'],
        0
    )

    # 3) Calculate how many standard deviations below zero the prediction is (if < 0)
    ticker_df_adjusted['std_dev_below_zero_15_5_zscore_t3'] = np.where(
        ticker_df_adjusted['close_slope_15_5_zscore_t3_prediction'] < 0,
        ticker_df_adjusted['close_slope_15_5_zscore_t3_prediction']
        / ticker_df_adjusted['close_slope_15_5_zscore_std_150_t3'],
        0
    )


    return ticker_df_adjusted


def predict_hma15_5_zscore_t4(ticker_df_adjusted, model_hma15_5_zscore_t4, scaler_hma15_5_zscore_t4):
    """
    Predicts HMA 15 zscore values using the t4 model and adds prediction and comparison columns.
    
    Args:
        ticker_df_adjusted (pd.DataFrame): Input dataframe containing required columns
        model_hma15_5_zscore_t4: The pre-trained model
        scaler_hma15_5_zscore_t4: The pre-trained scaler
    
    Returns:
        pd.DataFrame: DataFrame with added prediction and analysis columns
    """
    columns_to_include = [
        'close_slope_15_5_zscore',
        'close_slope_15_2_zscore',
        'close_slope_25_2_zscore',
        'close_slope_25_5_zscore',
        'close_slope_35_2_zscore'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'close_slope_15_5_zscore'
    
    # Make predictions
    ticker_df_adjusted = predict_sequences(
        model_hma15_5_zscore_t4,
        ticker_df_adjusted,
        columns_to_include,
        seq_length,
        1000,
        'close_slope_15_5_zscore_t4_prediction',
        should_scale=should_scale
    )
    
    # Calculate difference between predicted and actual values
    ticker_df_adjusted['close_slope_15_5_zscore_t4_diff'] = (
        ticker_df_adjusted['close_slope_15_5_zscore_t4_prediction'] - 
        ticker_df_adjusted[original_feature]
    )
    
    # Create binary columns for differences and predictions
    ticker_df_adjusted['close_slope_15_5_zscore_t4_diff_positive'] = (
        ticker_df_adjusted['close_slope_15_5_zscore_t4_diff'] > 0
    ).astype(int)
    
    ticker_df_adjusted['close_slope_15_5_zscore_t4_prediction_positive'] = (
        ticker_df_adjusted['close_slope_15_5_zscore_t4_prediction'] > 0
    ).astype(int)
    
    # Add acceleration feature (first difference of the prediction)
    ticker_df_adjusted['close_slope_15_5_zscore_t4_accel'] = ticker_df_adjusted['close_slope_15_5_zscore_t4_prediction'].diff()

    # 1) Calculate a 150-bar rolling standard deviation of the ORIGINAL feature
    ticker_df_adjusted['close_slope_15_5_zscore_std_150_t4'] = (
        ticker_df_adjusted[original_feature].rolling(window=150).std()
    )

    # 2) Calculate how many standard deviations above zero the prediction is (if > 0)
    ticker_df_adjusted['std_dev_above_zero_15_5_zscore_t4'] = np.where(
        ticker_df_adjusted['close_slope_15_5_zscore_t4_prediction'] > 0,
        ticker_df_adjusted['close_slope_15_5_zscore_t4_prediction']
        / ticker_df_adjusted['close_slope_15_5_zscore_std_150_t4'],
        0
    )

    # 3) Calculate how many standard deviations below zero the prediction is (if < 0)
    ticker_df_adjusted['std_dev_below_zero_15_5_zscore_t4'] = np.where(
        ticker_df_adjusted['close_slope_15_5_zscore_t4_prediction'] < 0,
        ticker_df_adjusted['close_slope_15_5_zscore_t4_prediction']
        / ticker_df_adjusted['close_slope_15_5_zscore_std_150_t4'],
        0
    )

    return ticker_df_adjusted



def predict_hma25_5_zscore_t1(ticker_df_adjusted, model_hma25_5_zscore_t1, scaler_hma25_5_zscore_t1):
    """
    Predicts HMA 25 zscore values using the t1 model and adds prediction and comparison columns.
    
    Args:
        ticker_df_adjusted (pd.DataFrame): Input dataframe containing required columns
        model_hma25_5_zscore_t1: The pre-trained model
        scaler_hma25_5_zscore_t1: The pre-trained scaler
    
    Returns:
        pd.DataFrame: DataFrame with added prediction and analysis columns
    """
    columns_to_include = [
        'close_slope_15_5_zscore',
        'close_slope_15_2_zscore',
        'close_slope_25_2_zscore',
        'close_slope_25_5_zscore',
        'close_slope_35_2_zscore',
        'close_slope_35_5_zscore',
        'close_slope_45_2_zscore',
        'close_slope_45_5_zscore'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'close_slope_25_5_zscore'
    
    # Make predictions
    ticker_df_adjusted = predict_sequences(
        model_hma25_5_zscore_t1,
        ticker_df_adjusted,
        columns_to_include,
        seq_length,
        1000,
        'close_slope_25_5_zscore_t1_prediction',
        should_scale=should_scale
    )
    
    # Calculate difference between predicted and actual values
    ticker_df_adjusted['close_slope_25_5_zscore_t1_diff'] = (
        ticker_df_adjusted['close_slope_25_5_zscore_t1_prediction'] - 
        ticker_df_adjusted[original_feature]
    )
    
    # Create binary columns for differences and predictions
    ticker_df_adjusted['close_slope_25_5_zscore_t1_diff_positive'] = (
        ticker_df_adjusted['close_slope_25_5_zscore_t1_diff'] > 0
    ).astype(int)
    
    ticker_df_adjusted['close_slope_25_5_zscore_t1_prediction_positive'] = (
        ticker_df_adjusted['close_slope_25_5_zscore_t1_prediction'] > 0
    ).astype(int)
    
    # Add acceleration feature (first difference of the prediction)
    ticker_df_adjusted['close_slope_25_5_zscore_t1_accel'] = ticker_df_adjusted['close_slope_25_5_zscore_t1_prediction'].diff()
    
    # 1) Calculate a 250-bar rolling standard deviation of the ORIGINAL feature
    ticker_df_adjusted['close_slope_25_5_zscore_std_150'] = (
        ticker_df_adjusted[original_feature].rolling(window=150).std()
    )

    # 2) Calculate how many standard deviations above zero the prediction is (if > 0)
    ticker_df_adjusted['std_dev_above_zero_25_5_zscore_t1'] = np.where(
        ticker_df_adjusted['close_slope_25_5_zscore_t1_prediction'] > 0,
        ticker_df_adjusted['close_slope_25_5_zscore_t1_prediction']
        / ticker_df_adjusted['close_slope_25_5_zscore_std_150'],
        0
    )

    # 3) Calculate how many standard deviations below zero the prediction is (if < 0)
    ticker_df_adjusted['std_dev_below_zero_25_5_zscore_t1'] = np.where(
        ticker_df_adjusted['close_slope_25_5_zscore_t1_prediction'] < 0,
        ticker_df_adjusted['close_slope_25_5_zscore_t1_prediction']
        / ticker_df_adjusted['close_slope_25_5_zscore_std_150'],
        0
    )

    return ticker_df_adjusted


def predict_hma25_5_zscore_t2(ticker_df_adjusted, model_hma25_5_zscore_t2, scaler_hma25_5_zscore_t2):
    """
    Predicts HMA 25 zscore values using the t2 model and adds prediction and comparison columns.
    
    Args:
        ticker_df_adjusted (pd.DataFrame): Input dataframe containing required columns
        model_hma25_5_zscore_t2: The pre-trained model
        scaler_hma25_5_zscore_t2: The pre-trained scaler
    
    Returns:
        pd.DataFrame: DataFrame with added prediction and analysis columns
    """
    columns_to_include = [
        'close_slope_15_5_zscore',
        'close_slope_15_2_zscore',
        'close_slope_25_2_zscore',
        'close_slope_25_5_zscore',
        'close_slope_35_2_zscore',
        'close_slope_35_5_zscore',
        'close_slope_45_2_zscore',
        'close_slope_45_5_zscore'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'close_slope_25_5_zscore'
    
    # Make predictions
    ticker_df_adjusted = predict_sequences(
        model_hma25_5_zscore_t2,
        ticker_df_adjusted,
        columns_to_include,
        seq_length,
        1000,
        'close_slope_25_5_zscore_t2_prediction',
        should_scale=should_scale
    )
    
    # Calculate difference between predicted and actual values
    ticker_df_adjusted['close_slope_25_5_zscore_t2_diff'] = (
        ticker_df_adjusted['close_slope_25_5_zscore_t2_prediction'] - 
        ticker_df_adjusted[original_feature]
    )
    
    # Create binary columns for differences and predictions
    ticker_df_adjusted['close_slope_25_5_zscore_t2_diff_positive'] = (
        ticker_df_adjusted['close_slope_25_5_zscore_t2_diff'] > 0
    ).astype(int)
    
    ticker_df_adjusted['close_slope_25_5_zscore_t2_prediction_positive'] = (
        ticker_df_adjusted['close_slope_25_5_zscore_t2_prediction'] > 0
    ).astype(int)
    
    # Add acceleration feature (first difference of the prediction)
    ticker_df_adjusted['close_slope_25_5_zscore_t2_accel'] = ticker_df_adjusted['close_slope_25_5_zscore_t2_prediction'].diff()
    
    # 1) Calculate a 250-bar rolling standard deviation of the ORIGINAL feature
    ticker_df_adjusted['close_slope_25_5_zscore_std_150'] = (
        ticker_df_adjusted[original_feature].rolling(window=150).std()
    )

    # 2) Calculate how many standard deviations above zero the prediction is (if > 0)
    ticker_df_adjusted['std_dev_above_zero_25_5_zscore_t2'] = np.where(
        ticker_df_adjusted['close_slope_25_5_zscore_t2_prediction'] > 0,
        ticker_df_adjusted['close_slope_25_5_zscore_t2_prediction']
        / ticker_df_adjusted['close_slope_25_5_zscore_std_150'],
        0
    )

    # 3) Calculate how many standard deviations below zero the prediction is (if < 0)
    ticker_df_adjusted['std_dev_below_zero_25_5_zscore_t2'] = np.where(
        ticker_df_adjusted['close_slope_25_5_zscore_t2_prediction'] < 0,
        ticker_df_adjusted['close_slope_25_5_zscore_t2_prediction']
        / ticker_df_adjusted['close_slope_25_5_zscore_std_150'],
        0
    )

    return ticker_df_adjusted


def predict_hma25_5_zscore_t3(ticker_df_adjusted, model_hma25_5_zscore_t3, scaler_hma25_5_zscore_t3):
    """
    Predicts HMA 25 zscore values using the t3 model and adds prediction and comparison columns.
    
    Args:
        ticker_df_adjusted (pd.DataFrame): Input dataframe containing required columns
        model_hma25_5_zscore_t3: The pre-trained model
        scaler_hma25_5_zscore_t3: The pre-trained scaler
    
    Returns:
        pd.DataFrame: DataFrame with added prediction and analysis columns
    """
    columns_to_include = [
        'close_slope_15_5_zscore',
        'close_slope_15_2_zscore',
        'close_slope_25_2_zscore',
        'close_slope_25_5_zscore',
        'close_slope_35_2_zscore',
        'close_slope_35_5_zscore',
        'close_slope_45_2_zscore',
        'close_slope_45_5_zscore'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'close_slope_25_5_zscore'
    
    # Make predictions
    ticker_df_adjusted = predict_sequences(
        model_hma25_5_zscore_t3,
        ticker_df_adjusted,
        columns_to_include,
        seq_length,
        1000,
        'close_slope_25_5_zscore_t3_prediction',
        should_scale=should_scale
    )
    
    # Calculate difference between predicted and actual values
    ticker_df_adjusted['close_slope_25_5_zscore_t3_diff'] = (
        ticker_df_adjusted['close_slope_25_5_zscore_t3_prediction'] - 
        ticker_df_adjusted[original_feature]
    )
    
    # Create binary columns for differences and predictions
    ticker_df_adjusted['close_slope_25_5_zscore_t3_diff_positive'] = (
        ticker_df_adjusted['close_slope_25_5_zscore_t3_diff'] > 0
    ).astype(int)
    
    ticker_df_adjusted['close_slope_25_5_zscore_t3_prediction_positive'] = (
        ticker_df_adjusted['close_slope_25_5_zscore_t3_prediction'] > 0
    ).astype(int)
    
    # Add acceleration feature (first difference of the prediction)
    ticker_df_adjusted['close_slope_25_5_zscore_t3_accel'] = ticker_df_adjusted['close_slope_25_5_zscore_t3_prediction'].diff()
    
    # 1) Calculate a 250-bar rolling standard deviation of the ORIGINAL feature
    ticker_df_adjusted['close_slope_25_5_zscore_std_150'] = (
        ticker_df_adjusted[original_feature].rolling(window=150).std()
    )

    # 2) Calculate how many standard deviations above zero the prediction is (if > 0)
    ticker_df_adjusted['std_dev_above_zero_25_5_zscore_t3'] = np.where(
        ticker_df_adjusted['close_slope_25_5_zscore_t3_prediction'] > 0,
        ticker_df_adjusted['close_slope_25_5_zscore_t3_prediction']
        / ticker_df_adjusted['close_slope_25_5_zscore_std_150'],
        0
    )

    # 3) Calculate how many standard deviations below zero the prediction is (if < 0)
    ticker_df_adjusted['std_dev_below_zero_25_5_zscore_t3'] = np.where(
        ticker_df_adjusted['close_slope_25_5_zscore_t3_prediction'] < 0,
        ticker_df_adjusted['close_slope_25_5_zscore_t3_prediction']
        / ticker_df_adjusted['close_slope_25_5_zscore_std_150'],
        0
    )

    return ticker_df_adjusted


def predict_hma25_5_zscore_t4(ticker_df_adjusted, model_hma25_5_zscore_t4, scaler_hma25_5_zscore_t4):
    """
    Predicts HMA 25 zscore values using the t4 model and adds prediction and comparison columns.
    
    Args:
        ticker_df_adjusted (pd.DataFrame): Input dataframe containing required columns
        model_hma25_5_zscore_t4: The pre-trained model
        scaler_hma25_5_zscore_t4: The pre-trained scaler
    
    Returns:
        pd.DataFrame: DataFrame with added prediction and analysis columns
    """
    columns_to_include = [
        'close_slope_15_5_zscore',
        'close_slope_15_2_zscore',
        'close_slope_25_2_zscore',
        'close_slope_25_5_zscore',
        'close_slope_35_2_zscore',
        'close_slope_35_5_zscore',
        'close_slope_45_2_zscore',
        'close_slope_45_5_zscore'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'close_slope_25_5_zscore'
    
    # Make predictions
    ticker_df_adjusted = predict_sequences(
        model_hma25_5_zscore_t4,
        ticker_df_adjusted,
        columns_to_include,
        seq_length,
        1000,
        'close_slope_25_5_zscore_t4_prediction',
        should_scale=should_scale
    )
    
    # Calculate difference between predicted and actual values
    ticker_df_adjusted['close_slope_25_5_zscore_t4_diff'] = (
        ticker_df_adjusted['close_slope_25_5_zscore_t4_prediction'] - 
        ticker_df_adjusted[original_feature]
    )
    
    # Create binary columns for differences and predictions
    ticker_df_adjusted['close_slope_25_5_zscore_t4_diff_positive'] = (
        ticker_df_adjusted['close_slope_25_5_zscore_t4_diff'] > 0
    ).astype(int)
    
    ticker_df_adjusted['close_slope_25_5_zscore_t4_prediction_positive'] = (
        ticker_df_adjusted['close_slope_25_5_zscore_t4_prediction'] > 0
    ).astype(int)
    
    # Add acceleration feature (first difference of the prediction)
    ticker_df_adjusted['close_slope_25_5_zscore_t4_accel'] = ticker_df_adjusted['close_slope_25_5_zscore_t4_prediction'].diff()
    
    # 1) Calculate a 250-bar rolling standard deviation of the ORIGINAL feature
    ticker_df_adjusted['close_slope_25_5_zscore_std_150'] = (
        ticker_df_adjusted[original_feature].rolling(window=150).std()
    )

    # 2) Calculate how many standard deviations above zero the prediction is (if > 0)
    ticker_df_adjusted['std_dev_above_zero_25_5_zscore_t4'] = np.where(
        ticker_df_adjusted['close_slope_25_5_zscore_t4_prediction'] > 0,
        ticker_df_adjusted['close_slope_25_5_zscore_t4_prediction']
        / ticker_df_adjusted['close_slope_25_5_zscore_std_150'],
        0
    )

    # 3) Calculate how many standard deviations below zero the prediction is (if < 0)
    ticker_df_adjusted['std_dev_below_zero_25_5_zscore_t4'] = np.where(
        ticker_df_adjusted['close_slope_25_5_zscore_t4_prediction'] < 0,
        ticker_df_adjusted['close_slope_25_5_zscore_t4_prediction']
        / ticker_df_adjusted['close_slope_25_5_zscore_std_150'],
        0
    )

    return ticker_df_adjusted


def predict_hma35_5_zscore_t1(ticker_df_adjusted, model_hma35_5_zscore_t1, scaler_hma35_5_zscore_t1):
    """
    Predicts HMA 35 zscore values using the t1 model and adds prediction and comparison columns.
    
    Args:
        ticker_df_adjusted (pd.DataFrame): Input dataframe containing required columns
        model_hma35_5_zscore_t1: The pre-trained model
        scaler_hma35_5_zscore_t1: The pre-trained scaler
    
    Returns:
        pd.DataFrame: DataFrame with added prediction and analysis columns
    """
    columns_to_include = [
        'close_slope_15_5_zscore',
        'close_slope_15_2_zscore',
        'close_slope_25_2_zscore',
        'close_slope_25_5_zscore',
        'close_slope_35_2_zscore',
        'close_slope_35_5_zscore',
        'close_slope_45_2_zscore',
        'close_slope_45_5_zscore'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'close_slope_35_5_zscore'
    
    # Make predictions
    ticker_df_adjusted = predict_sequences(
        model_hma35_5_zscore_t1,
        ticker_df_adjusted,
        columns_to_include,
        seq_length,
        1000,
        'close_slope_35_5_zscore_t1_prediction',
        should_scale=should_scale
    )
    
    # Calculate difference between predicted and actual values
    ticker_df_adjusted['close_slope_35_5_zscore_t1_diff'] = (
        ticker_df_adjusted['close_slope_35_5_zscore_t1_prediction'] - 
        ticker_df_adjusted[original_feature]
    )
    
    # Create binary columns for differences and predictions
    ticker_df_adjusted['close_slope_35_5_zscore_t1_diff_positive'] = (
        ticker_df_adjusted['close_slope_35_5_zscore_t1_diff'] > 0
    ).astype(int)
    
    ticker_df_adjusted['close_slope_35_5_zscore_t1_prediction_positive'] = (
        ticker_df_adjusted['close_slope_35_5_zscore_t1_prediction'] > 0
    ).astype(int)
    
    # Add acceleration feature (first difference of the prediction)
    ticker_df_adjusted['close_slope_35_5_zscore_t1_accel'] = ticker_df_adjusted['close_slope_35_5_zscore_t1_prediction'].diff()
    
    # 1) Calculate a 350-bar rolling standard deviation of the ORIGINAL feature
    ticker_df_adjusted['close_slope_35_5_zscore_std_150'] = (
        ticker_df_adjusted[original_feature].rolling(window=150).std()
    )

    # 2) Calculate how many standard deviations above zero the prediction is (if > 0)
    ticker_df_adjusted['std_dev_above_zero_35_5_zscore_t1'] = np.where(
        ticker_df_adjusted['close_slope_35_5_zscore_t1_prediction'] > 0,
        ticker_df_adjusted['close_slope_35_5_zscore_t1_prediction']
        / ticker_df_adjusted['close_slope_35_5_zscore_std_150'],
        0
    )

    # 3) Calculate how many standard deviations below zero the prediction is (if < 0)
    ticker_df_adjusted['std_dev_below_zero_35_5_zscore_t1'] = np.where(
        ticker_df_adjusted['close_slope_35_5_zscore_t1_prediction'] < 0,
        ticker_df_adjusted['close_slope_35_5_zscore_t1_prediction']
        / ticker_df_adjusted['close_slope_35_5_zscore_std_150'],
        0
    )

    return ticker_df_adjusted


def predict_hma35_5_zscore_t2(ticker_df_adjusted, model_hma35_5_zscore_t2, scaler_hma35_5_zscore_t2):
    """
    Predicts HMA 35 zscore values using the t2 model and adds prediction and comparison columns.
    
    Args:
        ticker_df_adjusted (pd.DataFrame): Input dataframe containing required columns
        model_hma35_5_zscore_t2: The pre-trained model
        scaler_hma35_5_zscore_t2: The pre-trained scaler
    
    Returns:
        pd.DataFrame: DataFrame with added prediction and analysis columns
    """
    columns_to_include = [
        'close_slope_15_5_zscore',
        'close_slope_15_2_zscore',
        'close_slope_25_2_zscore',
        'close_slope_25_5_zscore',
        'close_slope_35_2_zscore',
        'close_slope_35_5_zscore',
        'close_slope_45_2_zscore',
        'close_slope_45_5_zscore'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'close_slope_35_5_zscore'
    
    # Make predictions
    ticker_df_adjusted = predict_sequences(
        model_hma35_5_zscore_t2,
        ticker_df_adjusted,
        columns_to_include,
        seq_length,
        1000,
        'close_slope_35_5_zscore_t2_prediction',
        should_scale=should_scale
    )
    
    # Calculate difference between predicted and actual values
    ticker_df_adjusted['close_slope_35_5_zscore_t2_diff'] = (
        ticker_df_adjusted['close_slope_35_5_zscore_t2_prediction'] - 
        ticker_df_adjusted[original_feature]
    )
    
    # Create binary columns for differences and predictions
    ticker_df_adjusted['close_slope_35_5_zscore_t2_diff_positive'] = (
        ticker_df_adjusted['close_slope_35_5_zscore_t2_diff'] > 0
    ).astype(int)
    
    ticker_df_adjusted['close_slope_35_5_zscore_t2_prediction_positive'] = (
        ticker_df_adjusted['close_slope_35_5_zscore_t2_prediction'] > 0
    ).astype(int)
    
    # Add acceleration feature (first difference of the prediction)
    ticker_df_adjusted['close_slope_35_5_zscore_t2_accel'] = ticker_df_adjusted['close_slope_35_5_zscore_t2_prediction'].diff()
    
    # 1) Calculate a 150-bar rolling standard deviation of the ORIGINAL feature
    ticker_df_adjusted['close_slope_35_5_zscore_std_150'] = (
        ticker_df_adjusted[original_feature].rolling(window=150).std()
    )

    # 2) Calculate how many standard deviations above zero the prediction is (if > 0)
    ticker_df_adjusted['std_dev_above_zero_35_5_zscore_t2'] = np.where(
        ticker_df_adjusted['close_slope_35_5_zscore_t2_prediction'] > 0,
        ticker_df_adjusted['close_slope_35_5_zscore_t2_prediction']
        / ticker_df_adjusted['close_slope_35_5_zscore_std_150'],
        0
    )

    # 3) Calculate how many standard deviations below zero the prediction is (if < 0)
    ticker_df_adjusted['std_dev_below_zero_35_5_zscore_t2'] = np.where(
        ticker_df_adjusted['close_slope_35_5_zscore_t2_prediction'] < 0,
        ticker_df_adjusted['close_slope_35_5_zscore_t2_prediction']
        / ticker_df_adjusted['close_slope_35_5_zscore_std_150'],
        0
    )

    return ticker_df_adjusted


def predict_hma35_5_zscore_t3(ticker_df_adjusted, model_hma35_5_zscore_t3, scaler_hma35_5_zscore_t3):
    """
    Predicts HMA 35 zscore values using the t3 model and adds prediction and comparison columns.
    
    Args:
        ticker_df_adjusted (pd.DataFrame): Input dataframe containing required columns
        model_hma35_5_zscore_t3: The pre-trained model
        scaler_hma35_5_zscore_t3: The pre-trained scaler
    
    Returns:
        pd.DataFrame: DataFrame with added prediction and analysis columns
    """
    columns_to_include = [
        'close_slope_15_5_zscore',
        'close_slope_15_2_zscore',
        'close_slope_25_2_zscore',
        'close_slope_25_5_zscore',
        'close_slope_35_2_zscore',
        'close_slope_35_5_zscore',
        'close_slope_45_2_zscore',
        'close_slope_45_5_zscore'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'close_slope_35_5_zscore'
    
    # Make predictions
    ticker_df_adjusted = predict_sequences(
        model_hma35_5_zscore_t3,
        ticker_df_adjusted,
        columns_to_include,
        seq_length,
        1000,
        'close_slope_35_5_zscore_t3_prediction',
        should_scale=should_scale
    )
    
    # Calculate difference between predicted and actual values
    ticker_df_adjusted['close_slope_35_5_zscore_t3_diff'] = (
        ticker_df_adjusted['close_slope_35_5_zscore_t3_prediction'] - 
        ticker_df_adjusted[original_feature]
    )
    
    # Create binary columns for differences and predictions
    ticker_df_adjusted['close_slope_35_5_zscore_t3_diff_positive'] = (
        ticker_df_adjusted['close_slope_35_5_zscore_t3_diff'] > 0
    ).astype(int)
    
    ticker_df_adjusted['close_slope_35_5_zscore_t3_prediction_positive'] = (
        ticker_df_adjusted['close_slope_35_5_zscore_t3_prediction'] > 0
    ).astype(int)
    
    # Add acceleration feature (first difference of the prediction)
    ticker_df_adjusted['close_slope_35_5_zscore_t3_accel'] = ticker_df_adjusted['close_slope_35_5_zscore_t3_prediction'].diff()
    
    # 1) Calculate a 150-bar rolling standard deviation of the ORIGINAL feature
    ticker_df_adjusted['close_slope_35_5_zscore_std_150'] = (
        ticker_df_adjusted[original_feature].rolling(window=150).std()
    )

    # 2) Calculate how many standard deviations above zero the prediction is (if > 0)
    ticker_df_adjusted['std_dev_above_zero_35_5_zscore_t3'] = np.where(
        ticker_df_adjusted['close_slope_35_5_zscore_t3_prediction'] > 0,
        ticker_df_adjusted['close_slope_35_5_zscore_t3_prediction']
        / ticker_df_adjusted['close_slope_35_5_zscore_std_150'],
        0
    )

    # 3) Calculate how many standard deviations below zero the prediction is (if < 0)
    ticker_df_adjusted['std_dev_below_zero_35_5_zscore_t3'] = np.where(
        ticker_df_adjusted['close_slope_35_5_zscore_t3_prediction'] < 0,
        ticker_df_adjusted['close_slope_35_5_zscore_t3_prediction']
        / ticker_df_adjusted['close_slope_35_5_zscore_std_150'],
        0
    )

    return ticker_df_adjusted


def predict_hma35_5_zscore_t4(ticker_df_adjusted, model_hma35_5_zscore_t4, scaler_hma35_5_zscore_t4):
    """
    Predicts HMA 35 zscore values using the t4 model and adds prediction and comparison columns.
    
    Args:
        ticker_df_adjusted (pd.DataFrame): Input dataframe containing required columns
        model_hma35_5_zscore_t4: The pre-trained model
        scaler_hma35_5_zscore_t4: The pre-trained scaler
    
    Returns:
        pd.DataFrame: DataFrame with added prediction and analysis columns
    """
    columns_to_include = [
        'close_slope_15_5_zscore',
        'close_slope_15_2_zscore',
        'close_slope_25_2_zscore',
        'close_slope_25_5_zscore',
        'close_slope_35_2_zscore',
        'close_slope_35_5_zscore',
        'close_slope_45_2_zscore',
        'close_slope_45_5_zscore'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'close_slope_35_5_zscore'
    
    # Make predictions
    ticker_df_adjusted = predict_sequences(
        model_hma35_5_zscore_t4,
        ticker_df_adjusted,
        columns_to_include,
        seq_length,
        1000,
        'close_slope_35_5_zscore_t4_prediction',
        should_scale=should_scale
    )
    
    # Calculate difference between predicted and actual values
    ticker_df_adjusted['close_slope_35_5_zscore_t4_diff'] = (
        ticker_df_adjusted['close_slope_35_5_zscore_t4_prediction'] - 
        ticker_df_adjusted[original_feature]
    )
    
    # Create binary columns for differences and predictions
    ticker_df_adjusted['close_slope_35_5_zscore_t4_diff_positive'] = (
        ticker_df_adjusted['close_slope_35_5_zscore_t4_diff'] > 0
    ).astype(int)
    
    ticker_df_adjusted['close_slope_35_5_zscore_t4_prediction_positive'] = (
        ticker_df_adjusted['close_slope_35_5_zscore_t4_prediction'] > 0
    ).astype(int)
    
    # Add acceleration feature (first difference of the prediction)
    ticker_df_adjusted['close_slope_35_5_zscore_t4_accel'] = ticker_df_adjusted['close_slope_35_5_zscore_t4_prediction'].diff()
    
    # 1) Calculate a 150-bar rolling standard deviation of the ORIGINAL feature
    ticker_df_adjusted['close_slope_35_5_zscore_std_150'] = (
        ticker_df_adjusted[original_feature].rolling(window=150).std()
    )

    # 2) Calculate how many standard deviations above zero the prediction is (if > 0)
    ticker_df_adjusted['std_dev_above_zero_35_5_zscore_t4'] = np.where(
        ticker_df_adjusted['close_slope_35_5_zscore_t4_prediction'] > 0,
        ticker_df_adjusted['close_slope_35_5_zscore_t4_prediction']
        / ticker_df_adjusted['close_slope_35_5_zscore_std_150'],
        0
    )

    # 3) Calculate how many standard deviations below zero the prediction is (if < 0)
    ticker_df_adjusted['std_dev_below_zero_35_5_zscore_t4'] = np.where(
        ticker_df_adjusted['close_slope_35_5_zscore_t4_prediction'] < 0,
        ticker_df_adjusted['close_slope_35_5_zscore_t4_prediction']
        / ticker_df_adjusted['close_slope_35_5_zscore_std_150'],
        0
    )

    return ticker_df_adjusted


def predict_hma45_5_zscore_t1(ticker_df_adjusted, model_hma45_5_zscore_t1, scaler_hma45_5_zscore_t1):
    """
    Predicts HMA 45 zscore values using the t1 model and adds prediction and comparison columns.
    
    Args:
        ticker_df_adjusted (pd.DataFrame): Input dataframe containing required columns
        model_hma45_5_zscore_t1: The pre-trained model
        scaler_hma45_5_zscore_t1: The pre-trained scaler
    
    Returns:
        pd.DataFrame: DataFrame with added prediction and analysis columns
    """
    columns_to_include = [
        'close_slope_15_5_zscore',
        'close_slope_15_2_zscore',
        'close_slope_25_2_zscore',
        'close_slope_25_5_zscore',
        'close_slope_35_2_zscore',
        'close_slope_35_5_zscore',
        'close_slope_45_2_zscore',
        'close_slope_45_5_zscore'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'close_slope_45_5_zscore'
    
    # Make predictions
    ticker_df_adjusted = predict_sequences(
        model_hma45_5_zscore_t1,
        ticker_df_adjusted,
        columns_to_include,
        seq_length,
        1000,
        'close_slope_45_5_zscore_t1_prediction',
        should_scale=should_scale
    )
    
    # Calculate difference between predicted and actual values
    ticker_df_adjusted['close_slope_45_5_zscore_t1_diff'] = (
        ticker_df_adjusted['close_slope_45_5_zscore_t1_prediction'] - 
        ticker_df_adjusted[original_feature]
    )
    
    # Create binary columns for differences and predictions
    ticker_df_adjusted['close_slope_45_5_zscore_t1_diff_positive'] = (
        ticker_df_adjusted['close_slope_45_5_zscore_t1_diff'] > 0
    ).astype(int)
    
    ticker_df_adjusted['close_slope_45_5_zscore_t1_prediction_positive'] = (
        ticker_df_adjusted['close_slope_45_5_zscore_t1_prediction'] > 0
    ).astype(int)
    
    # Add acceleration feature (first difference of the prediction)
    ticker_df_adjusted['close_slope_45_5_zscore_t1_accel'] = ticker_df_adjusted['close_slope_45_5_zscore_t1_prediction'].diff()
    
    # 1) Calculate a 150-bar rolling standard deviation of the ORIGINAL feature
    ticker_df_adjusted['close_slope_45_5_zscore_std_150'] = (
        ticker_df_adjusted[original_feature].rolling(window=150).std()
    )

    # 2) Calculate how many standard deviations above zero the prediction is (if > 0)
    ticker_df_adjusted['std_dev_above_zero_45_5_zscore_t1'] = np.where(
        ticker_df_adjusted['close_slope_45_5_zscore_t1_prediction'] > 0,
        ticker_df_adjusted['close_slope_45_5_zscore_t1_prediction']
        / ticker_df_adjusted['close_slope_45_5_zscore_std_150'],
        0
    )

    # 3) Calculate how many standard deviations below zero the prediction is (if < 0)
    ticker_df_adjusted['std_dev_below_zero_45_5_zscore_t1'] = np.where(
        ticker_df_adjusted['close_slope_45_5_zscore_t1_prediction'] < 0,
        ticker_df_adjusted['close_slope_45_5_zscore_t1_prediction']
        / ticker_df_adjusted['close_slope_45_5_zscore_std_150'],
        0
    )

    return ticker_df_adjusted


def predict_hma45_5_zscore_t2(ticker_df_adjusted, model_hma45_5_zscore_t2, scaler_hma45_5_zscore_t2):
    """
    Predicts HMA 45 zscore values using the t2 model and adds prediction and comparison columns.
    
    Args:
        ticker_df_adjusted (pd.DataFrame): Input dataframe containing required columns
        model_hma45_5_zscore_t2: The pre-trained model
        scaler_hma45_5_zscore_t2: The pre-trained scaler
    
    Returns:
        pd.DataFrame: DataFrame with added prediction and analysis columns
    """
    columns_to_include = [
        'close_slope_15_5_zscore',
        'close_slope_15_2_zscore',
        'close_slope_25_2_zscore',
        'close_slope_25_5_zscore',
        'close_slope_35_2_zscore',
        'close_slope_35_5_zscore',
        'close_slope_45_2_zscore',
        'close_slope_45_5_zscore'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'close_slope_45_5_zscore'
    
    # Make predictions
    ticker_df_adjusted = predict_sequences(
        model_hma45_5_zscore_t2,
        ticker_df_adjusted,
        columns_to_include,
        seq_length,
        1000,
        'close_slope_45_5_zscore_t2_prediction',
        should_scale=should_scale
    )
    
    # Calculate difference between predicted and actual values
    ticker_df_adjusted['close_slope_45_5_zscore_t2_diff'] = (
        ticker_df_adjusted['close_slope_45_5_zscore_t2_prediction'] - 
        ticker_df_adjusted[original_feature]
    )
    
    # Create binary columns for differences and predictions
    ticker_df_adjusted['close_slope_45_5_zscore_t2_diff_positive'] = (
        ticker_df_adjusted['close_slope_45_5_zscore_t2_diff'] > 0
    ).astype(int)
    
    ticker_df_adjusted['close_slope_45_5_zscore_t2_prediction_positive'] = (
        ticker_df_adjusted['close_slope_45_5_zscore_t2_prediction'] > 0
    ).astype(int)
    
    # Add acceleration feature (first difference of the prediction)
    ticker_df_adjusted['close_slope_45_5_zscore_t2_accel'] = ticker_df_adjusted['close_slope_45_5_zscore_t2_prediction'].diff()
    
    # 1) Calculate a 150-bar rolling standard deviation of the ORIGINAL feature
    ticker_df_adjusted['close_slope_45_5_zscore_std_150'] = (
        ticker_df_adjusted[original_feature].rolling(window=150).std()
    )

    # 2) Calculate how many standard deviations above zero the prediction is (if > 0)
    ticker_df_adjusted['std_dev_above_zero_45_5_zscore_t2'] = np.where(
        ticker_df_adjusted['close_slope_45_5_zscore_t2_prediction'] > 0,
        ticker_df_adjusted['close_slope_45_5_zscore_t2_prediction']
        / ticker_df_adjusted['close_slope_45_5_zscore_std_150'],
        0
    )

    # 3) Calculate how many standard deviations below zero the prediction is (if < 0)
    ticker_df_adjusted['std_dev_below_zero_45_5_zscore_t2'] = np.where(
        ticker_df_adjusted['close_slope_45_5_zscore_t2_prediction'] < 0,
        ticker_df_adjusted['close_slope_45_5_zscore_t2_prediction']
        / ticker_df_adjusted['close_slope_45_5_zscore_std_150'],
        0
    )

    return ticker_df_adjusted


def predict_hma45_5_zscore_t3(ticker_df_adjusted, model_hma45_5_zscore_t3, scaler_hma45_5_zscore_t3):
    """
    Predicts HMA 45 zscore values using the t3 model and adds prediction and comparison columns.
    
    Args:
        ticker_df_adjusted (pd.DataFrame): Input dataframe containing required columns
        model_hma45_5_zscore_t3: The pre-trained model
        scaler_hma45_5_zscore_t3: The pre-trained scaler
    
    Returns:
        pd.DataFrame: DataFrame with added prediction and analysis columns
    """
    columns_to_include = [
        'close_slope_15_5_zscore',
        'close_slope_15_2_zscore',
        'close_slope_25_2_zscore',
        'close_slope_25_5_zscore',
        'close_slope_35_2_zscore',
        'close_slope_35_5_zscore',
        'close_slope_45_2_zscore',
        'close_slope_45_5_zscore'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'close_slope_45_5_zscore'
    
    # Make predictions
    ticker_df_adjusted = predict_sequences(
        model_hma45_5_zscore_t3,
        ticker_df_adjusted,
        columns_to_include,
        seq_length,
        1000,
        'close_slope_45_5_zscore_t3_prediction',
        should_scale=should_scale
    )
    
    # Calculate difference between predicted and actual values
    ticker_df_adjusted['close_slope_45_5_zscore_t3_diff'] = (
        ticker_df_adjusted['close_slope_45_5_zscore_t3_prediction'] - 
        ticker_df_adjusted[original_feature]
    )
    
    # Create binary columns for differences and predictions
    ticker_df_adjusted['close_slope_45_5_zscore_t3_diff_positive'] = (
        ticker_df_adjusted['close_slope_45_5_zscore_t3_diff'] > 0
    ).astype(int)
    
    ticker_df_adjusted['close_slope_45_5_zscore_t3_prediction_positive'] = (
        ticker_df_adjusted['close_slope_45_5_zscore_t3_prediction'] > 0
    ).astype(int)
    
    # Add acceleration feature (first difference of the prediction)
    ticker_df_adjusted['close_slope_45_5_zscore_t3_accel'] = ticker_df_adjusted['close_slope_45_5_zscore_t3_prediction'].diff()
    
    # 1) Calculate a 150-bar rolling standard deviation of the ORIGINAL feature
    ticker_df_adjusted['close_slope_45_5_zscore_std_150'] = (
        ticker_df_adjusted[original_feature].rolling(window=150).std()
    )

    # 2) Calculate how many standard deviations above zero the prediction is (if > 0)
    ticker_df_adjusted['std_dev_above_zero_45_5_zscore_t3'] = np.where(
        ticker_df_adjusted['close_slope_45_5_zscore_t3_prediction'] > 0,
        ticker_df_adjusted['close_slope_45_5_zscore_t3_prediction']
        / ticker_df_adjusted['close_slope_45_5_zscore_std_150'],
        0
    )

    # 3) Calculate how many standard deviations below zero the prediction is (if < 0)
    ticker_df_adjusted['std_dev_below_zero_45_5_zscore_t3'] = np.where(
        ticker_df_adjusted['close_slope_45_5_zscore_t3_prediction'] < 0,
        ticker_df_adjusted['close_slope_45_5_zscore_t3_prediction']
        / ticker_df_adjusted['close_slope_45_5_zscore_std_150'],
        0
    )

    return ticker_df_adjusted


def predict_hma45_5_zscore_t4(ticker_df_adjusted, model_hma45_5_zscore_t4, scaler_hma45_5_zscore_t4):
    """
    Predicts HMA 45 zscore values using the t4 model and adds prediction and comparison columns.
    
    Args:
        ticker_df_adjusted (pd.DataFrame): Input dataframe containing required columns
        model_hma45_5_zscore_t4: The pre-trained model
        scaler_hma45_5_zscore_t4: The pre-trained scaler
    
    Returns:
        pd.DataFrame: DataFrame with added prediction and analysis columns
    """
    columns_to_include = [
        'close_slope_15_5_zscore',
        'close_slope_15_2_zscore',
        'close_slope_25_2_zscore',
        'close_slope_25_5_zscore',
        'close_slope_35_2_zscore',
        'close_slope_35_5_zscore',
        'close_slope_45_2_zscore',
        'close_slope_45_5_zscore'
    ]
    
    seq_length = 7
    should_scale = False
    original_feature = 'close_slope_45_5_zscore'
    
    # Make predictions
    ticker_df_adjusted = predict_sequences(
        model_hma45_5_zscore_t4,
        ticker_df_adjusted,
        columns_to_include,
        seq_length,
        1000,
        'close_slope_45_5_zscore_t4_prediction',
        should_scale=should_scale
    )
    
    # Calculate difference between predicted and actual values
    ticker_df_adjusted['close_slope_45_5_zscore_t4_diff'] = (
        ticker_df_adjusted['close_slope_45_5_zscore_t4_prediction'] - 
        ticker_df_adjusted[original_feature]
    )
    
    # Create binary columns for differences and predictions
    ticker_df_adjusted['close_slope_45_5_zscore_t4_diff_positive'] = (
        ticker_df_adjusted['close_slope_45_5_zscore_t4_diff'] > 0
    ).astype(int)
    
    ticker_df_adjusted['close_slope_45_5_zscore_t4_prediction_positive'] = (
        ticker_df_adjusted['close_slope_45_5_zscore_t4_prediction'] > 0
    ).astype(int)
    
    # Add acceleration feature (first difference of the prediction)
    ticker_df_adjusted['close_slope_45_5_zscore_t4_accel'] = ticker_df_adjusted['close_slope_45_5_zscore_t4_prediction'].diff()
    
    # 1) Calculate a 150-bar rolling standard deviation of the ORIGINAL feature
    ticker_df_adjusted['close_slope_45_5_zscore_std_150'] = (
        ticker_df_adjusted[original_feature].rolling(window=150).std()
    )

    # 2) Calculate how many standard deviations above zero the prediction is (if > 0)
    ticker_df_adjusted['std_dev_above_zero_45_5_zscore_t4'] = np.where(
        ticker_df_adjusted['close_slope_45_5_zscore_t4_prediction'] > 0,
        ticker_df_adjusted['close_slope_45_5_zscore_t4_prediction']
        / ticker_df_adjusted['close_slope_45_5_zscore_std_150'],
        0
    )

    # 3) Calculate how many standard deviations below zero the prediction is (if < 0)
    ticker_df_adjusted['std_dev_below_zero_45_5_zscore_t4'] = np.where(
        ticker_df_adjusted['close_slope_45_5_zscore_t4_prediction'] < 0,
        ticker_df_adjusted['close_slope_45_5_zscore_t4_prediction']
        / ticker_df_adjusted['close_slope_45_5_zscore_std_150'],
        0
    )

    return ticker_df_adjusted












######################
# HMA Binary
######################


def predict_hma25_binary_t3(ticker_df_adjusted, model_hma25_binary_t3, scaler_hma25_binary_t3=None):
    """
    Make predictions using the HMA 25 Binary T+3 RNN model.
    
    Args:
        ticker_df_adjusted (pd.DataFrame): Input DataFrame containing the required columns
        model_hma25_binary_t3: The trained RNN model
        scaler_hma25_binary_t3: Optional scaler for the input features
        
    Returns:
        pd.DataFrame: DataFrame with added prediction columns
    """
    # Define the required columns for prediction
    columns_to_include = [
        'close_slope_35_2_raw',
        'close_slope_25_2_raw',
        'close_slope_15_2_raw',
        'close_slope_25_2_raw_positive',
        'close_2nd_deriv_25_5_positive',
        'close_2nd_deriv_25_5_raw',
    ]
    
    # Set default parameters
    seq_length = 10
    should_scale = False if scaler_hma25_binary_t3 is None else True
    
    # Make predictions using the existing predict_sequences function
    ticker_df_adjusted = predict_sequences(
        model=model_hma25_binary_t3,
        df=ticker_df_adjusted,
        columns_to_include=columns_to_include,
        seq_length=seq_length,
        batch_size=1000,
        new_column_name='close_slope_25_2_binary_t3_prediction',
        should_scale=should_scale
    )
    
    # Add the binary prediction column
    ticker_df_adjusted['close_slope_25_2_binary_t3_prediction_positive'] = (
        ticker_df_adjusted['close_slope_25_2_binary_t3_prediction'] > 0.5
    ).astype(int)
    
    return ticker_df_adjusted

def predict_hma25_binary_t4(ticker_df_adjusted, model_hma25_binary_t4, scaler_hma25_binary_t4):
    """
    Predicts binary classification using the HMA25 model and adds prediction columns to the dataframe.
    
    Args:
        ticker_df_adjusted (pd.DataFrame): Input dataframe containing required columns
        model_hma25_binary_t4: The pre-trained model
        scaler_hma25_binary_t4: The pre-trained scaler
    
    Returns:
        pd.DataFrame: DataFrame with added prediction columns
    """
    columns_to_include = [
        'close_slope_35_2_raw',
        'close_slope_25_2_raw',
        'close_slope_15_2_raw',
        'close_slope_25_2_raw_positive',
        'close_2nd_deriv_25_5_positive',
        'close_2nd_deriv_25_5_raw',
    ]
    
    seq_length = 10
    should_scale = False
    
    # Make predictions
    ticker_df_adjusted = predict_sequences(
        model_hma25_binary_t4,
        ticker_df_adjusted,
        columns_to_include,
        seq_length,
        1000,
        'close_slope_25_2_binary_t4_prediction',
        should_scale=should_scale
    )
    
    # Add binary prediction column
    ticker_df_adjusted['close_slope_25_2_binary_t4_prediction_positive'] = (
        ticker_df_adjusted['close_slope_25_2_binary_t4_prediction'] > 0.5
    ).astype(int)
    
    return ticker_df_adjusted

def predict_hma25_binary_t5(ticker_df_adjusted, model_hma25_binary_t5, scaler_hma25_binary_t5):
    """
    Predicts binary classification using the HMA25 T5 model and adds prediction columns to the dataframe.
    
    Args:
        ticker_df_adjusted (pd.DataFrame): Input dataframe containing required columns
        model_hma25_binary_t5: The pre-trained model
        scaler_hma25_binary_t5: The pre-trained scaler
    
    Returns:
        pd.DataFrame: DataFrame with added prediction columns
    """
    columns_to_include = [
        'close_slope_35_2_raw',
        'close_slope_25_2_raw',
        'close_slope_15_2_raw',
        'close_slope_25_2_raw_positive',
        'close_2nd_deriv_25_5_positive',
        'close_2nd_deriv_25_5_raw',
    ]
    
    seq_length = 10
    should_scale = False
    
    # Make predictions
    ticker_df_adjusted = predict_sequences(
        model_hma25_binary_t5,
        ticker_df_adjusted,
        columns_to_include,
        seq_length,
        1000,
        'close_slope_25_2_binary_t5_prediction',
        should_scale=should_scale
    )
    
    # Add binary prediction column
    ticker_df_adjusted['close_slope_25_2_binary_t5_prediction_positive'] = (
        ticker_df_adjusted['close_slope_25_2_binary_t5_prediction'] > 0.5
    ).astype(int)
    
    return ticker_df_adjusted


######################
# CCI
######################

def predict_close_raw_cci_hma_14_t1(ticker_df_adjusted, model_close_raw_cci_hma_14_t1, scaler_close_raw_cci_hma_14_t1):
   """
   Predicts CCI HMA values using the t1 model and adds prediction and comparison columns to the dataframe.
   
   Args:
       ticker_df_adjusted (pd.DataFrame): Input dataframe containing required columns
       model_close_raw_cci_hma_14_t1: The pre-trained model
       scaler_close_raw_cci_hma_14_t1: The pre-trained scaler
   
   Returns:
       pd.DataFrame: DataFrame with added prediction and comparison columns
   """
   columns_to_include = [
       'close_CCI_short_raw_hma_7',
       'close_CCI_short_raw_hma_10', 
       'close_CCI_short_raw_hma_14',
       'close_CCI_short_raw_hma_18',
       'close_CCI_short_raw_hma_22',
   ]
   
   seq_length = 10
   should_scale = True
   original_feature = 'close_CCI_short_raw_hma_14'
   
   # Make predictions
   ticker_df_adjusted = predict_sequences(
       model_close_raw_cci_hma_14_t1,
       ticker_df_adjusted,
       columns_to_include,
       seq_length,
       1000,
       'close_raw_CCI_hma_14_t1_prediction',
       scaler=scaler_close_raw_cci_hma_14_t1,
       should_scale=should_scale
   )
   
   # Calculate difference between predicted and actual values
   ticker_df_adjusted['close_CCI_short_raw_hma_14_t1_diff'] = (
       ticker_df_adjusted['close_raw_CCI_hma_14_t1_prediction'] - 
       ticker_df_adjusted[original_feature]
   )
   
   # Create binary columns for differences and predictions
   ticker_df_adjusted['close_CCI_short_raw_hma_14_t1_diff_positive'] = (
       ticker_df_adjusted['close_CCI_short_raw_hma_14_t1_diff'] > 0
   ).astype(int)
   
   ticker_df_adjusted['close_raw_CCI_hma_14_t1_prediction_positive'] = (
       ticker_df_adjusted['close_raw_CCI_hma_14_t1_prediction'] > 0
   ).astype(int)
   
   return ticker_df_adjusted

def predict_close_raw_cci_hma_14_t2(ticker_df_adjusted, model_close_raw_cci_hma_14_t2, scaler_close_raw_cci_hma_14_t2):
   """
   Predicts CCI HMA values using the t2 model and adds prediction and comparison columns to the dataframe.
   
   Args:
       ticker_df_adjusted (pd.DataFrame): Input dataframe containing required columns
       model_close_raw_cci_hma_14_t2: The pre-trained model 
       scaler_close_raw_cci_hma_14_t2: The pre-trained scaler
   
   Returns:
       pd.DataFrame: DataFrame with added prediction and comparison columns
   """
   columns_to_include = [
       'close_CCI_short_raw_hma_7',
       'close_CCI_short_raw_hma_10',
       'close_CCI_short_raw_hma_14',
       'close_CCI_short_raw_hma_18',
       'close_CCI_short_raw_hma_22',
   ]
   
   seq_length = 10
   should_scale = True
   original_feature = 'close_CCI_short_raw_hma_14'
   
   # Make predictions
   ticker_df_adjusted = predict_sequences(
       model_close_raw_cci_hma_14_t2,
       ticker_df_adjusted,
       columns_to_include,
       seq_length,
       1000,
       'close_raw_CCI_hma_14_t2_prediction',
       scaler=scaler_close_raw_cci_hma_14_t2,
       should_scale=should_scale
   )
   
   # Calculate difference between predicted and actual values
   ticker_df_adjusted['close_CCI_short_raw_hma_14_t2_diff'] = (
       ticker_df_adjusted['close_raw_CCI_hma_14_t2_prediction'] - 
       ticker_df_adjusted[original_feature]
   )
   
   # Create binary columns for differences and predictions
   ticker_df_adjusted['close_CCI_short_raw_hma_14_t2_diff_positive'] = (
       ticker_df_adjusted['close_CCI_short_raw_hma_14_t2_diff'] > 0
   ).astype(int)
   
   ticker_df_adjusted['close_raw_CCI_hma_14_t2_prediction_positive'] = (
       ticker_df_adjusted['close_raw_CCI_hma_14_t2_prediction'] > 0
   ).astype(int)
   
   return ticker_df_adjusted

def predict_close_raw_cci_hma_14_t3(ticker_df_adjusted, model_close_raw_cci_hma_14_t3, scaler_close_raw_cci_hma_14_t3):
   """
   Predicts CCI HMA values using the t3 model and adds prediction and comparison columns to the dataframe.
   
   Args:
       ticker_df_adjusted (pd.DataFrame): Input dataframe containing required columns
       model_close_raw_cci_hma_14_t3: The pre-trained model
       scaler_close_raw_cci_hma_14_t3: The pre-trained scaler
   
   Returns:
       pd.DataFrame: DataFrame with added prediction and comparison columns
   """
   columns_to_include = [
       'close_CCI_short_raw_hma_7',
       'close_CCI_short_raw_hma_10',
       'close_CCI_short_raw_hma_14',
       'close_CCI_short_raw_hma_18',
       'close_CCI_short_raw_hma_22',
   ]
   
   seq_length = 10
   should_scale = True
   original_feature = 'close_CCI_short_raw_hma_14'
   
   # Make predictions
   ticker_df_adjusted = predict_sequences(
       model_close_raw_cci_hma_14_t3,
       ticker_df_adjusted,
       columns_to_include,
       seq_length,
       1000,
       'close_raw_CCI_hma_14_t3_prediction',
       scaler=scaler_close_raw_cci_hma_14_t3,
       should_scale=should_scale
   )
   
   # Calculate difference between predicted and actual values
   ticker_df_adjusted['close_CCI_short_raw_hma_14_t3_diff'] = (
       ticker_df_adjusted['close_raw_CCI_hma_14_t3_prediction'] - 
       ticker_df_adjusted[original_feature]
   )
   
   # Create binary columns for differences and predictions
   ticker_df_adjusted['close_CCI_short_raw_hma_14_t3_diff_positive'] = (
       ticker_df_adjusted['close_CCI_short_raw_hma_14_t3_diff'] > 0
   ).astype(int)
   
   ticker_df_adjusted['close_raw_CCI_hma_14_t3_prediction_positive'] = (
       ticker_df_adjusted['close_raw_CCI_hma_14_t3_prediction'] > 0
   ).astype(int)
   
   return ticker_df_adjusted

def predict_close_raw_cci_hma_18_t3(ticker_df_adjusted, model_close_raw_cci_hma_18_t3, scaler_close_raw_cci_hma_18_t3):
   """
   Predicts CCI HMA 18 values using the t3 model and adds prediction and comparison columns to the dataframe.
   
   Args:
       ticker_df_adjusted (pd.DataFrame): Input dataframe containing required columns
       model_close_raw_cci_hma_18_t3: The pre-trained model
       scaler_close_raw_cci_hma_18_t3: The pre-trained scaler
   
   Returns:
       pd.DataFrame: DataFrame with added prediction and comparison columns
   """
   columns_to_include = [
       'close_CCI_short_raw_hma_7',
       'close_CCI_short_raw_hma_10',
       'close_CCI_short_raw_hma_14',
       'close_CCI_short_raw_hma_18',
       'close_CCI_short_raw_hma_22',
   ]
   
   seq_length = 10
   should_scale = True
   original_feature = 'close_CCI_short_raw_hma_18'
   
   # Make predictions
   ticker_df_adjusted = predict_sequences(
       model_close_raw_cci_hma_18_t3,
       ticker_df_adjusted,
       columns_to_include,
       seq_length,
       1000,
       'close_raw_CCI_hma_18_t3_prediction',
       scaler=scaler_close_raw_cci_hma_18_t3,
       should_scale=should_scale
   )
   
   # Calculate difference between predicted and actual values
   ticker_df_adjusted['close_CCI_short_raw_hma_18_t3_diff'] = (
       ticker_df_adjusted['close_raw_CCI_hma_18_t3_prediction'] - 
       ticker_df_adjusted[original_feature]
   )
   
   # Create binary columns for differences and predictions
   ticker_df_adjusted['close_CCI_short_raw_hma_18_t3_diff_positive'] = (
       ticker_df_adjusted['close_CCI_short_raw_hma_18_t3_diff'] > 0
   ).astype(int)
   
   ticker_df_adjusted['close_raw_CCI_hma_18_t3_prediction_positive'] = (
       ticker_df_adjusted['close_raw_CCI_hma_18_t3_prediction'] > 0
   ).astype(int)
   
   return ticker_df_adjusted


def predict_close_raw_cci_hma_22_t3(ticker_df_adjusted, model_close_raw_cci_hma_22_t3, scaler_close_raw_cci_hma_22_t3):
   """
   Predicts CCI HMA 22 values using the t3 model and adds prediction and comparison columns to the dataframe.
   
   Args:
       ticker_df_adjusted (pd.DataFrame): Input dataframe containing required columns
       model_close_raw_cci_hma_22_t3: The pre-trained model
       scaler_close_raw_cci_hma_22_t3: The pre-trained scaler
   
   Returns:
       pd.DataFrame: DataFrame with added prediction and comparison columns
   """
   columns_to_include = [
       'close_CCI_short_raw_hma_7',
       'close_CCI_short_raw_hma_10',
       'close_CCI_short_raw_hma_14',
       'close_CCI_short_raw_hma_18',
       'close_CCI_short_raw_hma_22',
   ]
   
   seq_length = 10
   should_scale = True
   original_feature = 'close_CCI_short_raw_hma_22'
   
   # Make predictions
   ticker_df_adjusted = predict_sequences(
       model_close_raw_cci_hma_22_t3,
       ticker_df_adjusted,
       columns_to_include,
       seq_length,
       1000,
       'close_raw_CCI_hma_22_t3_prediction',
       scaler=scaler_close_raw_cci_hma_22_t3,
       should_scale=should_scale
   )
   
   # Calculate difference between predicted and actual values
   ticker_df_adjusted['close_CCI_short_raw_hma_22_t3_diff'] = (
       ticker_df_adjusted['close_raw_CCI_hma_22_t3_prediction'] - 
       ticker_df_adjusted[original_feature]
   )
   
   # Create binary columns for differences and predictions
   ticker_df_adjusted['close_CCI_short_raw_hma_22_t3_diff_positive'] = (
       ticker_df_adjusted['close_CCI_short_raw_hma_22_t3_diff'] > 0
   ).astype(int)
   
   ticker_df_adjusted['close_raw_CCI_hma_22_t3_prediction_positive'] = (
       ticker_df_adjusted['close_raw_CCI_hma_22_t3_prediction'] > 0
   ).astype(int)
   
   return ticker_df_adjusted


######################
# HMA Slope
######################

def predict_hma15_t1(ticker_df_adjusted, model_hma15_t1, scaler_hma15_t1):
   """
   Predicts HMA 15 values using the t1 model and adds prediction, comparison, and standard deviation columns to the dataframe.
   
   Args:
       ticker_df_adjusted (pd.DataFrame): Input dataframe containing required columns
       model_hma15_t1: The pre-trained model
       scaler_hma15_t1: The pre-trained scaler
   
   Returns:
       pd.DataFrame: DataFrame with added prediction and analysis columns
   """
   columns_to_include = [
       'close_slope_25_2_raw',
       'close_slope_15_2_raw',
       'close_slope_10_2_raw',
   ]
   
   seq_length = 10
   should_scale = False
   original_feature = 'close_slope_15_2_raw'
   underlying_original_feature = 'close_15_raw'
   
   # Make predictions
   ticker_df_adjusted = predict_sequences(
       model_hma15_t1,
       ticker_df_adjusted,
       columns_to_include,
       seq_length,
       1000,
       'close_slope_15_2_t1_prediction',
       should_scale=should_scale
   )
   
   # Calculate difference between predicted and actual values
   ticker_df_adjusted['close_slope_15_2_t1_diff'] = (
       ticker_df_adjusted['close_slope_15_2_t1_prediction'] - 
       ticker_df_adjusted[original_feature]
   )
   
   # Create binary columns for differences and predictions
   ticker_df_adjusted['close_slope_15_2_t1_diff_positive'] = (
       ticker_df_adjusted['close_slope_15_2_t1_diff'] > 0
   ).astype(int)
   
   ticker_df_adjusted['close_slope_15_2_t1_prediction_positive'] = (
       ticker_df_adjusted['close_slope_15_2_t1_prediction'] > 0
   ).astype(int)
   
   # Add close plus slope prediction
   ticker_df_adjusted['close_15_plus_slope_prediction_t1'] = (
       ticker_df_adjusted[underlying_original_feature] + 
       ticker_df_adjusted['close_slope_15_2_t1_prediction']
   )
   
   # Calculate trailing standard deviation
   ticker_df_adjusted['close_15_std_150'] = ticker_df_adjusted[underlying_original_feature].rolling(window=150).std()
   
   # Calculate standard deviations above zero
   ticker_df_adjusted['std_dev_above_zero_15_t1'] = np.where(
       ticker_df_adjusted['close_15_plus_slope_prediction_t1'] > 0,
       ticker_df_adjusted['close_15_plus_slope_prediction_t1'] / ticker_df_adjusted['close_15_std_150'],
       0
   )

   ticker_df_adjusted['close_slope_15_2_t1_accel'] = ticker_df_adjusted['close_slope_15_2_t1_prediction'].diff()
   
   return ticker_df_adjusted

def predict_hma15_t2(ticker_df_adjusted, model_hma15_t2, scaler_hma15_t2):
    """
    Predicts HMA 15 values using the t2 model and adds prediction, comparison, and standard deviation columns to the dataframe.
    
    Args:
        ticker_df_adjusted (pd.DataFrame): Input dataframe containing required columns
        model_hma15_t2: The pre-trained model
        scaler_hma15_t2: The pre-trained scaler
    
    Returns:
        pd.DataFrame: DataFrame with added prediction and analysis columns
    """
    columns_to_include = [
        'close_slope_25_2_raw',
        'close_slope_15_2_raw',
        'close_slope_10_2_raw',
    ]
    
    seq_length = 10
    should_scale = False
    original_feature = 'close_slope_15_2_raw'
    underlying_original_feature = 'close_15_raw'
    
    # Make predictions
    ticker_df_adjusted = predict_sequences(
        model_hma15_t2,
        ticker_df_adjusted,
        columns_to_include,
        seq_length,
        1000,
        'close_slope_15_2_t2_prediction',
        should_scale=should_scale
    )
    
    # Calculate difference between predicted and actual values
    ticker_df_adjusted['close_slope_15_2_t2_diff'] = (
        ticker_df_adjusted['close_slope_15_2_t2_prediction'] - 
        ticker_df_adjusted[original_feature]
    )
    
    # Create binary columns for differences and predictions
    ticker_df_adjusted['close_slope_15_2_t2_diff_positive'] = (
        ticker_df_adjusted['close_slope_15_2_t2_diff'] > 0
    ).astype(int)
    
    ticker_df_adjusted['close_slope_15_2_t2_prediction_positive'] = (
        ticker_df_adjusted['close_slope_15_2_t2_prediction'] > 0
    ).astype(int)
    
    # Add close plus slope prediction
    ticker_df_adjusted['close_15_plus_slope_prediction_t2'] = (
        ticker_df_adjusted[underlying_original_feature] + 
        ticker_df_adjusted['close_slope_15_2_t2_prediction']
    )
    
    # Calculate trailing standard deviation
    ticker_df_adjusted['close_15_std_150'] = ticker_df_adjusted[underlying_original_feature].rolling(window=150).std()
    
    # Calculate standard deviations above zero
    ticker_df_adjusted['std_dev_above_zero_15_t2'] = np.where(
        ticker_df_adjusted['close_15_plus_slope_prediction_t2'] > 0,
        ticker_df_adjusted['close_15_plus_slope_prediction_t2'] / ticker_df_adjusted['close_15_std_150'],
        0
    )

    ticker_df_adjusted['close_slope_15_2_t2_accel'] = ticker_df_adjusted['close_slope_15_2_t2_prediction'].diff()
    
    return ticker_df_adjusted

def predict_hma15_t3(ticker_df_adjusted, model_hma15_t3, scaler_hma15_t3):
    """
    Predicts HMA 15 values using the t3 model and adds prediction, comparison, and standard deviation columns to the dataframe.
    
    Args:
        ticker_df_adjusted (pd.DataFrame): Input dataframe containing required columns
        model_hma15_t3: The pre-trained model
        scaler_hma15_t3: The pre-trained scaler
    
    Returns:
        pd.DataFrame: DataFrame with added prediction and analysis columns
    """
    columns_to_include = [
        'close_slope_25_2_raw',
        'close_slope_15_2_raw',
        'close_slope_10_2_raw',
    ]
    
    seq_length = 10
    should_scale = False
    original_feature = 'close_slope_15_2_raw'
    underlying_original_feature = 'close_15_raw'
    
    # Make predictions
    ticker_df_adjusted = predict_sequences(
        model_hma15_t3,
        ticker_df_adjusted,
        columns_to_include,
        seq_length,
        1000,
        'close_slope_15_2_t3_prediction',
        should_scale=should_scale
    )
    
    # Calculate difference between predicted and actual values
    ticker_df_adjusted['close_slope_15_2_t3_diff'] = (
        ticker_df_adjusted['close_slope_15_2_t3_prediction'] - 
        ticker_df_adjusted[original_feature]
    )
    
    # Create binary columns for differences and predictions
    ticker_df_adjusted['close_slope_15_2_t3_diff_positive'] = (
        ticker_df_adjusted['close_slope_15_2_t3_diff'] > 0
    ).astype(int)
    
    ticker_df_adjusted['close_slope_15_2_t3_prediction_positive'] = (
        ticker_df_adjusted['close_slope_15_2_t3_prediction'] > 0
    ).astype(int)
    
    # Add close plus slope prediction
    ticker_df_adjusted['close_15_plus_slope_prediction_t3'] = (
        ticker_df_adjusted[underlying_original_feature] + 
        ticker_df_adjusted['close_slope_15_2_t3_prediction']
    )
    
    # Calculate trailing standard deviation
    ticker_df_adjusted['close_15_std_150'] = ticker_df_adjusted[underlying_original_feature].rolling(window=150).std()
    
    # Calculate standard deviations above zero
    ticker_df_adjusted['std_dev_above_zero_15_t3'] = np.where(
        ticker_df_adjusted['close_15_plus_slope_prediction_t3'] > 0,
        ticker_df_adjusted['close_15_plus_slope_prediction_t3'] / ticker_df_adjusted['close_15_std_150'],
        0
    )

    ticker_df_adjusted['close_slope_15_2_t3_accel'] = ticker_df_adjusted['close_slope_15_2_t3_prediction'].diff()
    
    return ticker_df_adjusted

def predict_hma25_t1(ticker_df_adjusted, model_hma25_t1, scaler_hma25_t1):
    """
    Predicts HMA 25 values using the t1 model and adds prediction, comparison, and standard deviation columns to the dataframe.
    
    Args:
        ticker_df_adjusted (pd.DataFrame): Input dataframe containing required columns
        model_hma25_t1: The pre-trained model
        scaler_hma25_t1: The pre-trained scaler
    
    Returns:
        pd.DataFrame: DataFrame with added prediction and analysis columns
    """
    columns_to_include = [
        'close_slope_35_2_raw',
        'close_slope_25_2_raw',
        'close_slope_15_2_raw',
    ]
    
    seq_length = 10
    should_scale = False
    original_feature = 'close_slope_25_2_raw'
    underlying_original_feature = 'close_25_raw'
    
    # Make predictions
    ticker_df_adjusted = predict_sequences(
        model_hma25_t1,
        ticker_df_adjusted,
        columns_to_include,
        seq_length,
        1000,
        'close_slope_25_2_t1_prediction',
        should_scale=should_scale
    )
    
    # Calculate difference between predicted and actual values
    ticker_df_adjusted['close_slope_25_2_t1_diff'] = (
        ticker_df_adjusted['close_slope_25_2_t1_prediction'] - 
        ticker_df_adjusted[original_feature]
    )
    
    # Create binary columns for differences and predictions
    ticker_df_adjusted['close_slope_25_2_t1_diff_positive'] = (
        ticker_df_adjusted['close_slope_25_2_t1_diff'] > 0
    ).astype(int)
    
    ticker_df_adjusted['close_slope_25_2_t1_prediction_positive'] = (
        ticker_df_adjusted['close_slope_25_2_t1_prediction'] > 0
    ).astype(int)
    
    # Add close plus slope prediction
    ticker_df_adjusted['close_25_plus_slope_prediction_t1'] = (
        ticker_df_adjusted[underlying_original_feature] + 
        ticker_df_adjusted['close_slope_25_2_t1_prediction']
    )
    
    # Calculate trailing standard deviation
    ticker_df_adjusted['close_25_std_150'] = ticker_df_adjusted[underlying_original_feature].rolling(window=150).std()
    
    # Calculate standard deviations above zero
    ticker_df_adjusted['std_dev_above_zero_25_t1'] = np.where(
        ticker_df_adjusted['close_25_plus_slope_prediction_t1'] > 0,
        ticker_df_adjusted['close_25_plus_slope_prediction_t1'] / ticker_df_adjusted['close_25_std_150'],
        0
    )

    ticker_df_adjusted['close_slope_25_2_t1_accel'] = ticker_df_adjusted['close_slope_25_2_t1_prediction'].diff()
    
    return ticker_df_adjusted

def predict_hma25_t2(ticker_df_adjusted, model_hma25_t2, scaler_hma25_t2):
   """
   Predicts HMA 25 values using the t2 model and adds prediction, comparison, and standard deviation columns to the dataframe.
   
   Args:
       ticker_df_adjusted (pd.DataFrame): Input dataframe containing required columns
       model_hma25_t2: The pre-trained model
       scaler_hma25_t2: The pre-trained scaler
   
   Returns:
       pd.DataFrame: DataFrame with added prediction and analysis columns
   """
   columns_to_include = [
       'close_slope_35_2_raw',
       'close_slope_25_2_raw',
       'close_slope_15_2_raw',
   ]
   
   seq_length = 10
   should_scale = False
   original_feature = 'close_slope_25_2_raw'
   underlying_original_feature = 'close_25_raw'
   
   # Make predictions
   ticker_df_adjusted = predict_sequences(
       model_hma25_t2,
       ticker_df_adjusted,
       columns_to_include,
       seq_length,
       1000,
       'close_slope_25_2_t2_prediction',
       should_scale=should_scale
   )
   
   # Calculate difference between predicted and actual values
   ticker_df_adjusted['close_slope_25_2_t2_diff'] = (
       ticker_df_adjusted['close_slope_25_2_t2_prediction'] - 
       ticker_df_adjusted[original_feature]
   )
   
   # Create binary columns for differences and predictions
   ticker_df_adjusted['close_slope_25_2_t2_diff_positive'] = (
       ticker_df_adjusted['close_slope_25_2_t2_diff'] > 0
   ).astype(int)
   
   ticker_df_adjusted['close_slope_25_2_t2_prediction_positive'] = (
       ticker_df_adjusted['close_slope_25_2_t2_prediction'] > 0
   ).astype(int)
   
   # Add close plus slope prediction
   ticker_df_adjusted['close_25_plus_slope_prediction_t2'] = (
       ticker_df_adjusted[underlying_original_feature] + 
       ticker_df_adjusted['close_slope_25_2_t2_prediction']
   )
   
   # Calculate trailing standard deviation
   ticker_df_adjusted['close_25_std_150'] = ticker_df_adjusted[underlying_original_feature].rolling(window=150).std()
   
   # Calculate standard deviations above zero
   ticker_df_adjusted['std_dev_above_zero_25_t2'] = np.where(
       ticker_df_adjusted['close_25_plus_slope_prediction_t2'] > 0,
       ticker_df_adjusted['close_25_plus_slope_prediction_t2'] / ticker_df_adjusted['close_25_std_150'],
       0
   )

   ticker_df_adjusted['close_slope_25_2_t2_accel'] = ticker_df_adjusted['close_slope_25_2_t2_prediction'].diff()
   
   return ticker_df_adjusted

def predict_hma25_t3(ticker_df_adjusted, model_hma25_t3, scaler_hma25_t3):
   """
   Predicts HMA 25 values using the t3 model and adds prediction, comparison, and standard deviation columns to the dataframe.
   
   Args:
       ticker_df_adjusted (pd.DataFrame): Input dataframe containing required columns
       model_hma25_t3: The pre-trained model
       scaler_hma25_t3: The pre-trained scaler
   
   Returns:
       pd.DataFrame: DataFrame with added prediction and analysis columns
   """
   columns_to_include = [
       'close_slope_35_2_raw',
       'close_slope_25_2_raw',
       'close_slope_15_2_raw',
   ]
   
   seq_length = 10
   should_scale = False
   original_feature = 'close_slope_25_2_raw'
   underlying_original_feature = 'close_25_raw'
   
   # Make predictions
   ticker_df_adjusted = predict_sequences(
       model_hma25_t3,
       ticker_df_adjusted,
       columns_to_include,
       seq_length,
       1000,
       'close_slope_25_2_t3_prediction',
       should_scale=should_scale
   )
   
   # Calculate difference between predicted and actual values
   ticker_df_adjusted['close_slope_25_2_t3_diff'] = (
       ticker_df_adjusted['close_slope_25_2_t3_prediction'] - 
       ticker_df_adjusted[original_feature]
   )
   
   # Create binary columns for differences and predictions
   ticker_df_adjusted['close_slope_25_2_t3_diff_positive'] = (
       ticker_df_adjusted['close_slope_25_2_t3_diff'] > 0
   ).astype(int)
   
   ticker_df_adjusted['close_slope_25_2_t3_prediction_positive'] = (
       ticker_df_adjusted['close_slope_25_2_t3_prediction'] > 0
   ).astype(int)
   
   # Add close plus slope prediction
   ticker_df_adjusted['close_25_plus_slope_prediction_t3'] = (
       ticker_df_adjusted[underlying_original_feature] + 
       ticker_df_adjusted['close_slope_25_2_t3_prediction']
   )
   
   # Calculate trailing standard deviation
   ticker_df_adjusted['close_25_std_150'] = ticker_df_adjusted[underlying_original_feature].rolling(window=150).std()
   
   # Calculate standard deviations above zero
   ticker_df_adjusted['std_dev_above_zero_25_t3'] = np.where(
       ticker_df_adjusted['close_25_plus_slope_prediction_t3'] > 0,
       ticker_df_adjusted['close_25_plus_slope_prediction_t3'] / ticker_df_adjusted['close_25_std_150'],
       0
   )

   ticker_df_adjusted['close_slope_25_2_t3_accel'] = ticker_df_adjusted['close_slope_25_2_t3_prediction'].diff()
   
   return ticker_df_adjusted

def predict_hma25_t4(ticker_df_adjusted, model_hma25_t4, scaler_hma25_t4):
    """
    Predicts HMA 25 values using the t4 model and adds prediction, comparison, and standard deviation columns to the dataframe.

    Args:
        ticker_df_adjusted (pd.DataFrame): Input dataframe containing required columns
        model_hma25_t4: The pre-trained model
        scaler_hma25_t4: The pre-trained scaler

    Returns:
        pd.DataFrame: DataFrame with added prediction and analysis columns
    """
    columns_to_include = [
        'close_slope_35_2_raw',
        'close_slope_25_2_raw',
        'close_slope_15_2_raw',
    ]

    seq_length = 10
    should_scale = False
    original_feature = 'close_slope_25_2_raw'
    underlying_original_feature = 'close_25_raw'

    # Make predictions
    ticker_df_adjusted = predict_sequences(
        model_hma25_t4,
        ticker_df_adjusted,
        columns_to_include,
        seq_length,
        1000,
        'close_slope_25_2_t4_prediction',
        should_scale=should_scale
    )

    # Calculate difference between predicted and actual values
    ticker_df_adjusted['close_slope_25_2_t4_diff'] = (
        ticker_df_adjusted['close_slope_25_2_t4_prediction'] - 
        ticker_df_adjusted[original_feature]
    )

    # Create binary columns for differences and predictions
    ticker_df_adjusted['close_slope_25_2_t4_diff_positive'] = (
        ticker_df_adjusted['close_slope_25_2_t4_diff'] > 0
    ).astype(int)

    ticker_df_adjusted['close_slope_25_2_t4_prediction_positive'] = (
        ticker_df_adjusted['close_slope_25_2_t4_prediction'] > 0
    ).astype(int)

    # Add close plus slope prediction
    ticker_df_adjusted['close_25_plus_slope_prediction_t4'] = (
        ticker_df_adjusted[underlying_original_feature] + 
        ticker_df_adjusted['close_slope_25_2_t4_prediction']
    )

    # Calculate trailing standard deviation
    ticker_df_adjusted['close_25_std_150'] = ticker_df_adjusted[underlying_original_feature].rolling(window=150).std()

    # Calculate standard deviations above zero
    ticker_df_adjusted['std_dev_above_zero_25_t4'] = np.where(
        ticker_df_adjusted['close_25_plus_slope_prediction_t4'] > 0,
        ticker_df_adjusted['close_25_plus_slope_prediction_t4'] / ticker_df_adjusted['close_25_std_150'],
        0
    )

    ticker_df_adjusted['close_slope_25_2_t4_accel'] = ticker_df_adjusted['close_slope_25_2_t4_prediction'].diff()

    return ticker_df_adjusted

def predict_hma25_t5(ticker_df_adjusted, model_hma25_t5, scaler_hma25_t5):
    """
    Predicts HMA 25 values using the t5 model and adds prediction, comparison, and standard deviation columns to the dataframe.

    Args:
        ticker_df_adjusted (pd.DataFrame): Input dataframe containing required columns
        model_hma25_t5: The pre-trained model
        scaler_hma25_t5: The pre-trained scaler

    Returns:
        pd.DataFrame: DataFrame with added prediction and analysis columns
    """
    columns_to_include = [
        'close_slope_35_2_raw',
        'close_slope_25_2_raw',
        'close_slope_15_2_raw',
        'close_slope_25_2_raw_positive',
        'close_2nd_deriv_25_5_positive',
    ]

    seq_length = 10
    should_scale = False
    original_feature = 'close_slope_25_2_raw'
    underlying_original_feature = 'close_25_raw'

    # Make predictions
    ticker_df_adjusted = predict_sequences(
        model_hma25_t5,
        ticker_df_adjusted,
        columns_to_include,
        seq_length,
        1000,
        'close_slope_25_2_t5_prediction',
        should_scale=should_scale
    )

    # Calculate difference between predicted and actual values
    ticker_df_adjusted['close_slope_25_2_t5_diff'] = (
        ticker_df_adjusted['close_slope_25_2_t5_prediction'] - 
        ticker_df_adjusted[original_feature]
    )

    # Create binary columns for differences and predictions
    ticker_df_adjusted['close_slope_25_2_t5_diff_positive'] = (
        ticker_df_adjusted['close_slope_25_2_t5_diff'] > 0
    ).astype(int)

    ticker_df_adjusted['close_slope_25_2_t5_prediction_positive'] = (
        ticker_df_adjusted['close_slope_25_2_t5_prediction'] > 0
    ).astype(int)

    # Add close plus slope prediction
    ticker_df_adjusted['close_25_plus_slope_prediction_t5'] = (
        ticker_df_adjusted[underlying_original_feature] + 
        ticker_df_adjusted['close_slope_25_2_t5_prediction']
    )

    # Calculate trailing standard deviation
    ticker_df_adjusted['close_25_std_150'] = ticker_df_adjusted[underlying_original_feature].rolling(window=150).std()

    # Calculate standard deviations above zero
    ticker_df_adjusted['std_dev_above_zero_25_t5'] = np.where(
        ticker_df_adjusted['close_25_plus_slope_prediction_t5'] > 0,
        ticker_df_adjusted['close_25_plus_slope_prediction_t5'] / ticker_df_adjusted['close_25_std_150'],
        0
    )

    ticker_df_adjusted['close_slope_25_2_t5_accel'] = ticker_df_adjusted['close_slope_25_2_t5_prediction'].diff()

    return ticker_df_adjusted

def predict_hma35_t1(ticker_df_adjusted, model_hma35_t1, scaler_hma35_t1):
    """
    Predicts HMA 35 values using the t1 model and adds prediction, comparison, and standard deviation columns to the dataframe.

    Args:
        ticker_df_adjusted (pd.DataFrame): Input dataframe containing required columns
        model_hma35_t1: The pre-trained model
        scaler_hma35_t1: The pre-trained scaler

    Returns:
        pd.DataFrame: DataFrame with added prediction and analysis columns
    """
    columns_to_include = [
        'close_slope_45_2_raw',
        'close_slope_35_2_raw', 
        'close_slope_25_2_raw',
    ]

    seq_length = 10
    should_scale = False
    original_feature = 'close_slope_35_2_raw'
    underlying_original_feature = 'close_35_raw'

    # Make predictions
    ticker_df_adjusted = predict_sequences(
        model_hma35_t1,
        ticker_df_adjusted,
        columns_to_include,
        seq_length,
        1000,
        'close_slope_35_2_t1_prediction',
        should_scale=should_scale
    )

    # Calculate difference between predicted and actual values
    ticker_df_adjusted['close_slope_35_2_t1_diff'] = (
        ticker_df_adjusted['close_slope_35_2_t1_prediction'] - 
        ticker_df_adjusted[original_feature]
    )

    # Create binary columns for differences and predictions
    ticker_df_adjusted['close_slope_35_2_t1_diff_positive'] = (
        ticker_df_adjusted['close_slope_35_2_t1_diff'] > 0
    ).astype(int)

    ticker_df_adjusted['close_slope_35_2_t1_prediction_positive'] = (
        ticker_df_adjusted['close_slope_35_2_t1_prediction'] > 0
    ).astype(int)

    # Add close plus slope prediction
    ticker_df_adjusted['close_35_plus_slope_prediction_t1'] = (
        ticker_df_adjusted[underlying_original_feature] + 
        ticker_df_adjusted['close_slope_35_2_t1_prediction']
    )

    # Calculate trailing standard deviation
    ticker_df_adjusted['close_35_std_150'] = ticker_df_adjusted[underlying_original_feature].rolling(window=150).std()

    # Calculate standard deviations above zero
    ticker_df_adjusted['std_dev_above_zero_35_t1'] = np.where(
        ticker_df_adjusted['close_35_plus_slope_prediction_t1'] > 0,
        ticker_df_adjusted['close_35_plus_slope_prediction_t1'] / ticker_df_adjusted['close_35_std_150'],
        0
    )

    ticker_df_adjusted['close_slope_35_2_t1_accel'] = ticker_df_adjusted['close_slope_35_2_t1_prediction'].diff()

    return ticker_df_adjusted

def predict_hma35_t2(ticker_df_adjusted, model_hma35_t2, scaler_hma35_t2):
    """
    Predicts HMA 35 values using the t2 model and adds prediction, comparison, and standard deviation columns to the dataframe.

    Args:
        ticker_df_adjusted (pd.DataFrame): Input dataframe containing required columns
        model_hma35_t2: The pre-trained model
        scaler_hma35_t2: The pre-trained scaler

    Returns:
        pd.DataFrame: DataFrame with added prediction and analysis columns
    """
    columns_to_include = [
        'close_slope_45_2_raw',
        'close_slope_35_2_raw',
        'close_slope_25_2_raw',
    ]

    seq_length = 10
    should_scale = False
    original_feature = 'close_slope_35_2_raw'
    underlying_original_feature = 'close_35_raw'

    # Make predictions
    ticker_df_adjusted = predict_sequences(
        model_hma35_t2,
        ticker_df_adjusted,
        columns_to_include,
        seq_length,
        1000,
        'close_slope_35_2_t2_prediction',
        should_scale=should_scale
    )

    # Calculate difference between predicted and actual values
    ticker_df_adjusted['close_slope_35_2_t2_diff'] = (
        ticker_df_adjusted['close_slope_35_2_t2_prediction'] - 
        ticker_df_adjusted[original_feature]
    )

    # Create binary columns for differences and predictions
    ticker_df_adjusted['close_slope_35_2_t2_diff_positive'] = (
        ticker_df_adjusted['close_slope_35_2_t2_diff'] > 0
    ).astype(int)

    ticker_df_adjusted['close_slope_35_2_t2_prediction_positive'] = (
        ticker_df_adjusted['close_slope_35_2_t2_prediction'] > 0
    ).astype(int)

    # Add close plus slope prediction
    ticker_df_adjusted['close_35_plus_slope_prediction_t2'] = (
        ticker_df_adjusted[underlying_original_feature] + 
        ticker_df_adjusted['close_slope_35_2_t2_prediction']
    )

    # Calculate trailing standard deviation
    ticker_df_adjusted['close_35_std_150'] = ticker_df_adjusted[underlying_original_feature].rolling(window=150).std()

    # Calculate standard deviations above zero
    ticker_df_adjusted['std_dev_above_zero_35_t2'] = np.where(
        ticker_df_adjusted['close_35_plus_slope_prediction_t2'] > 0,
        ticker_df_adjusted['close_35_plus_slope_prediction_t2'] / ticker_df_adjusted['close_35_std_150'],
        0
    )

    ticker_df_adjusted['close_slope_35_2_t2_accel'] = ticker_df_adjusted['close_slope_35_2_t2_prediction'].diff()

    return ticker_df_adjusted

def predict_hma35_t3(ticker_df_adjusted, model_hma35_t3, scaler_hma35_t3):
    """
    Predicts HMA 35 values using the t3 model and adds prediction, comparison, and standard deviation columns to the dataframe.

    Args:
        ticker_df_adjusted (pd.DataFrame): Input dataframe containing required columns
        model_hma35_t3: The pre-trained model
        scaler_hma35_t3: The pre-trained scaler

    Returns:
        pd.DataFrame: DataFrame with added prediction and analysis columns
    """
    columns_to_include = [
        'close_slope_45_2_raw',
        'close_slope_35_2_raw',
        'close_slope_25_2_raw',
    ]

    seq_length = 10
    should_scale = False
    original_feature = 'close_slope_35_2_raw'
    underlying_original_feature = 'close_35_raw'

    # Make predictions
    ticker_df_adjusted = predict_sequences(
        model_hma35_t3,
        ticker_df_adjusted,
        columns_to_include,
        seq_length,
        1000,
        'close_slope_35_2_t3_prediction',
        should_scale=should_scale
    )

    # Calculate difference between predicted and actual values
    ticker_df_adjusted['close_slope_35_2_t3_diff'] = (
        ticker_df_adjusted['close_slope_35_2_t3_prediction'] - 
        ticker_df_adjusted[original_feature]
    )

    # Create binary columns for differences and predictions
    ticker_df_adjusted['close_slope_35_2_t3_diff_positive'] = (
        ticker_df_adjusted['close_slope_35_2_t3_diff'] > 0
    ).astype(int)

    ticker_df_adjusted['close_slope_35_2_t3_prediction_positive'] = (
        ticker_df_adjusted['close_slope_35_2_t3_prediction'] > 0
    ).astype(int)

    # Add close plus slope prediction
    ticker_df_adjusted['close_35_plus_slope_prediction_t3'] = (
        ticker_df_adjusted[underlying_original_feature] + 
        ticker_df_adjusted['close_slope_35_2_t3_prediction']
    )

    # Calculate trailing standard deviation
    ticker_df_adjusted['close_35_std_150'] = ticker_df_adjusted[underlying_original_feature].rolling(window=150).std()

    # Calculate standard deviations above zero
    ticker_df_adjusted['std_dev_above_zero_35_t3'] = np.where(
        ticker_df_adjusted['close_35_plus_slope_prediction_t3'] > 0,
        ticker_df_adjusted['close_35_plus_slope_prediction_t3'] / ticker_df_adjusted['close_35_std_150'],
        0
    )

    ticker_df_adjusted['close_slope_35_2_t3_accel'] = ticker_df_adjusted['close_slope_35_2_t3_prediction'].diff()

    return ticker_df_adjusted

def predict_hma35_t4(ticker_df_adjusted, model_hma35_t4, scaler_hma35_t4):
    """
    Predicts HMA 35 values using the t4 model and adds prediction, comparison, and standard deviation columns to the dataframe.

    Args:
        ticker_df_adjusted (pd.DataFrame): Input dataframe containing required columns
        model_hma35_t4: The pre-trained model
        scaler_hma35_t4: The pre-trained scaler

    Returns:
        pd.DataFrame: DataFrame with added prediction and analysis columns
    """
    columns_to_include = [
        'close_slope_45_2_raw',
        'close_slope_35_2_raw',
        'close_slope_25_2_raw',
    ]

    seq_length = 10
    should_scale = False
    original_feature = 'close_slope_35_2_raw'
    underlying_original_feature = 'close_35_raw'

    # Make predictions
    ticker_df_adjusted = predict_sequences(
        model_hma35_t4,
        ticker_df_adjusted,
        columns_to_include,
        seq_length,
        1000,
        'close_slope_35_2_t4_prediction',
        should_scale=should_scale
    )

    # Calculate difference between predicted and actual values
    ticker_df_adjusted['close_slope_35_2_t4_diff'] = (
        ticker_df_adjusted['close_slope_35_2_t4_prediction'] - 
        ticker_df_adjusted[original_feature]
    )

    # Create binary columns for differences and predictions
    ticker_df_adjusted['close_slope_35_2_t4_diff_positive'] = (
        ticker_df_adjusted['close_slope_35_2_t4_diff'] > 0
    ).astype(int)

    ticker_df_adjusted['close_slope_35_2_t4_prediction_positive'] = (
        ticker_df_adjusted['close_slope_35_2_t4_prediction'] > 0
    ).astype(int)

    # Add close plus slope prediction
    ticker_df_adjusted['close_35_plus_slope_prediction_t4'] = (
        ticker_df_adjusted[underlying_original_feature] + 
        ticker_df_adjusted['close_slope_35_2_t4_prediction']
    )

    # Calculate trailing standard deviation
    ticker_df_adjusted['close_35_std_150'] = ticker_df_adjusted[underlying_original_feature].rolling(window=200).std()

    # Calculate standard deviations above zero
    ticker_df_adjusted['std_dev_above_zero_35_t4'] = np.where(
        ticker_df_adjusted['close_35_plus_slope_prediction_t4'] > 0,
        ticker_df_adjusted['close_35_plus_slope_prediction_t4'] / ticker_df_adjusted['close_35_std_150'],
        0
    )

    ticker_df_adjusted['close_slope_35_2_t4_accel'] = ticker_df_adjusted['close_slope_35_2_t4_prediction'].diff()

    return ticker_df_adjusted

def predict_hma45_t1(ticker_df_adjusted, model_hma45_t1, scaler_hma45_t1):
    """
    Predicts HMA 45 values using the t1 model and adds prediction, comparison, and standard deviation columns to the dataframe.

    Args:
        ticker_df_adjusted (pd.DataFrame): Input dataframe containing required columns
        model_hma45_t1: The pre-trained model
        scaler_hma45_t1: The pre-trained scaler

    Returns:
        pd.DataFrame: DataFrame with added prediction and analysis columns
    """
    columns_to_include = [
        'close_slope_60_2_raw',
        'close_slope_45_2_raw',
        'close_slope_35_2_raw',
    ]

    seq_length = 10
    should_scale = False
    original_feature = 'close_slope_45_2_raw'
    underlying_original_feature = 'close_45_raw'

    # Make predictions
    ticker_df_adjusted = predict_sequences(
        model_hma45_t1,
        ticker_df_adjusted,
        columns_to_include,
        seq_length,
        1000,
        'close_slope_45_2_t1_prediction',
        should_scale=should_scale
    )

    # Calculate difference between predicted and actual values
    ticker_df_adjusted['close_slope_45_2_t1_diff'] = (
        ticker_df_adjusted['close_slope_45_2_t1_prediction'] - 
        ticker_df_adjusted[original_feature]
    )

    # Create binary columns for differences and predictions
    ticker_df_adjusted['close_slope_45_2_t1_diff_positive'] = (
        ticker_df_adjusted['close_slope_45_2_t1_diff'] > 0
    ).astype(int)

    ticker_df_adjusted['close_slope_45_2_t1_prediction_positive'] = (
        ticker_df_adjusted['close_slope_45_2_t1_prediction'] > 0
    ).astype(int)

    # Add close plus slope prediction
    ticker_df_adjusted['close_45_plus_slope_prediction_t1'] = (
        ticker_df_adjusted[underlying_original_feature] + 
        ticker_df_adjusted['close_slope_45_2_t1_prediction']
    )

    # Calculate trailing standard deviation
    ticker_df_adjusted['close_45_std_150'] = ticker_df_adjusted[underlying_original_feature].rolling(window=150).std()

    # Calculate standard deviations above zero
    ticker_df_adjusted['std_dev_above_zero_45_t1'] = np.where(
        ticker_df_adjusted['close_45_plus_slope_prediction_t1'] > 0,
        ticker_df_adjusted['close_45_plus_slope_prediction_t1'] / ticker_df_adjusted['close_45_std_150'],
        0
    )

    ticker_df_adjusted['close_slope_45_2_t1_accel'] = ticker_df_adjusted['close_slope_45_2_t1_prediction'].diff()

    return ticker_df_adjusted

def predict_hma45_t2(ticker_df_adjusted, model_hma45_t2, scaler_hma45_t2):
    """
    Predicts HMA 45 values using the t2 model and adds prediction, comparison, and standard deviation columns to the dataframe.

    Args:
        ticker_df_adjusted (pd.DataFrame): Input dataframe containing required columns
        model_hma45_t2: The pre-trained model
        scaler_hma45_t2: The pre-trained scaler

    Returns:
        pd.DataFrame: DataFrame with added prediction and analysis columns
    """
    columns_to_include = [
        'close_slope_60_2_raw',
        'close_slope_45_2_raw',
        'close_slope_35_2_raw',
    ]

    seq_length = 10
    should_scale = False
    original_feature = 'close_slope_45_2_raw'
    underlying_original_feature = 'close_45_raw'

    # Make predictions
    ticker_df_adjusted = predict_sequences(
        model_hma45_t2,
        ticker_df_adjusted,
        columns_to_include,
        seq_length,
        1000,
        'close_slope_45_2_t2_prediction',
        should_scale=should_scale
    )

    # Calculate difference between predicted and actual values
    ticker_df_adjusted['close_slope_45_2_t2_diff'] = (
        ticker_df_adjusted['close_slope_45_2_t2_prediction'] - 
        ticker_df_adjusted[original_feature]
    )

    # Create binary columns for differences and predictions
    ticker_df_adjusted['close_slope_45_2_t2_diff_positive'] = (
        ticker_df_adjusted['close_slope_45_2_t2_diff'] > 0
    ).astype(int)

    ticker_df_adjusted['close_slope_45_2_t2_prediction_positive'] = (
        ticker_df_adjusted['close_slope_45_2_t2_prediction'] > 0
    ).astype(int)

    # Add close plus slope prediction
    ticker_df_adjusted['close_45_plus_slope_prediction_t2'] = (
        ticker_df_adjusted[underlying_original_feature] + 
        ticker_df_adjusted['close_slope_45_2_t2_prediction']
    )

    # Calculate trailing standard deviation
    ticker_df_adjusted['close_45_std_150'] = ticker_df_adjusted[underlying_original_feature].rolling(window=150).std()

    # Calculate standard deviations above zero
    ticker_df_adjusted['std_dev_above_zero_45_t2'] = np.where(
        ticker_df_adjusted['close_45_plus_slope_prediction_t2'] > 0,
        ticker_df_adjusted['close_45_plus_slope_prediction_t2'] / ticker_df_adjusted['close_45_std_150'],
        0
    )

    ticker_df_adjusted['close_slope_45_2_t2_accel'] = ticker_df_adjusted['close_slope_45_2_t2_prediction'].diff()

    return ticker_df_adjusted

def predict_hma45_t3(ticker_df_adjusted, model_hma45_t3, scaler_hma45_t3):
    """
    Predicts HMA 45 values using the t3 model and adds prediction, comparison, and standard deviation columns to the dataframe.

    Args:
        ticker_df_adjusted (pd.DataFrame): Input dataframe containing required columns
        model_hma45_t3: The pre-trained model
        scaler_hma45_t3: The pre-trained scaler

    Returns:
        pd.DataFrame: DataFrame with added prediction and analysis columns
    """
    columns_to_include = [
        'close_slope_60_2_raw',
        'close_slope_45_2_raw',
        'close_slope_35_2_raw',
    ]

    seq_length = 10
    should_scale = False
    original_feature = 'close_slope_45_2_raw'
    underlying_original_feature = 'close_45_raw'

    # Make predictions
    ticker_df_adjusted = predict_sequences(
        model_hma45_t3,
        ticker_df_adjusted,
        columns_to_include,
        seq_length,
        1000,
        'close_slope_45_2_t3_prediction',
        should_scale=should_scale
    )

    # Calculate difference between predicted and actual values
    ticker_df_adjusted['close_slope_45_2_t3_diff'] = (
        ticker_df_adjusted['close_slope_45_2_t3_prediction'] - 
        ticker_df_adjusted[original_feature]
    )

    # Create binary columns for differences and predictions
    ticker_df_adjusted['close_slope_45_2_t3_diff_positive'] = (
        ticker_df_adjusted['close_slope_45_2_t3_diff'] > 0
    ).astype(int)

    ticker_df_adjusted['close_slope_45_2_t3_prediction_positive'] = (
        ticker_df_adjusted['close_slope_45_2_t3_prediction'] > 0
    ).astype(int)

    # Add close plus slope prediction
    ticker_df_adjusted['close_45_plus_slope_prediction_t3'] = (
        ticker_df_adjusted[underlying_original_feature] + 
        ticker_df_adjusted['close_slope_45_2_t3_prediction']
    )

    # Calculate trailing standard deviation
    ticker_df_adjusted['close_45_std_150'] = ticker_df_adjusted[underlying_original_feature].rolling(window=150).std()

    # Calculate standard deviations above zero
    ticker_df_adjusted['std_dev_above_zero_45_t3'] = np.where(
        ticker_df_adjusted['close_45_plus_slope_prediction_t3'] > 0,
        ticker_df_adjusted['close_45_plus_slope_prediction_t3'] / ticker_df_adjusted['close_45_std_150'],
        0
    )

    ticker_df_adjusted['close_slope_45_2_t3_accel'] = ticker_df_adjusted['close_slope_45_2_t3_prediction'].diff()

    return ticker_df_adjusted

def predict_hma45_t4(ticker_df_adjusted, model_hma45_t4, scaler_hma45_t4):
    """
    Predicts HMA 45 values using the t4 model and adds prediction, comparison, and standard deviation columns to the dataframe.

    Args:
        ticker_df_adjusted (pd.DataFrame): Input dataframe containing required columns
        model_hma45_t4: The pre-trained model
        scaler_hma45_t4: The pre-trained scaler

    Returns:
        pd.DataFrame: DataFrame with added prediction and analysis columns
    """
    columns_to_include = [
        'close_slope_60_2_raw',
        'close_slope_45_2_raw',
        'close_slope_35_2_raw',
    ]

    seq_length = 10
    should_scale = False
    original_feature = 'close_slope_45_2_raw'
    underlying_original_feature = 'close_45_raw'

    # Make predictions
    ticker_df_adjusted = predict_sequences(
        model_hma45_t4,
        ticker_df_adjusted,
        columns_to_include,
        seq_length,
        1000,
        'close_slope_45_2_t4_prediction',
        should_scale=should_scale
    )

    # Calculate difference between predicted and actual values
    ticker_df_adjusted['close_slope_45_2_t4_diff'] = (
        ticker_df_adjusted['close_slope_45_2_t4_prediction'] - 
        ticker_df_adjusted[original_feature]
    )

    # Create binary columns for differences and predictions
    ticker_df_adjusted['close_slope_45_2_t4_diff_positive'] = (
        ticker_df_adjusted['close_slope_45_2_t4_diff'] > 0
    ).astype(int)

    ticker_df_adjusted['close_slope_45_2_t4_prediction_positive'] = (
        ticker_df_adjusted['close_slope_45_2_t4_prediction'] > 0
    ).astype(int)

    # Add close plus slope prediction
    ticker_df_adjusted['close_45_plus_slope_prediction_t4'] = (
        ticker_df_adjusted[underlying_original_feature] + 
        ticker_df_adjusted['close_slope_45_2_t4_prediction']
    )

    # Calculate trailing standard deviation
    ticker_df_adjusted['close_45_std_150'] = ticker_df_adjusted[underlying_original_feature].rolling(window=150).std()

    # Calculate standard deviations above zero
    ticker_df_adjusted['std_dev_above_zero_45_t4'] = np.where(
        ticker_df_adjusted['close_45_plus_slope_prediction_t4'] > 0,
        ticker_df_adjusted['close_45_plus_slope_prediction_t4'] / ticker_df_adjusted['close_45_std_150'],
        0
    )

    ticker_df_adjusted['close_slope_45_2_t4_accel'] = ticker_df_adjusted['close_slope_45_2_t4_prediction'].diff()

    return ticker_df_adjusted