import pandas as pd 



def drop_non_feature_columns(df, include_symbol=False):
    """
    Drops non-feature columns from the DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame to remove columns from
    include_symbol : bool, default=False
        Whether to keep the 'symbol' column as a feature
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with non-feature columns removed
    """
    # Create list of columns to drop
    columns_to_drop = ['close', 'high', 'low', 'close_raw', 'high_raw', 'low_raw']
    
    if not include_symbol:
        columns_to_drop.append('symbol')
    
    # Drop columns and return new DataFrame
    return df.drop(columns=columns_to_drop, errors='ignore')

# Usage:
# df_features_master_for_model = drop_non_feature_columns(df_features_master_for_model, include_symbol=False)

def drop_na_with_warning(df, threshold_pct=10):
    """
    Drops NA values from DataFrame and warns if percentage of dropped rows exceeds threshold.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame to remove NA values from
    threshold_pct : float, default=10
        Percentage threshold for warning about dropped rows
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with NA values removed
    """
    initial_rows = len(df)
    df_cleaned = df.dropna()
    dropped_rows = initial_rows - len(df_cleaned)
    
    if dropped_rows > 0:
        dropped_pct = (dropped_rows / initial_rows) * 100
        if dropped_pct > threshold_pct:
            print(f"WARNING: Dropping NA values removed {dropped_pct:.1f}% of rows "
                  f"({dropped_rows:,} out of {initial_rows:,} rows)")
    
    return df_cleaned

def split_time_series_with_buffer(df, train_percent=0.7, buffer_percent=0.01):
    """
    Split a time series DataFrame into train and test sets with a buffer period between them.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame with datetime index
    train_percent : float, optional (default=0.7)
        Percentage of data to use for training (0-1)
    buffer_percent : float, optional (default=0.01)
        Percentage of data to use as buffer between train and test sets (0-1)
        
    Returns:
    --------
    tuple : (DataFrame, DataFrame, Timestamp)
        Returns (train_df, test_df, test_data_begin_timestamp)
    """
    if not 0 < train_percent < 1:
        raise ValueError("train_percent must be between 0 and 1")
    
    if not 0 < buffer_percent < (1 - train_percent):
        raise ValueError("Buffer must be positive and smaller than remaining data after train split")
    
    # Calculate split indices
    split_train_index = int(len(df) * train_percent)
    buffer_end_index = int(len(df) * (train_percent + buffer_percent))
    
    # Split the dataframe
    df_train = df.iloc[:split_train_index]
    df_test = df.iloc[buffer_end_index:]
    
    # Store test start timestamp
    test_data_begin_timestamp = df_test.index[0]
    
    # Print date ranges for verification
    print("\nTrain set:")
    print(f"Min date: {df_train.index.min()}")
    print(f"Max date: {df_train.index.max()}")
    
    print("\nBuffer period:")
    print(f"From: {df_train.index.max()}")
    print(f"To: {df_test.index.min()}")
    
    print("\nTest set:")
    print(f"Min date: {df_test.index.min()}")
    print(f"Max date: {df_test.index.max()}")
    
    return df_train, df_test, test_data_begin_timestamp

# Example usage:
# train_df, test_df, test_start = split_time_series_with_buffer(
#     df_features_master_for_model,
#     train_percent=0.7,
#     buffer_percent=0.01
# )

def prepare_and_validate_data(df_train, df_test):
    """
    Prepare training and test datasets by separating features and labels,
    and validate that timestamp is not present and classes match.
    
    Args:
        df_train (pd.DataFrame): Training dataset
        df_test (pd.DataFrame): Test dataset
        
    Returns:
        tuple: (X_train, y_train, X_test, y_test)
    
    Raises:
        AssertionError: If validation checks fail
    """
    # Define the columns to keep (exclude 'label' for features)
    feature_columns = [col for col in df_train.columns if col != 'label']
    
    # Separate features and label for train and test datasets
    X_train = df_train[feature_columns].copy()
    y_train = df_train['label'].copy()
    
    X_test = df_test[feature_columns].copy()
    y_test = df_test['label'].copy()
    
    # Verify that 'timestamp' is not in the feature sets
    assert 'timestamp' not in X_train.columns, "'timestamp' should not be in X_train"
    assert 'timestamp' not in X_test.columns, "'timestamp' should not be in X_test"
    
    # Verify that classes match between train and test sets
    assert sorted(pd.Series(y_train).unique()) == sorted(pd.Series(y_test).unique()), \
        "Train and test sets must contain the same classes"
    
    return X_train, y_train, X_test, y_test


def get_fitted_scaler(X_train, scaler_type='standard'):
    """
    Identify features to scale and return a fitted scaler.
    Excludes binary features and specific technical indicators from scaling.
    
    Args:
        X_train (pd.DataFrame): Training features
        scaler_type (str): Type of scaler to use. Options:
            'standard', 'minmax', 'maxabs', 'robust', 
            'quantile_uniform', 'quantile_normal', 'power'
    
    Returns:
        tuple: (fitted scaler, list of features to scale)
    """
    from sklearn.preprocessing import (
        StandardScaler, MinMaxScaler, MaxAbsScaler, 
        RobustScaler, QuantileTransformer, PowerTransformer
    )
    import numpy as np
    
    # Initialize the appropriate scaler
    scalers = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'maxabs': MaxAbsScaler(),
        'robust': RobustScaler(),
        'quantile_uniform': QuantileTransformer(output_distribution='uniform'),
        'quantile_normal': QuantileTransformer(output_distribution='normal'),
        'power': PowerTransformer(method='yeo-johnson')
    }
    
    scaler = scalers.get(scaler_type.lower(), StandardScaler())
    
    # Technical indicators to exclude from scaling
    technical_indicators = [
        'cci', 'roc', 'rsi', 'ppo', 'stoch',
        'close_raw_percentile', 'close_percentile'
    ]
    
    # Identify binary features (columns with 2 unique values of 0 and 1)
    binary_features = [
        col for col in X_train.columns 
        if (X_train[col].nunique() == 2) and (X_train[col].isin([0, 1]).all())
    ]
    
    # Automatically identify numeric features
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    
    # Features to scale (numeric features excluding binary features and technical indicators)
    features_to_scale = []
    for col in numeric_features:
        if col not in binary_features:
            # Check if column contains any technical indicator names (case insensitive)
            should_exclude = any(
                indicator.lower() in col.lower()
                for indicator in technical_indicators
            )
            if not should_exclude:
                features_to_scale.append(col)
    
    # Identify integer columns and create a copy of the data for fitting
    X_train_temp = X_train.copy()
    int_cols_to_convert = X_train_temp[features_to_scale].select_dtypes(include=['int']).columns
    X_train_temp[int_cols_to_convert] = X_train_temp[int_cols_to_convert].astype('float32')
    
    # Fit the scaler
    scaler.fit(X_train_temp[features_to_scale])
    
    return scaler, features_to_scale

def apply_scaling(df, scaler, features_to_scale):
    """
    Apply scaling to a dataframe using a fitted scaler.
    
    Args:
        df (pd.DataFrame): DataFrame to scale
        scaler: Fitted sklearn scaler object
        features_to_scale (list): List of column names to scale
    
    Returns:
        pd.DataFrame: Scaled DataFrame
    """
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Convert integer columns to float32 before scaling
    int_cols = df[features_to_scale].select_dtypes(include=['int']).columns
    df[int_cols] = df[int_cols].astype('float32')
    
    # Apply scaling
    df.loc[:, features_to_scale] = scaler.transform(df[features_to_scale]).astype('float32')
    
    return df

def round_numeric_features(df, decimals=2, exclude_columns=None):
    """
    Round all numeric columns in a dataframe to specified decimal places, with optional column exclusions.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        decimals (int): Number of decimal places to round to (default: 2)
        exclude_columns (list[str], optional): List of column names to exclude from rounding
    
    Returns:
        pd.DataFrame: DataFrame with rounded numeric columns
    
    Raises:
        ValueError: If any column in exclude_columns is not present in the DataFrame
    """
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Convert exclude_columns to empty list if None
    exclude_columns = exclude_columns or []
    
    # Check if excluded columns exist in DataFrame
    missing_columns = [col for col in exclude_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Columns not found in DataFrame: {missing_columns}")
    
    # Identify numeric columns, excluding specified columns
    numeric_columns = df.select_dtypes(include='number').columns
    columns_to_round = [col for col in numeric_columns if col not in exclude_columns]
    
    # Round numeric columns
    df.loc[:, columns_to_round] = df[columns_to_round].round(decimals)
    
    return df

def calculate_weights(y_train):
    """
    Calculate class weights and sample weights for imbalanced classification.
    
    Args:
        y_train (pd.Series): Training labels
    
    Returns:
        tuple: (class_weights dict, sample_weights array)
    """
    from collections import Counter
    
    # Calculate the frequency of each class in the training set
    class_counts = Counter(y_train)
    total_samples = len(y_train)
    num_classes = y_train.nunique()
    
    # Calculate class weights
    class_weights = {
        cls: total_samples / (num_classes * count) 
        for cls, count in class_counts.items()
    }
    
    # Map class weights to each sample in y_train
    sample_weights = y_train.map(class_weights).astype(float)
    
    return class_weights, sample_weights




def prepare_test_data(df_features_master_for_predictions: pd.DataFrame, 
                     test_data_begin_timestamp: pd.Timestamp) -> pd.DataFrame:
    """
    Prepares test data by handling timezone conversions and filtering based on a start timestamp.
    
    Args:
        df_features_master_for_predictions (pd.DataFrame): Master DataFrame with features for predictions
        test_data_begin_timestamp (pd.Timestamp): Starting timestamp for test data
    
    Returns:
        pd.DataFrame: Filtered DataFrame containing only data from test_data_begin_timestamp onwards
    """
    # Create a copy to avoid modifying the original DataFrame
    df = df_features_master_for_predictions.copy()
    
    # Handle timezone issues in the prepared predictions DataFrame
    if df.index.tz is not None:
        df.index = (
            df.index
                .tz_convert('US/Eastern')
                .tz_localize(None)
        )
    
    # Convert test_data_begin_timestamp if it has timezone information
    if isinstance(test_data_begin_timestamp, pd.Timestamp) and test_data_begin_timestamp.tz is not None:
        test_data_begin_timestamp = test_data_begin_timestamp.tz_convert('US/Eastern').tz_localize(None)
    
    # Filter data based on the test start timestamp
    df_features_master_for_predictions_test = df[df.index >= test_data_begin_timestamp]
    
    return df_features_master_for_predictions_test


import pandas as pd
from typing import Dict, Tuple

def split_into_deciles(df_features_master_for_predictions_test: pd.DataFrame, 
                      verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, 
                                                   pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                                   pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                                   pd.DataFrame, pd.DataFrame]:
    """
    Splits the test dataset into 10 equal-sized chunks (deciles) by timestamp.
    
    Args:
        df_features_master_for_predictions_test (pd.DataFrame): Input DataFrame to be split
        verbose (bool): Whether to print information about each chunk
    
    Returns:
        Tuple[pd.DataFrame, ...]: Returns 11 DataFrames in the following order:
            - decile_1 (0-10%)
            - decile_2 (10-20%)
            - decile_3 (20-30%)
            - decile_4 (30-40%)
            - decile_5 (40-50%)
            - decile_6 (50-60%)
            - decile_7 (60-70%)
            - decile_8 (70-80%)
            - decile_9 (80-90%)
            - decile_10 (90-100%)
            - full_test_set (complete dataset)
    """
    # First sort by timestamp
    df = df_features_master_for_predictions_test.sort_index(ascending=True)
    
    # Calculate the size of each 10% chunk
    total_rows = len(df)
    chunk_size = total_rows // 10
    
    # Create a dictionary to store each chunk with descriptive names
    decile_chunks: Dict[str, pd.DataFrame] = {}
    
    for decile in range(10):
        # Calculate start and end indices for each chunk
        start_index = decile * chunk_size
        
        # For the last chunk, make sure to include any remaining rows
        if decile == 9:
            end_index = total_rows
        else:
            end_index = start_index + chunk_size
        
        # Create the chunk and store it in dictionary with descriptive name
        chunk_name = f"decile_{decile+1}"  # decile_1 for 0-10%, decile_2 for 10-20%, etc.
        decile_chunks[chunk_name] = df.iloc[start_index:end_index]
        
        # Print information about each chunk if verbose is True
        if verbose:
            start_percent = decile * 10
            end_percent = (decile + 1) * 10
            print(f"Chunk {decile+1} ({start_percent}%-{end_percent}%): {len(decile_chunks[chunk_name])} rows")
    
    # Create a full copy of the test set
    full_test_set = df.copy()
    
    # Return all deciles and the full test set
    return (
        decile_chunks['decile_1'],   # 0-10%
        decile_chunks['decile_2'],   # 10-20%
        decile_chunks['decile_3'],   # 20-30%
        decile_chunks['decile_4'],   # 30-40%
        decile_chunks['decile_5'],   # 40-50%
        decile_chunks['decile_6'],   # 50-60%
        decile_chunks['decile_7'],   # 60-70%
        decile_chunks['decile_8'],   # 70-80%
        decile_chunks['decile_9'],   # 80-90%
        decile_chunks['decile_10'],  # 90-100%
        full_test_set               # Complete dataset
    )



