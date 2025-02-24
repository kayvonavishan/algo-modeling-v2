import psutil
import os


def cleanup_trial_variables(additional_vars=None):
    """
    Cleanup specified variables from local namespace and run garbage collection.
    
    Args:
        additional_vars (list, optional): Additional variable names to cleanup
    """
    default_vars = [
        'df_features_master',
        'df_predictions',
        'df_features_master_for_predictions',
        'df_features_master_for_predictions_prepared',
        'train_data',
        'test_data',
        # Decile-specific DataFrames
        'df_predictions_decile_1', 'df_predictions_decile_2', 'df_predictions_decile_3',
        'df_predictions_decile_4', 'df_predictions_decile_5', 'df_predictions_decile_6',
        'df_predictions_decile_7', 'df_predictions_decile_8', 'df_predictions_decile_9',
        'df_predictions_decile_10',
        'df_with_signals_decile_1', 'df_with_signals_decile_2', 'df_with_signals_decile_3',
        'df_with_signals_decile_4', 'df_with_signals_decile_5', 'df_with_signals_decile_6',
        'df_with_signals_decile_7', 'df_with_signals_decile_8', 'df_with_signals_decile_9',
        'df_with_signals_decile_10'
    ]
    
    # Combine default and additional variables if any
    vars_to_cleanup = default_vars + (additional_vars or [])
    
    # Cleanup variables
    for var in vars_to_cleanup:
        if var in locals():
            del locals()[var]
    
    gc.collect()


######################
# LIMIT CPU USAGE!!
######################

def cleanup_dataframes():
    """
    Clean up ticker_df_adjusted and df_features DataFrames if they exist in local scope,
    then run garbage collection.
    """
    import gc
    
    variables_to_cleanup = [
        'ticker_df_adjusted',
        'df_features'
    ]
    
    for var in variables_to_cleanup:
        if var in locals():
            del locals()[var]
    
    gc.collect()

def cleanup_memory():
    """
    Clean up memory by deleting specific DataFrames and running garbage collection.
    This function checks for and removes commonly used large DataFrames from the
    global namespace, then runs Python's garbage collector.
    """
    import gc
    
    variables_to_cleanup = [
        'df_features_master_for_model',
        'df_predictions',
        'df_features_master_for_predictions_prepared',
        'train_data',
        'test_data'
    ]
    
    for var in variables_to_cleanup:
        if var in globals():
            del globals()[var]
    
    gc.collect()

def limit_cpu_cores(usage_fraction=0.9):
    """
    Limits the current process to use a fraction of available CPU cores.

    :param usage_fraction: Fraction of total CPU cores to utilize (e.g., 0.9 for 90%)
    """
    process = psutil.Process(os.getpid())

    # Set process priority to BELOW_NORMAL
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    print(f"Process priority set to: {process.nice()}")

    # Determine the number of cores to use based on the desired usage fraction
    total_cores = psutil.cpu_count(logical=True)
    cores_to_use = max(1, int(total_cores * usage_fraction))
    selected_cores = list(range(cores_to_use))

    # Set processor affinity to limit the process to the selected cores
    process.cpu_affinity(selected_cores)
    print(f"Process is limited to cores: {selected_cores}")

######################
# MONITOR MEMORY USAGE 
######################
import psutil
import gc

def monitor_memory():
    """
    Monitor current memory usage and return memory metrics.
    
    Returns:
    tuple: (memory_mb, memory_percent)
        - memory_mb: Memory usage in megabytes
        - memory_percent: Memory usage as a percentage
    """
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
    memory_percent = process.memory_percent()
    
    # Print memory info for debugging
    print(f"Memory usage: {memory_mb:.2f} MB ({memory_percent:.2f}%)")
    
    return memory_mb, memory_percent


def set_working_directory(directory_path):
    os.chdir(directory_path)
    print(f"Working directory set to: {os.getcwd()}")


def optimize_dataframe_fast(df):
    # Select float and integer columns
    float_cols = df.select_dtypes(include=['float64', 'float']).columns
    int_cols = df.select_dtypes(include=['int64', 'int']).columns

    # Downcast integer columns to int32
    df[int_cols] = df[int_cols].astype('int32')

    # Downcast float columns to float32
    df[float_cols] = df[float_cols].astype('float32')
    
    return df
