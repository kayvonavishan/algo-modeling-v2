################################
# IMPORT FEATURE FUNCTIONS
#################################
from basic_feature_functions import *

######################
# IMPORT RNN MODELS AND SCALERS 
######################
from RNN_load import * 

####################################
# IMPORT FUNCTIONS  FOR RNN FEATURES 
###################################
from RNN_features import *

from tqdm import tqdm

def calculate_precomputed_features(data_dict, optuna_ticker_list=None):
    """
    Calculate precomputed features for each ticker dataframe in data_dict.

    Parameters:
        data_dict (dict): Dictionary mapping ticker symbols to their respective DataFrames.
        optuna_ticker_list (list, optional): List of symbols to process. Defaults to ['TSLA'] if None.

    Returns:
        dict: Updated data_dict with the computed features for each symbol.
    """
    if optuna_ticker_list is None:
        optuna_ticker_list = ['TQQQ']
    
    for symbol in tqdm(optuna_ticker_list, desc="Processing Symbols"):
        # Load data and create a working copy
        ticker_df_adjusted = data_dict[symbol].copy()

        # Define raw price columns
        ticker_df_adjusted['close_raw'] = ticker_df_adjusted['close']
        ticker_df_adjusted['high_raw'] = ticker_df_adjusted['high']
        ticker_df_adjusted['low_raw'] = ticker_df_adjusted['low']
        ticker_df_adjusted['open_raw'] = ticker_df_adjusted['open']

        ###############################################################
        # CALCULATES SLOPE FOR DIFFERENT PERIODS AND APPLY HMA (RAW)
        ###############################################################
        slope_periods = [30, 20, 10, 2]  # Ordered from largest to smallest
        hma_period = 10
        ticker_df_adjusted = add_raw_slope_features(ticker_df_adjusted, 
                                                    slope_periods=slope_periods,
                                                    hma_period=hma_period)
    
        ######################
        # FEATURES FOR RAW PRICE
        ######################
        custom_periods = [350, 300, 250, 200, 150, 100, 75, 50, 40, 30, 25, 20, 15, 10, 5]
        ticker_df_adjusted = add_long_term_hma_features_raw(ticker_df_adjusted, 
                                                            periods=custom_periods,
                                                            calculate_slopes=True,
                                                            slope_period=2)
        
        # What percent of the time has HMA long-term slope been positive?
        ticker_df_adjusted = add_raw_slope_trend_percentages(ticker_df_adjusted)

        # Percent difference metrics
        ticker_df_adjusted = add_raw_distance_metrics(ticker_df_adjusted)
        
        # Price position relative to moving averages
        custom_periods = [150, 75, 40, 30, 25, 20, 15, 10, 5]
        ticker_df_adjusted = add_price_position_indicators_raw(ticker_df_adjusted, periods=custom_periods)

        # Standard deviation features
        custom_periods = [40, 30, 20]
        ticker_df_adjusted = add_std_deviation_features_raw(ticker_df_adjusted, periods=custom_periods)

        # HMA distance features and cross-period features
        ticker_df_adjusted = calculate_hma_distance_features_raw(ticker_df_adjusted)
        ticker_df_adjusted = calculate_hma_cross_period_features_raw(ticker_df_adjusted)

        ######################
        # CCI FEATURES (RAW)
        ######################
        ticker_df_adjusted = calculate_cci_features_raw(ticker_df_adjusted, calculate_hma)

        ######################
        # ROC/RSI/PPO/STOCH FEATURES (RAW)
        ######################
        ticker_df_adjusted = calculate_technical_indicators_raw(ticker_df_adjusted)

        ######################
        # FEATURE: CANDLES 
        ######################
        ticker_df_adjusted = add_candle_pattern_features(ticker_df_adjusted,
                                                         open_col='open_raw',
                                                         close_col='close_raw')

        ######################
        # Price Percentiles 
        ######################
        custom_windows = [200, 100, 50, 20, 10, 5]
        ticker_df_adjusted = add_rolling_percentile_features(ticker_df_adjusted,
                                                             windows=custom_windows,
                                                             column='close_raw')

        ######################
        # Trailing z-score 
        ######################
        custom_windows = [500, 200, 100, 20]
        ticker_df_adjusted = add_rolling_zscore_features(ticker_df_adjusted,
                                                         windows=custom_windows,
                                                         column='close_raw')

        ######################
        # RNN PREDICTIONS FEATURES 
        ######################
        custom_periods = [100, 60, 45, 35, 25, 15, 10, 5, 4]
        custom_slope_periods = [2, 5]
        ticker_df_adjusted, feature_names = features_for_RNN_models(ticker_df_adjusted, 
                                                periods=custom_periods,
                                                slope_periods=custom_slope_periods)

        # HMA 25 Binary predictions
        ticker_df_adjusted = predict_hma25_binary_t3(ticker_df_adjusted, model_hma25_binary_t3)
        ticker_df_adjusted = predict_hma25_binary_t4(ticker_df_adjusted, model_hma25_binary_t4, scaler_hma25_binary_t4)
        ticker_df_adjusted = predict_hma25_binary_t5(ticker_df_adjusted, model_hma25_binary_t5, scaler_hma25_binary_t5)
        
        # CCI HMA 14 predictions
        ticker_df_adjusted = predict_close_raw_cci_hma_14_t1(ticker_df_adjusted, model_close_raw_cci_hma_14_t1, scaler_close_raw_cci_hma_14_t1)
        ticker_df_adjusted = predict_close_raw_cci_hma_14_t2(ticker_df_adjusted, model_close_raw_cci_hma_14_t2, scaler_close_raw_cci_hma_14_t2)
        ticker_df_adjusted = predict_close_raw_cci_hma_14_t3(ticker_df_adjusted, model_close_raw_cci_hma_14_t3, scaler_close_raw_cci_hma_14_t3)
        
        # CCI HMA 18/22 predictions
        ticker_df_adjusted = predict_close_raw_cci_hma_18_t3(ticker_df_adjusted, model_close_raw_cci_hma_18_t3, scaler_close_raw_cci_hma_18_t3)
        ticker_df_adjusted = predict_close_raw_cci_hma_22_t3(ticker_df_adjusted, model_close_raw_cci_hma_22_t3, scaler_close_raw_cci_hma_22_t3)
        
        # HMA 15 predictions
        ticker_df_adjusted = predict_hma15_t1(ticker_df_adjusted, model_hma15_t1, scaler_hma15_t1)
        ticker_df_adjusted = predict_hma15_t2(ticker_df_adjusted, model_hma15_t2, scaler_hma15_t2)
        ticker_df_adjusted = predict_hma15_t3(ticker_df_adjusted, model_hma15_t3, scaler_hma15_t3)
        
        # HMA 25 predictions
        ticker_df_adjusted = predict_hma25_t1(ticker_df_adjusted, model_hma25_t1, scaler_hma25_t1)
        ticker_df_adjusted = predict_hma25_t2(ticker_df_adjusted, model_hma25_t2, scaler_hma25_t2)
        ticker_df_adjusted = predict_hma25_t3(ticker_df_adjusted, model_hma25_t3, scaler_hma25_t3)
        ticker_df_adjusted = predict_hma25_t4(ticker_df_adjusted, model_hma25_t4, scaler_hma25_t4)
        ticker_df_adjusted = predict_hma25_t5(ticker_df_adjusted, model_hma25_t5, scaler_hma25_t5)
        
        # HMA 35 predictions
        ticker_df_adjusted = predict_hma35_t1(ticker_df_adjusted, model_hma35_t1, scaler_hma35_t1)
        ticker_df_adjusted = predict_hma35_t2(ticker_df_adjusted, model_hma35_t2, scaler_hma35_t2)
        ticker_df_adjusted = predict_hma35_t3(ticker_df_adjusted, model_hma35_t3, scaler_hma35_t3)
        ticker_df_adjusted = predict_hma35_t4(ticker_df_adjusted, model_hma35_t4, scaler_hma35_t4)
        
        # HMA 45 predictions
        ticker_df_adjusted = predict_hma45_t1(ticker_df_adjusted, model_hma45_t1, scaler_hma45_t1)
        ticker_df_adjusted = predict_hma45_t2(ticker_df_adjusted, model_hma45_t2, scaler_hma45_t2)
        ticker_df_adjusted = predict_hma45_t3(ticker_df_adjusted, model_hma45_t3, scaler_hma45_t3)
        ticker_df_adjusted = predict_hma45_t4(ticker_df_adjusted, model_hma45_t4, scaler_hma45_t4)

        # squeeze momentum slope
        ticker_df_adjusted = predict_sqz_momentum_slope_30_15min_t1(ticker_df_adjusted, model_sqz_momentum_slope_30_15min_t1, scaler_sqz_momentum_30_15min_t1)
        ticker_df_adjusted = predict_sqz_momentum_slope_30_15min_t2(ticker_df_adjusted, model_sqz_momentum_slope_30_15min_t2, scaler_sqz_momentum_30_15min_t2)
        ticker_df_adjusted = predict_sqz_momentum_slope_30_15min_t3(ticker_df_adjusted, model_sqz_momentum_slope_30_15min_t3, scaler_sqz_momentum_30_15min_t3)
        ticker_df_adjusted = predict_sqz_momentum_slope_30_15min_t4(ticker_df_adjusted, model_sqz_momentum_slope_30_15min_t4, scaler_sqz_momentum_30_15min_t4)
        ticker_df_adjusted = predict_sqz_momentum_slope_30_15min_t5(ticker_df_adjusted, model_sqz_momentum_slope_30_15min_t5, scaler_sqz_momentum_30_15min_t5)
        ticker_df_adjusted = predict_sqz_momentum_slope_30_15min_t6(ticker_df_adjusted, model_sqz_momentum_slope_30_15min_t6, scaler_sqz_momentum_30_15min_t6)
        ticker_df_adjusted = predict_sqz_momentum_slope_30_15min_t7(ticker_df_adjusted, model_sqz_momentum_slope_30_15min_t7, scaler_sqz_momentum_30_15min_t7)
        ticker_df_adjusted = predict_sqz_momentum_slope_30_15min_t8(ticker_df_adjusted, model_sqz_momentum_slope_30_15min_t8, scaler_sqz_momentum_30_15min_t8)

        # squeeze momentum
        ticker_df_adjusted = predict_sqz_momentum_30_15min_t1(ticker_df_adjusted, model_sqz_momentum_30_15min_t1, scaler_sqz_momentum_30_15min_t1)
        ticker_df_adjusted = predict_sqz_momentum_30_15min_t2(ticker_df_adjusted, model_sqz_momentum_30_15min_t2, scaler_sqz_momentum_30_15min_t2)
        ticker_df_adjusted = predict_sqz_momentum_30_15min_t3(ticker_df_adjusted, model_sqz_momentum_30_15min_t3, scaler_sqz_momentum_30_15min_t3)
        ticker_df_adjusted = predict_sqz_momentum_30_15min_t4(ticker_df_adjusted, model_sqz_momentum_30_15min_t4, scaler_sqz_momentum_30_15min_t4)
        ticker_df_adjusted = predict_sqz_momentum_30_15min_t5(ticker_df_adjusted, model_sqz_momentum_30_15min_t5, scaler_sqz_momentum_30_15min_t5)

        # RSI-7 predictions
        ticker_df_adjusted = predict_RSI_7_hma_15_15min_t1(ticker_df_adjusted, model_RSI_7_hma_15_15min_t1, scaler_RSI_7_hma_15_15min_t1)
        ticker_df_adjusted = predict_RSI_7_hma_15_15min_t2(ticker_df_adjusted, model_RSI_7_hma_15_15min_t2, scaler_RSI_7_hma_15_15min_t2)
        ticker_df_adjusted = predict_RSI_7_hma_15_15min_t3(ticker_df_adjusted, model_RSI_7_hma_15_15min_t3, scaler_RSI_7_hma_15_15min_t3)
        ticker_df_adjusted = predict_RSI_7_hma_15_15min_t4(ticker_df_adjusted, model_RSI_7_hma_15_15min_t4, scaler_RSI_7_hma_15_15min_t4)

        # RSI-14 predictions
        ticker_df_adjusted = predict_RSI_14_hma_15_15min_t1(ticker_df_adjusted, model_RSI_14_hma_15_15min_t1, scaler_RSI_14_hma_15_15min_t1)
        ticker_df_adjusted = predict_RSI_14_hma_15_15min_t2(ticker_df_adjusted, model_RSI_14_hma_15_15min_t2, scaler_RSI_14_hma_15_15min_t2)
        ticker_df_adjusted = predict_RSI_14_hma_15_15min_t3(ticker_df_adjusted, model_RSI_14_hma_15_15min_t3, scaler_RSI_14_hma_15_15min_t3)
        ticker_df_adjusted = predict_RSI_14_hma_15_15min_t4(ticker_df_adjusted, model_RSI_14_hma_15_15min_t4, scaler_RSI_14_hma_15_15min_t4)
        
        # RSI-28 predictions
        ticker_df_adjusted = predict_RSI_28_hma_15_15min_t1(ticker_df_adjusted, model_RSI_28_hma_15_15min_t1, scaler_RSI_28_hma_15_15min_t1)
        ticker_df_adjusted = predict_RSI_28_hma_15_15min_t2(ticker_df_adjusted, model_RSI_28_hma_15_15min_t2, scaler_RSI_28_hma_15_15min_t2)
        ticker_df_adjusted = predict_RSI_28_hma_15_15min_t3(ticker_df_adjusted, model_RSI_28_hma_15_15min_t3, scaler_RSI_28_hma_15_15min_t3)
        ticker_df_adjusted = predict_RSI_28_hma_15_15min_t4(ticker_df_adjusted, model_RSI_28_hma_15_15min_t4, scaler_RSI_28_hma_15_15min_t4)

        # HMA 15 predictions (z-score)
        ticker_df_adjusted = predict_hma15_5_zscore_t1(ticker_df_adjusted, model_hma15_5_zscore_t1, scaler_hma15_5_zscore_t1)
        ticker_df_adjusted = predict_hma15_5_zscore_t2(ticker_df_adjusted, model_hma15_5_zscore_t2, scaler_hma15_5_zscore_t2)
        ticker_df_adjusted = predict_hma15_5_zscore_t3(ticker_df_adjusted, model_hma15_5_zscore_t3, scaler_hma15_5_zscore_t3)
        ticker_df_adjusted = predict_hma15_5_zscore_t4(ticker_df_adjusted, model_hma15_5_zscore_t4, scaler_hma15_5_zscore_t4)

        # HMA 25 predictions (z-score)
        ticker_df_adjusted = predict_hma25_5_zscore_t1(ticker_df_adjusted, model_hma25_5_zscore_t1, scaler_hma25_5_zscore_t1)
        ticker_df_adjusted = predict_hma25_5_zscore_t2(ticker_df_adjusted, model_hma25_5_zscore_t2, scaler_hma25_5_zscore_t2)
        ticker_df_adjusted = predict_hma25_5_zscore_t3(ticker_df_adjusted, model_hma25_5_zscore_t3, scaler_hma25_5_zscore_t3)
        ticker_df_adjusted = predict_hma25_5_zscore_t4(ticker_df_adjusted, model_hma25_5_zscore_t4, scaler_hma25_5_zscore_t4)

        # HMA 35 predictions (z-score)
        ticker_df_adjusted = predict_hma35_5_zscore_t1(ticker_df_adjusted, model_hma35_5_zscore_t1, scaler_hma35_5_zscore_t1)
        ticker_df_adjusted = predict_hma35_5_zscore_t2(ticker_df_adjusted, model_hma35_5_zscore_t2, scaler_hma35_5_zscore_t2)
        ticker_df_adjusted = predict_hma35_5_zscore_t3(ticker_df_adjusted, model_hma35_5_zscore_t3, scaler_hma35_5_zscore_t3)
        ticker_df_adjusted = predict_hma35_5_zscore_t4(ticker_df_adjusted, model_hma35_5_zscore_t4, scaler_hma35_5_zscore_t4)

        # HMA 45 predictions (z-score)
        ticker_df_adjusted = predict_hma45_5_zscore_t1(ticker_df_adjusted, model_hma45_5_zscore_t1, scaler_hma45_5_zscore_t1)
        ticker_df_adjusted = predict_hma45_5_zscore_t2(ticker_df_adjusted, model_hma45_5_zscore_t2, scaler_hma45_5_zscore_t2)
        ticker_df_adjusted = predict_hma45_5_zscore_t3(ticker_df_adjusted, model_hma45_5_zscore_t3, scaler_hma45_5_zscore_t3)
        ticker_df_adjusted = predict_hma45_5_zscore_t4(ticker_df_adjusted, model_hma45_5_zscore_t4, scaler_hma45_5_zscore_t4)
        
        # VWAP 
        ticker_df_adjusted = predict_vwap_zscore_10_15min_t6(ticker_df_adjusted, model_vwap_zscore_10_15min_t6, scaler_vwap_zscore_10_15min_t6)
        ticker_df_adjusted = predict_vwap_zscore_10_15min_t7(ticker_df_adjusted, model_vwap_zscore_10_15min_t7, scaler_vwap_zscore_10_15min_t7)
        ticker_df_adjusted = predict_vwap_zscore_10_15min_t8(ticker_df_adjusted, model_vwap_zscore_10_15min_t8, scaler_vwap_zscore_10_15min_t8)
        ticker_df_adjusted = predict_vwap_zscore_10_15min_t9(ticker_df_adjusted, model_vwap_zscore_10_15min_t9, scaler_vwap_zscore_10_15min_t9)

        # composite 
        ticker_df_adjusted = create_logical_group_composites(ticker_df_adjusted, threshold=2.0)

        # percentiles 
        ticker_df_adjusted = predict_close_raw_percentile_20_15min_t1(ticker_df_adjusted, model_close_raw_percentile_20_15min_t1, scaler_close_raw_percentile_20_15min_t1)
        ticker_df_adjusted = predict_close_raw_percentile_20_15min_t2(ticker_df_adjusted, model_close_raw_percentile_20_15min_t2, scaler_close_raw_percentile_20_15min_t2)
        ticker_df_adjusted = predict_close_raw_percentile_50_15min_t1(ticker_df_adjusted, model_close_raw_percentile_50_15min_t1, scaler_close_raw_percentile_50_15min_t1)
        ticker_df_adjusted = predict_close_raw_percentile_50_15min_t2(ticker_df_adjusted, model_close_raw_percentile_50_15min_t2, scaler_close_raw_percentile_50_15min_t2)
        ticker_df_adjusted = predict_close_raw_percentile_50_15min_t3(ticker_df_adjusted, model_close_raw_percentile_50_15min_t3, scaler_close_raw_percentile_50_15min_t3)

        # Trailing return z-score EMA predictions
        ticker_df_adjusted = predict_trailing_return_20_zscore_ema_15min_t1(ticker_df_adjusted, model_trailing_return_20_zscore_ema_15min_t1, scaler_trailing_return_20_zscore_ema_15min_t1)
        ticker_df_adjusted = predict_trailing_return_20_zscore_ema_15min_t2(ticker_df_adjusted, model_trailing_return_20_zscore_ema_15min_t2, scaler_trailing_return_20_zscore_ema_15min_t2)
        ticker_df_adjusted = predict_trailing_return_20_zscore_ema_15min_t3(ticker_df_adjusted, model_trailing_return_20_zscore_ema_15min_t3, scaler_trailing_return_20_zscore_ema_15min_t3)
        ticker_df_adjusted = predict_trailing_return_20_zscore_ema_15min_t4(ticker_df_adjusted, model_trailing_return_20_zscore_ema_15min_t4, scaler_trailing_return_20_zscore_ema_15min_t4)
        ticker_df_adjusted = predict_trailing_return_20_zscore_ema_15min_t5(ticker_df_adjusted, model_trailing_return_20_zscore_ema_15min_t5, scaler_trailing_return_20_zscore_ema_15min_t5)

        # remove underlying RNN features 
        ticker_df_adjusted = remove_RNN_features(ticker_df_adjusted, feature_names)
        
        # Update the dictionary and clean up
        data_dict[symbol] = ticker_df_adjusted.copy()
        del ticker_df_adjusted

    return data_dict
