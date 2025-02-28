from RNN_stuff import load_model
from RNN_stuff import load_scaler


######################
# RETURNS
######################

model_trailing_return_20_zscore_ema_15min_t1 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/trailing_return_20_zscore_ema_15min_t1.pth',
    input_size=7,
    num_features=8
)
scaler_trailing_return_20_zscore_ema_15min_t1 = None

model_trailing_return_20_zscore_ema_15min_t2 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/trailing_return_20_zscore_ema_15min_t2.pth',
    input_size=7,
    num_features=8
)
scaler_trailing_return_20_zscore_ema_15min_t2 = None

model_trailing_return_20_zscore_ema_15min_t3 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/trailing_return_20_zscore_ema_15min_t3.pth',
    input_size=7,
    num_features=8
)
scaler_trailing_return_20_zscore_ema_15min_t3 = None

model_trailing_return_20_zscore_ema_15min_t4 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/trailing_return_20_zscore_ema_15min_t4.pth',
    input_size=7,
    num_features=8
)
scaler_trailing_return_20_zscore_ema_15min_t4 = None

model_trailing_return_20_zscore_ema_15min_t5 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/trailing_return_20_zscore_ema_15min_t5.pth',
    input_size=7,
    num_features=8
)
scaler_trailing_return_20_zscore_ema_15min_t5 = None


######################
# PRICE PERCENTILES 
######################

model_close_raw_percentile_20_15min_t1 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_raw_percentile_20_15min_t1.pth',
    input_size=7,
    num_features=6
)
scaler_close_raw_percentile_20_15min_t1 = None

model_close_raw_percentile_20_15min_t2 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_raw_percentile_20_15min_t2.pth',
    input_size=7,
    num_features=6
)
scaler_close_raw_percentile_20_15min_t2 = None

model_close_raw_percentile_50_15min_t1 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_raw_percentile_50_15min_t1.pth',
    input_size=7,
    num_features=6
)
scaler_close_raw_percentile_50_15min_t1 = None

model_close_raw_percentile_50_15min_t2 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_raw_percentile_50_15min_t2.pth',
    input_size=7,
    num_features=6
)
scaler_close_raw_percentile_50_15min_t2 = None

model_close_raw_percentile_50_15min_t3 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_raw_percentile_50_15min_t3.pth',
    input_size=7,
    num_features=6
)
scaler_close_raw_percentile_50_15min_t3 = None


######################
# LOAD RNN MODELS (vwap)
######################


# Model t6
model_vwap_zscore_10_15min_t6 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/vwap_zscore_10_15min_t6.pth',
    input_size=7,
    num_features=8
)
scaler_vwap_zscore_10_15min_t6 = None

# Model t7
model_vwap_zscore_10_15min_t7 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/vwap_zscore_10_15min_t7.pth',
    input_size=7,
    num_features=8
)
scaler_vwap_zscore_10_15min_t7 = None

# Model t8
model_vwap_zscore_10_15min_t8 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/vwap_zscore_10_15min_t8.pth',
    input_size=7,
    num_features=8
)
scaler_vwap_zscore_10_15min_t8 = None

# Model t9
model_vwap_zscore_10_15min_t9 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/vwap_zscore_10_15min_t9.pth',
    input_size=7,
    num_features=8
)
scaler_vwap_zscore_10_15min_t9 = None




######################
# LOAD RNN MODELS (new)
######################

# sqz_momentum_slope_30
model_sqz_momentum_slope_30_15min_t1 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/sqz_momentum_slope_30_15min_t2.pth',
    input_size=7,
    num_features=6
)
scaler_sqz_momentum_30_15min_t1 = None

# sqz_momentum_slope_30
model_sqz_momentum_slope_30_15min_t2 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/sqz_momentum_slope_30_15min_t2.pth',
    input_size=7,
    num_features=6
)
scaler_sqz_momentum_30_15min_t2 = None

# sqz_momentum_slope_30
model_sqz_momentum_slope_30_15min_t3 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/sqz_momentum_slope_30_15min_t3.pth',
    input_size=7,
    num_features=6
)
scaler_sqz_momentum_30_15min_t3 = None

# sqz_momentum_slope_30
model_sqz_momentum_slope_30_15min_t4 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/sqz_momentum_slope_30_15min_t4.pth',
    input_size=7,
    num_features=6
)
scaler_sqz_momentum_30_15min_t4 = None

# sqz_momentum_slope_30
model_sqz_momentum_slope_30_15min_t5 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/sqz_momentum_slope_30_15min_t5.pth',
    input_size=7,
    num_features=6
)
scaler_sqz_momentum_30_15min_t5 = None

# sqz_momentum_slope_30
model_sqz_momentum_slope_30_15min_t6 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/sqz_momentum_slope_30_15min_t6.pth',
    input_size=7,
    num_features=6
)
scaler_sqz_momentum_30_15min_t6 = None

# sqz_momentum_slope_30
model_sqz_momentum_slope_30_15min_t7 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/sqz_momentum_slope_30_15min_t7.pth',
    input_size=7,
    num_features=6
)
scaler_sqz_momentum_30_15min_t7 = None

# sqz_momentum_slope_30
model_sqz_momentum_slope_30_15min_t8 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/sqz_momentum_slope_30_15min_t8.pth',
    input_size=7,
    num_features=6
)
scaler_sqz_momentum_30_15min_t8 = None






# sqz_momentum_30
model_sqz_momentum_30_15min_t1 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/sqz_momentum_30_15min_t1.pth',
    input_size=7,
    num_features=6
)
scaler_sqz_momentum_30_15min_t1 = None

# sqz_momentum_slope_30
model_sqz_momentum_30_15min_t2 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/sqz_momentum_30_15min_t2.pth',
    input_size=7,
    num_features=6
)
scaler_sqz_momentum_30_15min_t2 = None

# sqz_momentum_slope_30
model_sqz_momentum_30_15min_t3 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/sqz_momentum_30_15min_t3.pth',
    input_size=7,
    num_features=6
)
scaler_sqz_momentum_30_15min_t3 = None

# sqz_momentum_slope_30
model_sqz_momentum_30_15min_t4 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/sqz_momentum_30_15min_t4.pth',
    input_size=7,
    num_features=6
)
scaler_sqz_momentum_30_15min_t4 = None

# sqz_momentum_slope_30
model_sqz_momentum_30_15min_t5 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/sqz_momentum_30_15min_t5.pth',
    input_size=7,
    num_features=6
)
scaler_sqz_momentum_30_15min_t5 = None







# close_RSI_7_hma_15
model_RSI_7_hma_15_15min_t1 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_RSI_7_hma_15_15min_t1.pth',
    input_size=7,
    num_features=6
)
scaler_RSI_7_hma_15_15min_t1 = None

model_RSI_7_hma_15_15min_t2 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_RSI_7_hma_15_15min_t2.pth',
    input_size=7,
    num_features=6
)
scaler_RSI_7_hma_15_15min_t2 = None

model_RSI_7_hma_15_15min_t3 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_RSI_7_hma_15_15min_t3.pth',
    input_size=7,
    num_features=6
)
scaler_RSI_7_hma_15_15min_t3 = None

model_RSI_7_hma_15_15min_t4 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_RSI_7_hma_15_15min_t4.pth',
    input_size=7,
    num_features=6
)
scaler_RSI_7_hma_15_15min_t4 = None

# close_RSI_14_hma_15
model_RSI_14_hma_15_15min_t1 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_RSI_14_hma_15_15min_t1.pth',
    input_size=7,
    num_features=6
)
scaler_RSI_14_hma_15_15min_t1 = None

model_RSI_14_hma_15_15min_t2 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_RSI_14_hma_15_15min_t2.pth',
    input_size=7,
    num_features=6
)
scaler_RSI_14_hma_15_15min_t2 = None

model_RSI_14_hma_15_15min_t3 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_RSI_14_hma_15_15min_t3.pth',
    input_size=7,
    num_features=6
)
scaler_RSI_14_hma_15_15min_t3 = None

model_RSI_14_hma_15_15min_t4 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_RSI_14_hma_15_15min_t4.pth',
    input_size=7,
    num_features=6
)
scaler_RSI_14_hma_15_15min_t4 = None

# close_RSI_28_hma_15
model_RSI_28_hma_15_15min_t1 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_RSI_28_hma_15_15min_t1.pth',
    input_size=7,
    num_features=6
)
scaler_RSI_28_hma_15_15min_t1 = None

model_RSI_28_hma_15_15min_t2 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_RSI_28_hma_15_15min_t2.pth',
    input_size=7,
    num_features=6
)
scaler_RSI_28_hma_15_15min_t2 = None

model_RSI_28_hma_15_15min_t3 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_RSI_28_hma_15_15min_t3.pth',
    input_size=7,
    num_features=6
)
scaler_RSI_28_hma_15_15min_t3 = None

model_RSI_28_hma_15_15min_t4 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_RSI_28_hma_15_15min_t4.pth',
    input_size=7,
    num_features=6
)
scaler_RSI_28_hma_15_15min_t4 = None






# close_slope_15_5_zscore
model_hma15_5_zscore_t1 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_15_5_zscore_15min_t1.pth',
    input_size=7,
    num_features=5
)
scaler_hma15_5_zscore_t1 = None

model_hma15_5_zscore_t2 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_15_5_zscore_15min_t2.pth',
    input_size=7,
    num_features=5
)
scaler_hma15_5_zscore_t2 = None

model_hma15_5_zscore_t3 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_15_5_zscore_15min_t3.pth',
    input_size=7,
    num_features=5
)
scaler_hma15_5_zscore_t3 = None

model_hma15_5_zscore_t4 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_15_5_zscore_15min_t4.pth',
    input_size=7,
    num_features=5
)
scaler_hma15_5_zscore_t4 = None


# close_slope_25_5_zscore
model_hma25_5_zscore_t1 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_25_5_zscore_15min_t1.pth',
    input_size=7,
    num_features=8
)
scaler_hma25_5_zscore_t1 = None

model_hma25_5_zscore_t2 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_25_5_zscore_15min_t2.pth',
    input_size=7,
    num_features=8
)
scaler_hma25_5_zscore_t2 = None

model_hma25_5_zscore_t3 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_25_5_zscore_15min_t3.pth',
    input_size=7,
    num_features=8
)
scaler_hma25_5_zscore_t3 = None

model_hma25_5_zscore_t4 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_25_5_zscore_15min_t4.pth',
    input_size=7,
    num_features=8
)
scaler_hma25_5_zscore_t4 = None

# close_slope_35_5_zscore
model_hma35_5_zscore_t1 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_35_5_zscore_15min_t1.pth',
    input_size=7,
    num_features=8
)
scaler_hma35_5_zscore_t1 = None

model_hma35_5_zscore_t2 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_35_5_zscore_15min_t2.pth',
    input_size=7,
    num_features=8
)
scaler_hma35_5_zscore_t2 = None

model_hma35_5_zscore_t3 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_35_5_zscore_15min_t3.pth',
    input_size=7,
    num_features=8
)
scaler_hma35_5_zscore_t3 = None

model_hma35_5_zscore_t4 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_35_5_zscore_15min_t4.pth',
    input_size=7,
    num_features=8
)
scaler_hma35_5_zscore_t4 = None

# close_slope_45_5_zscore
model_hma45_5_zscore_t1 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_45_5_zscore_15min_t1.pth',
    input_size=7,
    num_features=8
)
scaler_hma45_5_zscore_t1 = None

model_hma45_5_zscore_t2 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_45_5_zscore_15min_t2.pth',
    input_size=7,
    num_features=8
)
scaler_hma45_5_zscore_t2 = None

model_hma45_5_zscore_t3 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_45_5_zscore_15min_t3.pth',
    input_size=7,
    num_features=8
)
scaler_hma45_5_zscore_t3 = None

model_hma45_5_zscore_t4 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_45_5_zscore_15min_t4.pth',
    input_size=7,
    num_features=8
)
scaler_hma45_5_zscore_t4 = None

######################
# NEW SLOPE MODELS
######################

model_hma15_5_t2 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_15_5_raw_15min_t2.pth',
    input_size=7,
    num_features=6
)
scaler_hma15_5_t2 = None

model_hma15_5_t3 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_15_5_raw_15min_t3.pth',
    input_size=7,
    num_features=6
)
scaler_hma15_5_t3 = None

model_hma15_5_t4 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_15_5_raw_15min_t4.pth',
    input_size=7,
    num_features=6
)
scaler_hma15_5_t4 = None

model_hma15_5_t5 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_15_5_raw_15min_t5.pth',
    input_size=7,
    num_features=6
)
scaler_hma15_5_t5 = None

######################
# T3
######################

model_T3_6_t1 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_T3_6_slope_5_raw_15min_t1.pth',
    input_size=7,
    num_features=8
)
scaler_T3_6_t1 = None

model_T3_6_t2 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_T3_6_slope_5_raw_15min_t2.pth',
    input_size=7,
    num_features=8
)
scaler_T3_6_t2 = None

model_T3_6_t3 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_T3_6_slope_5_raw_15min_t3.pth',
    input_size=7,
    num_features=8
)
scaler_T3_6_t3 = None

model_T3_6_t4 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_T3_6_slope_5_raw_15min_t4.pth',
    input_size=7,
    num_features=8
)
scaler_T3_6_t4 = None

model_T3_6_t5 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_T3_6_slope_5_raw_15min_t5.pth',
    input_size=7,
    num_features=8
)
scaler_T3_6_t5 = None




######################
# LOAD RNN MODELS
######################


# Pre-load models and scalers for HMA 25 (Binary)
model_hma25_binary_t3 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_25_2_raw_15min_binary_t3.pth',
    input_size=10,
    num_features=6
)
scaler_hma25_binary_t3 = None

model_hma25_binary_t4 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_25_2_raw_15min_binary_t4.pth',
    input_size=10,
    num_features=6
)
scaler_hma25_binary_t4 = None

model_hma25_binary_t5 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_25_2_raw_15min_binary_t5.pth',
    input_size=10,
    num_features=6
)
scaler_hma25_binary_t5 = None



# Pre-load models and scalers for HMA 15
model_hma15_t1 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_15_2_raw_15min_t1.pth',
    input_size=10,
    num_features=3
)
scaler_hma15_t1 = None
#scaler_hma15_t1 = load_scaler('C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_15_2_raw_15min_t1_scaler.pth')

model_hma15_t2 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_15_2_raw_15min_t2.pth',
    input_size=10,
    num_features=3
)
scaler_hma15_t2 = None
#scaler_hma15_t2 = load_scaler('C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_15_2_raw_15min_t2_scaler.pth')

model_hma15_t3 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_15_2_raw_15min_t3.pth',
    input_size=10,
    num_features=3
)
scaler_hma15_t3 = None
#scaler_hma15_t3 = load_scaler('C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_15_2_raw_15min_t3_scaler.pth')

# Pre-load models and scalers for HMA 25
model_hma25_t1 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_25_2_raw_15min_t1.pth',
    input_size=10,
    num_features=3
)
scaler_hma25_t1 = None
#scaler_hma25_t1 = load_scaler('C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_25_2_raw_15min_t1_scaler.pth')

model_hma25_t2 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_25_2_raw_15min_t2.pth',
    input_size=10,
    num_features=3
)
scaler_hma25_t2 = None
#scaler_hma25_t2 = load_scaler('C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_25_2_raw_15min_t2_scaler.pth')

model_hma25_t3 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_25_2_raw_15min_t3.pth',
    input_size=10,
    num_features=3
)
scaler_hma25_t3 = None
#scaler_hma25_t3 = load_scaler('C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_25_2_raw_15min_t3_scaler.pth')

model_hma25_t4 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_25_2_raw_15min_t4.pth',
    input_size=10,
    num_features=3
)
scaler_hma25_t4 = None
#scaler_hma25_t4 = load_scaler('C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_25_2_raw_15min_t4_scaler.pth')

model_hma25_t5 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_25_2_raw_15min_t5.pth',
    input_size=10,
    num_features=5
)
scaler_hma25_t5 = None
#scaler_hma25_t4 = load_scaler('C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_25_2_raw_15min_t4_scaler.pth')

# Pre-load models and scalers for HMA 35
model_hma35_t1 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_35_2_raw_15min_t1.pth',
    input_size=10,
    num_features=3
)
scaler_hma35_t1 = None
#scaler_hma35_t1 = load_scaler('C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_35_2_raw_15min_t1_scaler.pth')

model_hma35_t2 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_35_2_raw_15min_t2.pth',
    input_size=10,
    num_features=3
)
scaler_hma35_t2 = None
#scaler_hma35_t2 = load_scaler('C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_35_2_raw_15min_t2_scaler.pth')

model_hma35_t3 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_35_2_raw_15min_t3.pth',
    input_size=10,
    num_features=3
)
scaler_hma35_t3 = None
#scaler_hma35_t3 = load_scaler('C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_35_2_raw_15min_t3_scaler.pth')

model_hma35_t4 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_35_2_raw_15min_t4.pth',
    input_size=10,
    num_features=3
)
scaler_hma35_t4 = None
#scaler_hma35_t4 = load_scaler('C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_35_2_raw_15min_t4_scaler.pth')

# Pre-load models and scalers for HMA 45
model_hma45_t1 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_45_2_raw_15min_t1.pth',
    input_size=10,
    num_features=3
)
scaler_hma45_t1 = None
#scaler_hma45_t1 = load_scaler('C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_45_2_raw_15min_t1_scaler.pth')

model_hma45_t2 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_45_2_raw_15min_t2.pth',
    input_size=10,
    num_features=3
)
scaler_hma45_t2 = None
#scaler_hma45_t2 = load_scaler('C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_45_2_raw_15min_t2_scaler.pth')

model_hma45_t3 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_45_2_raw_15min_t3.pth',
    input_size=10,
    num_features=3
)
scaler_hma45_t3 = None
#scaler_hma45_t3 = load_scaler('C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_45_2_raw_15min_t3_scaler.pth')

model_hma45_t4 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_45_2_raw_15min_t4.pth',
    input_size=10,
    num_features=3
)
scaler_hma45_t4 = None
#scaler_hma45_t4 = load_scaler('C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_slope_45_2_raw_15min_t4_scaler.pth')


# Pre-load models for CCI
model_close_raw_cci_hma_14_t1 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_raw_15min_cci_hma_14_t1.pth',
    input_size=10,
    num_features=5
)
scaler_close_raw_cci_hma_14_t1 = load_scaler('C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_raw_15min_cci_hma_14_t1_scaler.pth')

model_close_raw_cci_hma_14_t2 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_raw_15min_cci_hma_14_t2.pth',
    input_size=10,
    num_features=5
)
scaler_close_raw_cci_hma_14_t2 = load_scaler('C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_raw_15min_cci_hma_14_t2_scaler.pth')

model_close_raw_cci_hma_14_t3 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_raw_15min_cci_hma_14_t3.pth',
    input_size=10,
    num_features=5
)
scaler_close_raw_cci_hma_14_t3 = load_scaler('C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_raw_15min_cci_hma_14_t3_scaler.pth')

model_close_raw_cci_hma_18_t3 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_raw_15min_cci_hma_18_t3.pth',
    input_size=10,
    num_features=5
)
scaler_close_raw_cci_hma_18_t3 = load_scaler('C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_raw_15min_cci_hma_18_t3_scaler.pth')

model_close_raw_cci_hma_22_t3 = load_model(
    model_path='C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_raw_15min_cci_hma_22_t3.pth',
    input_size=10,
    num_features=5
)
scaler_close_raw_cci_hma_22_t3 = load_scaler('C:/Users/daraa/Desktop/algo-modeling-v2/RNN_models/close_raw_15min_cci_hma_22_t3_scaler.pth')
