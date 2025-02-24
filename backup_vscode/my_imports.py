import plotly.graph_objects as go
import numpy as np
import pandas as pd
import talib # to install run this in anaconda command prompt: conda install -c conda-forge ta-lib
from sklearn.preprocessing import MinMaxScaler
import boto3
#from xgboost import XGBClassifier
import plotly.graph_objects as go
from itables import init_notebook_mode
init_notebook_mode(all_interactive=True) # makes all dataframes pretty

from algo_feature_engineering.features.base_model import (
    process_base_model,
    base_model_parameters
)    
from algo_feature_engineering.features.ma import (
    moving_average_features_normalized,
    moving_average_features_slopes,
    moving_average_features_percent_diff,
    calculate_hma
)
from algo_feature_engineering.features.indicators import (
    volatility_features,
    macd_features,
    bollinger_band_features,
    parabolic_sar_features,
    rsi_features,
    roc_features,
    cci_features
)
from algo_feature_engineering.features.time import period_hour
from algo_feature_engineering.data_load_utils import (
    load_ticker_list,
    download_ticker_data
)
from algo_feature_engineering.features.trend import trend_features
from algo_feature_engineering.features.tsa import compute_auto_correlation
from algo_feature_engineering.labeling import triple_barrier_labels
from algo_feature_engineering.features.utils import calculate_trend_vectorized 
from algo_feature_engineering.features.tsa import calculate_weighted_distance

import warnings
# Suppress the specific RuntimeWarning
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in cast')
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning, message='DataFrame is highly fragmented')

