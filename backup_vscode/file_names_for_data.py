from my_imports import base_model_parameters
import boto3
from algo_feature_engineering.data_load_utils import (
    load_ticker_list,
    download_ticker_data
)

######################
# PARAMETERS 
######################
#params now stored in algo_feature_engineering.features.base_model.base_model_parameters
#only defining specific params from base model used in feature calculations. Define more as needed
z_score_hma_length = base_model_parameters["base_model"]["z_score_hma_length"]

# ######################
# # LABELING PARAMETERS 
# ######################
# look_ahead_window = 12 #base_model_parameters["labeling"]["look_ahead_window"]
# labeling_window = 100  #base_model_parameters["labeling"]["labeling_window"] 
# upper_coeff_smooth = 3.00 #1.5 #base_model_parameters["labeling"]["upper_coeff"]
# upper_coeff_raw = 3.00 #1.5
# lower_coeff_smooth = -1.5 #-1.5 #base_model_parameters["labeling"]["lower_coeff"]
# lower_coeff_raw = -1.5 #-1.5
# n_lagging_features_1 = base_model_parameters["labeling"]["n_lagging_features_1"]
# atr_period_labeling = 120 #base_model_parameters["labeling"]["atr_period_labeling"]
# close_hma_smoothing = 25 #25


# savgol_poly_order = 3



######################
# CONFIGS
######################
#TICKER_FILE = "../../ticker_lists/new_full_ticker_list_cs.txt"
# AWS S3 Configuration
S3_BUCKET_NAME = "algo-trading-data" 
#S3_FILE_PREFIX = f"precalc/new_full_ticker_list_cs/" #f"precalc/dara_10-17-24/" 
#S3_FILE_SUFFIX = "polygon_30minute_adj_precalc"
S3_FILE_SUFFIX = "polygon_15minute_adj_precalc"
S3_CLIENT = boto3.client('s3')

######################
# Set the ticker list here
######################


# Define the paths to your ticker list files
#TICKER_FILE_CS = "../../ticker_lists/new_full_ticker_list_cs_v2.txt"
TICKER_FILE_CS = r"C:\Users\daraa\Desktop\data\new_stock_ticker_list.txt"
TICKER_FILE_ETF = r"C:\Users\daraa\Desktop\algo-modeling\ticker_lists\new_full_ticker_list_etf_v2.txt" 
TICKER_FILE_CRYPTO = r"C:\Users\daraa\Desktop\data\crypto_coins_15min.txt"

# Define the corresponding S3_FILE_PREFIX for each ticker list
# S3_PREFIX_CS = "precalc/new_full_ticker_list_cs_v2/"
# S3_PREFIX_CS = "precalc/new_stock_ticker_list/"
# S3_PREFIX_ETF = "precalc/new_full_ticker_list_etf_v2/" 
S3_PREFIX_CS = "precalc/new_stock_ticker_list_15min/"
S3_PREFIX_ETF = "precalc/new_full_ticker_list_etf_v2_15min/" 
S3_PREFIX_CRYPTO = "precalc/crypto_coins_15min/"

# Dictionary mapping ticker types to their file paths and prefixes
TICKER_CONFIG = {
    'cs': {'file': TICKER_FILE_CS, 'prefix': S3_PREFIX_CS},
    'etf': {'file': TICKER_FILE_ETF, 'prefix': S3_PREFIX_ETF},
    'crypto': {'file': TICKER_FILE_CRYPTO, 'prefix': S3_PREFIX_CRYPTO}
}

# Specify which types of tickers you want to combine
selected_types = ['etf', 'cs', 'crypto']  # Modify this list to choose which types to include

# Load and combine selected ticker types
combined_symbol_prefix_list = []
for ticker_type in selected_types:
    if ticker_type in TICKER_CONFIG:
        trading_symbols = load_ticker_list(TICKER_CONFIG[ticker_type]['file'])
        symbol_prefix_list = [(symbol, TICKER_CONFIG[ticker_type]['prefix']) 
                            for symbol in trading_symbols]
        combined_symbol_prefix_list.extend(symbol_prefix_list)

from collections import OrderedDict
# Create an ordered dictionary to remove duplicates while preserving order
unique_symbol_prefix_dict = OrderedDict(combined_symbol_prefix_list)
# Convert back to a list of tuples
unique_combined_symbol_prefix_list = list(unique_symbol_prefix_dict.items())
# Create list of just the symbols
unique_combined_symbols = [symbol for symbol, _ in unique_combined_symbol_prefix_list]

print(unique_combined_symbol_prefix_list)
print(unique_combined_symbols)



