######################
# IMPORT FUNCTIONS AND MODULES
######################
import optuna
from tqdm.notebook import tqdm
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
import numpy as np
import plotly.express as px
from RNN_stuff import load_model
from RNN_stuff import load_scaler
from RNN_stuff import predict_sequences
from algo_feature_engineering.features.ma import (
    moving_average_features_normalized,
    moving_average_features_slopes,
    moving_average_features_percent_diff,
    calculate_hma
)
from algo_feature_engineering.features.trend import trend_features
from algo_feature_engineering.features.tsa import compute_auto_correlation
from algo_feature_engineering.labeling import triple_barrier_labels
from algo_feature_engineering.features.utils import calculate_trend_vectorized 
from algo_feature_engineering.features.tsa import calculate_weighted_distance
from load_data import load_and_process_data_dict
from grouped_feature_functions import calculate_precomputed_features
from algo_feature_engineering.features.base_model import (
    process_base_model,
    base_model_parameters
)    
import logging
import joblib
from datetime import datetime
import gc

################################
# IMPORT FEATURE FUNCTIONS
#################################
from basic_feature_functions import *

################################
# IMPORT UTILITY FUNCTIONS
#################################
from utility_functions import *

################################
# IMPORT CRITICAL INFRA FUNCTIONS
#################################
from pdf_function import *
from backtest_functions import *
from buy_sell_signals_functions import *
from add_predictions_functions import *
from labeling import *
from train_models import *
from feature_list import *
from data_prep_functions import *
from my_functions import *

######################
# IMPORT RNN MODELS AND SCALERS 
######################
from RNN_load import * 

####################################
# IMPORT FUNCTIONS  FOR RNN FEATURES 
###################################
from RNN_features import *

######################
# OBJECTIVE FUNCTION 
######################

# Define the objective function for Optuna
def objective(trial, data_dict):

    try:

        ######################
        # MEMORY CHECK
        ######################
        # Memory check at start of trial
        memory_mb, memory_percent = monitor_memory()
        if memory_percent > 90:
            logging.warning("High memory usage detected at start of trial")
            gc.collect()
   
        ######################
        # SET OPTUNA PARAMS
        ######################
        optuna_params = {
            # params for labeling
            'close_hma_smoothing': trial.suggest_int('close_hma_smoothing', low=7, high=40, step=5),
            'look_ahead_window': trial.suggest_int('look_ahead_window', low=4, high=12, step=1),
            'upper_coeff_smooth': trial.suggest_float('upper_coeff_smooth', low=1.0, high=4.0, step=0.2),
            'num_classes_per_side': trial.suggest_int('num_classes_per_side', low=4, high=7, step=1),
            #'lower_coeff_smooth': trial.suggest_float('lower_coeff_smooth', low=0.2, high=4.6, step=0.2),
            #'quality_multiplier_strength': trial.suggest_int('quality_multiplier_strength', low=1, high=8, step=1),
            #'use_calculate_path_quality_score': trial.suggest_categorical('use_calculate_path_quality_score', [True, False]),

            # params for stop loss
            'atr_period_for_stoploss': trial.suggest_int('atr_period_for_stoploss', low=10, high=60, step=10),
            'atr_coeff_for_stoploss': trial.suggest_float('atr_coeff_for_stoploss', low=2.6, high=3.8, step=0.2),
            #'use_performance_scaling': trial.suggest_categorical('use_performance_scaling', [True, False]),
            
            # params for buy/sell signals 
            'use_class_spread': trial.suggest_categorical('use_class_spread', [True, False]),
            'hma_period_for_class_spread': trial.suggest_int('hma_period_for_class_spread', low=10, high=24, step=4),
            'hma_adjust': trial.suggest_int('hma_adjust', low=0, high=16, step=2),
            'apply_930_logic': trial.suggest_categorical('apply_930_logic', [True, False]),
            'morning_hma_adjustment': trial.suggest_int('morning_hma_adjustment', low=0, high=6, step=2),
            #'hma_short_adjustment': trial.suggest_int('hma_short_adjustmentt', low=-4, high=10, step=2),
            #'hma_adjust_ma_period': trial.suggest_int('hma_adjust_ma_period', low=200, high=1000, step=200),
            #'buy_cut_off': trial.suggest_categorical('buy_cut_off', ['12:00', '12:30', '13:00', '13:30', '14:00', '14:30', '15:00']),

            # params for modeling
            #'use_ensemble': trial.suggest_categorical('use_ensemble', [True, False]),
            # 'max_depth': trial.suggest_int('max_depth', low=3, high=6, step=1),
            # 'learning_rate': trial.suggest_categorical('learning_rate', [0.35, 0.3, 0.1, 0.01, 0.005]),

            # params for position sizing
            'kelly_multiplier': trial.suggest_float('kelly_multiplier', low=2, high=7, step=1),
            'lookback_period_for_position_size': trial.suggest_int('lookback_period_for_position_size', low=10, high=100, step=10),      
            
        }

        # Add parameters based on use_performance_scaling
        if optuna_params.get('use_performance_scaling', True):
            # MODIFIED: Add stop_loss_scale_coeff only when use_performance_scaling is True
            optuna_params['stop_loss_scale_coeff'] = trial.suggest_int('stop_loss_scale_coeff', low=25, high=375, step=50)
        else:
            # Add parameters used when use_performance_scaling is False
            optuna_params['stop_loss_adjust'] = trial.suggest_float('stop_loss_adjust', low=1.0, high=1.6, step=0.05)
            optuna_params['stop_loss_adjust_sma_period'] = trial.suggest_int('stop_loss_adjust_sma_period', low=200, high=1000, step=100)


        use_hard_coded_params = False

        if use_hard_coded_params:
            # params for labeling
            close_hma_smoothing = 10
            look_ahead_window = 10
            upper_coeff_smooth = 3.0000
            lower_coeff_smooth = 3.0000
            num_classes_per_side = 5  
            quality_multiplier_strength = 6
            use_calculate_path_quality_score = False #True
            use_trajectory_score = True
            morning_hma_adjustment = 0

            # params for stop loss
            atr_period_for_stoploss = 5
            atr_coeff_for_stoploss = 2.7000
            use_performance_scaling = True  
            stop_loss_scale_coeff = 100 
            stop_loss_adjust = 1.1000  
            stop_loss_adjust_sma_period = 400 

            # params for buy/sell signals
            use_class_spread = True
            hma_period_for_class_spread = 12
            hma_adjust = 2
            hma_adjust_ma_period = 400
            buy_cut_off = "12:30"
            apply_930_logic = True
            hma_short_adjustment = 0

            # params for modeling
            use_ensemble = False  
            # max_depth = 4
            # learning_rate = 0.3500

            # params for position sizing
            kelly_multiplier = 2
            lookback_period_for_position_size = 50  # maintained from original

        else:
            # params for labeling
            close_hma_smoothing = optuna_params['close_hma_smoothing']
            look_ahead_window = optuna_params['look_ahead_window']
            upper_coeff_smooth = optuna_params['upper_coeff_smooth']
            lower_coeff_smooth = upper_coeff_smooth #optuna_params['lower_coeff_smooth'] 
            num_classes_per_side = optuna_params['num_classes_per_side']
            morning_hma_adjustment = optuna_params['morning_hma_adjustment']
            quality_multiplier_strength = optuna_params.get('quality_multiplier_strength', 3)
            #use_calculate_path_quality_score = optuna_params['use_calculate_path_quality_score']
            
            #params for stop loss
            atr_period_for_stoploss = optuna_params['atr_period_for_stoploss']
            atr_coeff_for_stoploss = optuna_params['atr_coeff_for_stoploss']
            use_performance_scaling = optuna_params.get('use_performance_scaling', True)
            stop_loss_scale_coeff = optuna_params.get('stop_loss_scale_coeff', 100) 
            stop_loss_adjust = optuna_params.get('stop_loss_adjust', 1.0)  # Default to 1.0 if not in params
            stop_loss_adjust_sma_period = optuna_params.get('stop_loss_adjust_sma_period', 400)  # Default to 400 if not in params
            
            # params for buy/sell signals
            use_class_spread = optuna_params['use_class_spread']
            hma_period_for_class_spread = optuna_params['hma_period_for_class_spread']
            hma_adjust = optuna_params['hma_adjust']
            hma_adjust_ma_period = optuna_params.get('hma_adjust_ma_period', 400) #(using stop_loss_adjust_sma_period instead)
            apply_930_logic = optuna_params['apply_930_logic']
            buy_cut_off = optuna_params.get('buy_cut_off', '14:00')
            hma_short_adjustment = optuna_params.get('hma_short_adjustment', 0)

            # params for modeling
            use_ensemble = optuna_params.get('use_ensemble', True)
            # max_depth = optuna_params['max_depth']
            # learning_rate = optuna_params['learning_rate']

            # params for position sizing
            kelly_multiplier = optuna_params.get('kelly_multiplier', 2)
            lookback_period_for_position_size = optuna_params['lookback_period_for_position_size']


        #################################
        # OTHER HARD CODED PARAMETERS 
        #################################

        #long or short
        is_model_type_short = True

        # buy and sell
        stop_loss_adjust = 1.05
        classes_for_spread = 1
        
        # labeling
        use_calculate_path_quality_score = False
        use_trajectory_score = True
        
        # hardcoded params (for labeling debug)
        labeling_debugging_mode = True
        debug_symbol = 'TQQQ'
        debug_start_date = '2024-05-25'  
        debug_end_date = '2024-06-10'

        # hardcoded params (overnight holds)
        sell_at_end_of_day = False
        use_next_day_prices = True #(in labeling)
        use_intraday_atr = False 
        overnight_position_size = 0.5 

        # hardcoded params (other)
        atr_period_labeling = 120
        savgol_poly_order = 5
        show_confusion_matrix = True
        show_roc_curves = True 
        show_xgboost_feature_importance = True

        # hardcoded params (xgboost)
        train_test_split_percent = 0.6500
        include_symbol_as_feature = False  
        num_epochs = 150
        print_interval = 5
        early_stopping_enabled = True  
        early_stopping_rounds = 50
        subsample =  0.977 
        colsample_bytree = 0.77292 
        alpha = 0.7650 
        lambda_p = 0.54667 
        min_child_weight= 1

        
        ######################
        # INITIALIZE DATAFRAME TO STORE ALL DATA 
        ######################
        
        # Initialize an empty DataFrame to store all data for all stocks
        df_features_master = None

        ######################
        # ITERATE OVER SYMBOLS
        ######################
        
        for symbol in tqdm(optuna_ticker_list, desc="Processing Symbols"):

            ######################
            # CLEAN UP PREVIOUS ITERATION
            ######################
            cleanup_dataframes()
                    
            ######################
            # LOAD DATA
            ######################
            ticker_df_adjusted = data_dict[symbol].copy()

            ######################
            # ADD SYMBOL AS FEATURE
            ######################
            ticker_df_adjusted['symbol'] = symbol
            ticker_df_adjusted['symbol'] = ticker_df_adjusted['symbol'].astype('category')
        
            #####################################
            # CREATE RAW PRICE AND SMOOTHED PRICE
            #####################################       
            # define smoothed prices 
            ticker_df_adjusted['close'] = calculate_hma(ticker_df_adjusted['close'], period=close_hma_smoothing)
            ticker_df_adjusted['high'] = calculate_hma(ticker_df_adjusted['high'], period=close_hma_smoothing)
            ticker_df_adjusted['low'] = calculate_hma(ticker_df_adjusted['low'], period=close_hma_smoothing)
            ticker_df_adjusted['open'] = calculate_hma(ticker_df_adjusted['open'], period=close_hma_smoothing)
        
            #####################################
            # SLOPE OF HMA (SMOOTHED PRICE) -- CONSIDER DELETING 
            #####################################
            custom_periods = [70, 45, 25, 15, 10, 5, 4]
            ticker_df_adjusted = add_slope_features(ticker_df_adjusted, periods=custom_periods)
        
            ##############################
            # FEATURES FOR SMOOTHED PRICE
            ###############################

            custom_periods = [350, 300, 250, 200, 150, 100, 75, 50, 40, 30, 25, 20, 15, 10, 5]
            ticker_df_adjusted = add_long_term_hma_features(ticker_df_adjusted, 
                                                        periods=custom_periods,
                                                        calculate_slopes=True,
                                                        slope_period=2)
            
            # what percent of the of the time has hma_long_term_slope been positive? 
            custom_periods = [350, 300, 250, 200, 150, 100, 75, 50, 20]
            ticker_df_adjusted = add_long_term_slope_percentages(ticker_df_adjusted, periods=custom_periods)
        
            # percent diff
            ticker_df_adjusted = add_distance_metrics(ticker_df_adjusted, epsilon=1e-9)
        
            # price below the MA
            custom_periods = [150, 75, 40, 30, 25, 20, 15, 10, 5]
            ticker_df_adjusted = add_price_position_indicators(ticker_df_adjusted, periods=custom_periods)
            
            # st. dev. features
            custom_periods = [40, 30, 20]
            ticker_df_adjusted = add_std_deviation_features(ticker_df_adjusted, periods=custom_periods)

            # Calculate HMA distance features
            ticker_df_adjusted = calculate_hma_distance_features(ticker_df_adjusted)

            # Calculate HMA cross-period features
            ticker_df_adjusted = calculate_hma_cross_period_features(ticker_df_adjusted)

            # Calculate trend coherence features
            ticker_df_adjusted = calculate_trend_coherence_features(ticker_df_adjusted)
        
            ######################
            # MORE TRENDS
            ######################
            ticker_df_adjusted = add_all_trend_features(ticker_df_adjusted)

            ######################
            # TREND_BUY_THRESHOLD OVER CERTAIN THRESHOLDS 
            ######################
            ticker_df_adjusted = add_trend_buy_threshold_columns(ticker_df_adjusted)

            ######################
            # CCI FEATURES (SMOOTH)
            ######################
            ticker_df_adjusted = calculate_cci_features(ticker_df_adjusted, calculate_hma)

            ######################
            # ROC/RSI/PPO/STOCH FEATURES (SMOOTH)
            ######################
            ticker_df_adjusted = calculate_technical_indicators(ticker_df_adjusted)

            ######################
            # STATIONARY INDICATORS (hma_long_term_slope_percent_positive_75, trend_buy)
            ######################   
            ticker_df_adjusted = add_technical_indicators(ticker_df_adjusted, 'hma_long_term_slope_percent_positive_75')
            ticker_df_adjusted = add_technical_indicators(ticker_df_adjusted,'trend_buy')

            ######################
            # MORE BUY TREND FEATURES 
            ######################  
            ticker_df_adjusted = misc_trend_buy_features(ticker_df_adjusted)

            ######################
            # FEATURE: HOUR
            ######################
            ticker_df_adjusted = add_time_range_features(ticker_df_adjusted)

            ######################
            # FEATURE: CANDLES 
            ######################
            ticker_df_adjusted = add_candle_pattern_features(ticker_df_adjusted,open_col='open',close_col='close')
                    
            ######################
            # VARFIOUS PRICE SLOPE FEATURES
            ######################
            ticker_df_adjusted = add_comprehensive_slope_features(ticker_df_adjusted, close_col='close',close_raw_col='close_raw')
        
            ######################
            # FEATURE: ENTROPY
            ######################
            ticker_df_adjusted = add_entropy_features(df=ticker_df_adjusted, close_col='close', high_col='high', low_col='low', trend_buy_col='trend_buy')
        
            ######################
            # FEATURE: CROSS OVER
            ######################
            custom_config = [
                {'column': 'trend_slope_2nd_derivative_buy', 'threshold': 0.0},
                {'column': 'trend_buy', 'threshold': 0.5},
                {'column': 'hma_long_term_slope_20', 'threshold': 0.0},
                {'column': 'hma_long_term_slope_50', 'threshold': 0.0},
                {'column': 'hma_long_term_slope_75', 'threshold': 0.0},
                {'column': 'hma_long_term_slope_100', 'threshold': 0.0},
                {'column': 'hma_long_term_slope_150', 'threshold': 0.0},
                {'column': 'hma_long_term_slope_200', 'threshold': 0.0},
                {'column': 'hma_long_term_slope_250', 'threshold': 0.0},
                {'column': 'hma_long_term_slope_300', 'threshold': 0.0},
                {'column': 'hma_long_term_slope_350', 'threshold': 0.0},
                # Add any other necessary configurations
            ]

            ticker_df_adjusted = add_all_crossover_features(ticker_df_adjusted, config=custom_config)

            ######################
            # Price Percentiles 
            #####################
            custom_windows = [200, 100, 50, 20, 10, 5]
            ticker_df_adjusted = add_rolling_percentile_features(ticker_df_adjusted, windows=custom_windows, column='close')

            ######################
            # Trailing z-score 
            ######################
            custom_windows = [500, 200, 100, 20]
            ticker_df_adjusted = add_rolling_zscore_features(ticker_df_adjusted, windows=custom_windows, column='close')

        
            ######################
            # RELATIVE ATR USED TO DETERMINE UPPER AND LOWER BARRIER (Savitzky-Golay Filter) 
            ######################
            # Calculate ATR percent change
            atr_percent_change = calculate_savgol_atr_percent(
                df=ticker_df_adjusted,
                window_length=close_hma_smoothing,
                polyorder=savgol_poly_order,
                atr_period=atr_period_labeling,
                use_intraday_atr=use_intraday_atr,
                calc_intraday_atr=calc_intraday_atr if use_intraday_atr else None
            )

            # Add only the final result to your dataframe
            ticker_df_adjusted['atr_percent_change_for_labeling_sav_gol'] = atr_percent_change

            ######################
            # RELATIVE ATR USED TO DETERMINE UPPER AND LOWER BARRIER 
            ######################
            ticker_df_adjusted = add_atr_percent_change(ticker_df_adjusted, atr_period_labeling=atr_period_labeling) 

            #########################################
            # VOLATILITY ADJUSTED MOMENTUM FEATURES
            ###########################################
            ticker_df_adjusted = calculate_volatility_adjusted_momentum_features(ticker_df_adjusted)
        
            ######################
            # TRAILING PRICE CHANGE
            ######################
            custom_lookbacks = [300, 200, 100, 20, 15, 10, 7, 5, 3]
            ticker_df_adjusted = add_trailing_pct_change_atr_features(ticker_df_adjusted, lookbacks=custom_lookbacks)

            ######################
            # LABELS 
            ######################
            ticker_df_adjusted, debug_output_labeling = label_financial_data(
                ticker_df=ticker_df_adjusted,
                symbol=symbol,   
                num_classes_per_side=num_classes_per_side,     
                look_ahead_window=look_ahead_window,                  
                upper_coeff_smooth=upper_coeff_smooth,  
                lower_coeff_smooth=lower_coeff_smooth,  
                use_next_day_prices = use_next_day_prices, 
                use_calculate_path_quality_score= use_calculate_path_quality_score,
                quality_multiplier_strength = quality_multiplier_strength,
                use_trajectory_score=use_trajectory_score,
                labeling_debugging_mode=labeling_debugging_mode,
                debug_symbol=debug_symbol,
                debug_start_date=debug_start_date,
                debug_end_date=debug_end_date,
                
            )
            
            ######################
            # CREATE DF_FEATURES
            ######################
            # Creating the df_features DataFrame
            df_features = pd.DataFrame(index=ticker_df_adjusted.index)
        
            # Define the feature columns
            feature_columns = ['close_raw_percentile_5', 'close_raw_percentile_10', 'close_raw_percentile_20', 'close_percentile_5', 'close_percentile_10', 'close_percentile_20']

            # Call the function with the specified parameters
            df_features = add_lagging_features(ticker_df_adjusted, feature_columns=feature_columns, step_size=1, num_steps=5)
        
            # add non-lagging features 
            df_features = copy_other_features(ticker_df_adjusted, df_features)
        
            # reduce data size
            df_features = optimize_dataframe_fast(df_features)
        
            # add df_features to df_features_master
            if df_features_master is None:
                df_features_master = df_features
            else:
                df_features_master = pd.concat([df_features_master, df_features], ignore_index=False)
        
        # check for duplicate rows
        assert not df_features_master.duplicated().any(), "Duplicate rows found in dataframe"
        
        #################################################################
        # COPY DATAFRAME: ONE FOR PREDICTIONS AND ONE FOR MODEL TRAINING
        #################################################################
        # Make a deep copy of df_features_master
        df_features_master_for_model = df_features_master.copy()
        
        # Make a deep copy of df_features_master
        df_features_master_for_predictions = df_features_master.copy()
        
        ###################################
        # PREPARE DATAFRFAME FOR MODELING
        ####################################

        # label mapping 
        df_features_master_for_model, label_encoder = prepare_labels(df_features_master_for_model)
        df_features_master_for_predictions = encode_prediction_labels(df_features_master_for_predictions, label_encoder)

        # Convert labels to integers
        df_features_master_for_model['label'] = df_features_master_for_model['label'].astype(int)
        df_features_master_for_predictions['label'] = df_features_master_for_predictions['label'].astype(int)

        # drop -100 and -999 labels        
        df_features_master_for_model = df_features_master_for_model[~df_features_master_for_model['label'].isin([-100, -999])]
        
        # drop non-feature columns 
        df_features_master_for_model = drop_non_feature_columns(df_features_master_for_model, include_symbol=include_symbol_as_feature)
        
        # Sort the dataframe by its index (timestamp) in ascending order
        df_features_master_for_model = df_features_master_for_model.sort_index(ascending=True)

        # Remove rows with NaN values
        df_features_master_for_model = drop_na_with_warning(df_features_master_for_model, threshold_pct=10)
        df_features_master_for_predictions = drop_na_with_warning(df_features_master_for_predictions, threshold_pct=10)

        # train/buffer/test split 
        df_train, df_test, test_data_begin_timestamp = split_time_series_with_buffer(df_features_master_for_model,train_percent=train_test_split_percent,buffer_percent=0.01)
        
        # drop the index 
        df_train = df_train.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)

        # split into X and Y
        X_train, y_train, X_test, y_test = prepare_and_validate_data(df_train, df_test)

        # Get the fitted scaler and features to scale
        scaler, features_to_scale = get_fitted_scaler(X_train, scaler_type='standard')

        # Scale the training and test sets
        X_train = apply_scaling(X_train, scaler, features_to_scale)
        X_test = apply_scaling(X_test, scaler, features_to_scale)
        df_features_master_for_predictions = apply_scaling(df_features_master_for_predictions, scaler, features_to_scale)

        # Round the scaled datasets
        X_train = round_numeric_features(X_train, decimals=2)
        X_test = round_numeric_features(X_test, decimals=2)
        df_features_master_for_predictions = round_numeric_features(df_features_master_for_predictions, decimals=2, exclude_columns=['close', 'high', 'low', 'close_raw', 'high_raw', 'low_raw'])

        # Calculate the class and sample weights
        class_weights, sample_weights = calculate_weights(y_train)

        # number of distinct classes 
        num_classes = len(np.unique(y_train))

        ######################
        # TRAIN MODELS
        ######################

        # Call the training function
        ensemble_info, models_info, device = train_and_save_models(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            sample_weights=sample_weights,
            main_model_path=main_model_path,
            num_classes=num_classes,
            early_stopping_enabled=True,
            early_stopping_rounds=early_stopping_rounds,
            num_epochs=num_epochs
        )
        
        #####################
        # GARBAGE COLLECTION  
        #####################
        cleanup_memory()
        
        #############################
        # SPLIT DATA FOR PREDICTIONS
        ##############################
        # test data only 
        df_features_master_for_predictions_test  = prepare_test_data(df_features_master_for_predictions, test_data_begin_timestamp)

        # split into deciles 
        (decile_1, decile_2, decile_3, decile_4, decile_5,
        decile_6, decile_7, decile_8, decile_9, decile_10,
        full_test_set) = split_into_deciles(df_features_master_for_predictions_test, verbose=True)

        #############################
        # ADD PREDICTIONS
        ##############################
        df_predictions_full_test, fig_ens_full_test = process_predictions_in_batches(full_test_set, batch_size=100000, model_path=main_model_path, debug=False, include_confusion_matrix=True, time_period="Full Test Set", use_ensemble=use_ensemble)
        all_ensemble_figs = []
        all_ensemble_figs.append(fig_ens_full_test)

        #############################
        # ADD BUY AND SELL SIGNALS 
        ##############################

        df_with_signals_full_test_set =  process_all_symbols(
                df=df_predictions_full_test,  
                hma_period = hma_period_for_class_spread,
                feature_suffix="",
                classes_for_spread=classes_for_spread,
                buy_signal_n_classes=num_classes_per_side,
                sell_signal_n_classes=num_classes_per_side,
                debug= False,
                use_class_spread=use_class_spread,
                use_adjusted_hma=True,
                hma_adjust_ma_period=stop_loss_adjust_sma_period, # just using the same period
                hma_adjust=hma_adjust,
                buy_time_cutoff=buy_cut_off,
                apply_930_logic=apply_930_logic,
                sell_at_end_of_day=sell_at_end_of_day,
                morning_hma_adjustment=morning_hma_adjustment,
                is_model_type_short=is_model_type_short
            )
        
        df_with_signals_full_test_set =  process_all_symbols(
                df=df_with_signals_full_test_set,  
                hma_period = (hma_period_for_class_spread - hma_short_adjustment), #shorter period 
                feature_suffix="_fast",
                classes_for_spread=classes_for_spread,
                buy_signal_n_classes=num_classes_per_side,
                sell_signal_n_classes=num_classes_per_side,
                debug= False,
                use_class_spread=use_class_spread,
                use_adjusted_hma=True,
                hma_adjust_ma_period=stop_loss_adjust_sma_period, 
                hma_adjust=hma_adjust,
                buy_time_cutoff=buy_cut_off,
                apply_930_logic=apply_930_logic,
                sell_at_end_of_day=sell_at_end_of_day,
                morning_hma_adjustment=morning_hma_adjustment,
                is_model_type_short=is_model_type_short
            )

        #############################
        # BACKTESTING
        ##############################
        results_full_test_set = run_backtest(
            df=df_with_signals_full_test_set,
            atr_coeff=atr_coeff_for_stoploss,
            initial_capital=10000,
            debug=False,
            kelly_debug=False,
            atr_period_for_stoploss=atr_period_for_stoploss,
            stop_loss_adjust=stop_loss_adjust,
            stop_loss_adjust_sma_period=stop_loss_adjust_sma_period,
            kelly_multiplier=kelly_multiplier,
            overnight_position_size=overnight_position_size, 
            lookback_period_for_position_size=lookback_period_for_position_size, 
            stop_loss_scale_coeff=stop_loss_scale_coeff,
            use_performance_scaling=use_performance_scaling,
            return_equity_curves=True,
            is_model_type_short=is_model_type_short
        )

        #############################
        # CREATE AND SAVE PDF
        ##############################
        # First convert the dates to timezone-aware datetime objects
        start_date = pd.to_datetime(debug_start_date).tz_localize('US/Eastern')
        end_date = pd.to_datetime(debug_end_date).tz_localize('US/Eastern')

        save_all_plots_in_one_pdf(
            trial_number=trial.number,
            models_info=models_info,
            X_test=X_test,
            y_test=y_test,
            num_classes=num_classes,
            pdf_output_dir=optuna_trials_dir,
            optuna_params=optuna_params, 
            df_train=df_train, 
            df_test=df_test,
            filtered_df=ticker_df_adjusted.loc[start_date:end_date],  
            backtesting_df=df_with_signals_full_test_set.tz_localize('US/Eastern').loc[start_date:end_date],
            df_features_master_for_predictions=df_features_master_for_predictions,
            start_date= debug_start_date, 
            end_date= debug_end_date, 
            device=device,
            show_feature_importance=True,
            show_roc_curves=True,
            extra_figures=all_ensemble_figs,
            debug_output=debug_output_labeling,
            equity_curve_plots=results_full_test_set.equity_curve_plots,
            trades_df=results_full_test_set.trades_df,
            backtesting_results=results_full_test_set
        )


        #############################
        # OPTUNA OBJECTIVE VALUE
        ##############################

        # Get the total returns from the full test set
        final_objective = results_full_test_set.summary_metrics['average_total_return_percentage']
        print(f"\nFull Test Set Total Return: {final_objective:.2f}%")
        return final_objective


    except MemoryError:
        logging.error("Memory Error encountered in trial!")
        print("Memory Error encountered!")
        gc.collect()
        return float('-inf')
    
    except Exception as e:
        logging.error(f"Error in trial: {str(e)}", exc_info=True)
        print(f"Error in trial: {str(e)}")
        return float('-inf')
    
    finally:
        cleanup_trial_variables()


# -------------------------------------------------------------------------
# Run Optuna 
# -------------------------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    import plotly.graph_objects as go
    from itables import init_notebook_mode
    init_notebook_mode(all_interactive=True) # makes all dataframes pretty
    import os 
    from tqdm import tqdm
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module='optuna')
    import talib
    ######################
    # LIMIT CPU UTILIZATION
    ######################
    limit_cpu_cores(0.9)

    ######################
    # SET WORKING DIRECTORY / FILE NAMES
    ######################
    # Usage
    working_dir = r'C:\Users\micha\myhome\algo\artifacts\optuna'
    optuna_trials_dir = r'C:\Users\micha\myhome\algo\artifacts\optuna\optuna_pdfs'
    set_working_directory(working_dir)

    # this is where all model
    main_model_path = 'my_model_info.pkl'

    ######################
    # SET TICKER LIST 
    ######################
    # set ticker list 
    optuna_ticker_list = ['TSLA']

    data_dict = load_and_process_data_dict(selected_symbols=optuna_ticker_list)

    #################################
    # CALCULATED PRECOMPUTED FEATURES
    #################################
    data_dict = calculate_precomputed_features(data_dict=data_dict, optuna_ticker_list=optuna_ticker_list)
    print(data_dict[optuna_ticker_list[0]].columns)

    # Run Optuna
    study = optuna.create_study(direction='maximize')
    try:
        study.optimize(lambda trial: objective(trial, data_dict), n_trials=400, catch=(Exception,))
        #study.optimize(objective, data_dict=data_dict, n_trials=400, catch=(Exception,))  

    except KeyboardInterrupt:
        print("\nStudy interrupted by user. Saving progress...")

    except MemoryError:
        print("\nMemory Error encountered in main study!")
        gc.collect()

    except Exception as e:
        print(f"\nStudy failed with error: {str(e)}")
        raise

    finally:
        # Save progress even if interrupted
        if hasattr(study, 'trials'):
            try:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                study_backup_name = f'study_backup_{timestamp}.pkl'
                joblib.dump(study, study_backup_name)
                print(f"Progress saved to {study_backup_name}")
            except Exception as e:
                print(f"Error saving progress: {str(e)}")

    print("Best Hyperparameters:")
    print(study.best_params)
    print("Best Uplift:")
    print(study.best_value)