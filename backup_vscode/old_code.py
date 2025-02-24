# import numpy as np
# import pandas as pd
# import talib
# from dataclasses import dataclass
# from typing import Dict, List, Optional
# from my_functions import calc_intraday_atr
# import matplotlib.pyplot as plt 
# from matplotlib.figure import Figure

# @dataclass
# class BacktestResults:
#     """Container for backtest results"""
#     trades_df: pd.DataFrame
#     summary_metrics: Dict
#     symbol_metrics: pd.DataFrame
#     equity_curves: Dict[str, pd.DataFrame]
#     equity_curve_plots: Optional[List[Figure]] = None

# class SymbolBacktester:
#     def __init__(self, initial_capital: float = 10000, 
#                 atr_period_for_stoploss: int = 28,  # Changed from atr_period
#                 stop_loss_adjust: float = 1.0, 
#                 stop_loss_adjust_sma_period: int = 200, 
#                 lookback_period_for_position_size: int = 50,  # For Kelly calculation
#                 ema_period: int = 20,
#                 kelly_multiplier: float = 3.0,  # Add this parameter
#                 debug: bool = False,
#                 overnight_position_size: float = 0.5,
#                 use_simpler_position_sizing: bool = True,
#                 use_performance_scaling: bool = True,
#                 stop_loss_scale_coeff: float = 25.0
#                 ):
#         self.initial_capital = initial_capital
#         self.atr_period_for_stoploss = atr_period_for_stoploss  # Updated variable name
#         self.stop_loss_adjust = stop_loss_adjust
#         self.stop_loss_adjust_sma_period = stop_loss_adjust_sma_period
#         self.portfolio_timestamps = []
#         self.portfolio_value = []
#         self.debug = debug

#         self.lookback_period_for_position_size = lookback_period_for_position_size
#         self.ema_period = ema_period
#         self.kelly_multiplier = kelly_multiplier
#         self.class_metrics = {}
#         self.current_position_size = 0.0
#         self.current_entry_price = 0.0
#         self.overnight_position_size = overnight_position_size
#         self.use_simpler_position_sizing = use_simpler_position_sizing
#         self.use_performance_scaling = use_performance_scaling
#         self.stop_loss_scale_coeff = stop_loss_scale_coeff
#         # Initialize historical_trades with all expected columns
#         self.historical_trades = pd.DataFrame(columns=[
#             'symbol', 'buy_timestamp', 'sell_timestamp', 'buy_price', 'sell_price',
#             'position_size', 'return', 'return_percentage', 'capital', 
#             'hold_time_hours', 'predicted_class', 'is_partial'
#         ])

#     import numpy as np
#     import pandas as pd

#     def calculate_kelly_fraction(self, pred_class: int, current_idx: int, market_df: pd.DataFrame, 
#                                 kelly_debug: bool = False) -> float:
#         """
#         Calculate Kelly fraction based on historical performance with option for simpler position sizing.
#         This version uses an exponential moving average (EMA) instead of a simple average for both the 
#         simpler sizing and the Kelly-based sizing logic.
#         """
#         # Basic checks remain the same
#         if 'predicted_class' not in self.historical_trades.columns:
#             if self.debug or kelly_debug:
#                 print("[WARNING] 'predicted_class' column missing in self.historical_trades. Returning default Kelly=0.5")
#             return 0.5

#         if self.historical_trades.empty:
#             if self.debug or kelly_debug:
#                 print("[WARNING] 'historical_trades' is empty. Returning default Kelly=0.25")
#             return 0.25

#         # Get historical trades for this class, limited to lookback
#         cutoff_time = market_df.index[current_idx]
#         historical_class_trades = self.historical_trades[
#             (self.historical_trades['predicted_class'] == pred_class) & 
#             (self.historical_trades['sell_timestamp'] < cutoff_time)
#         ].tail(self.lookback_period_for_position_size)

#         if kelly_debug and current_idx < 2000:
#             print(f"\nPosition Sizing Calculation for row {current_idx}:")
#             print(f"Predicted Class: {pred_class}")
#             print(f"Number of historical trades: {len(historical_class_trades)}")
        
#         if len(historical_class_trades) < 10:
#             if kelly_debug and current_idx < 2000:
#                 print("Insufficient historical trades - using default = 0.25")
#             return 0.25

#         # Sort by 'sell_timestamp' so EMA follows chronological order
#         historical_class_trades = historical_class_trades.sort_values('sell_timestamp')

#         # ============ Simpler Position Sizing (EMA) ============
#         # Instead of simple average over last N trades, we take the final value
#         # of the EMA on return_percentage.
#         ewm_return = historical_class_trades['return_percentage'].ewm(
#             span=len(historical_class_trades), 
#             adjust=False
#         ).mean()
#         # This is the final EMA-based "average" return for the class
#         avg_return_ewm = ewm_return.iloc[-1] / 100.0  # convert % to decimal

#         if self.use_simpler_position_sizing:
#             # If negative, position_size=0.01; otherwise 1.0
#             if avg_return_ewm < 0:
#                 position_size = 0.01
#             else:
#                 position_size = 1.0

#             if kelly_debug and current_idx < 2000:
#                 print("Using simpler position sizing:")
#                 print(f"EMA-based Average Return: {avg_return_ewm:.4f}")
#                 print(f"Position Size: {position_size:.2f}")

#             return position_size

#         # ============ Kelly-based Sizing (Using EMA) ============
#         # win_prob: 0 or 1 for each trade, then an EMA
#         winners_mask = historical_class_trades['return'] > 0
#         # 0/1 series the same length as historical_class_trades
#         win_01_series = winners_mask.astype(float)
#         ewm_win_prob = win_01_series.ewm(
#             span=len(win_01_series), 
#             adjust=False
#         ).mean()
#         win_prob_ewm = ewm_win_prob.iloc[-1]  # fraction in [0..1]

#         # Average win (in decimal form)
#         pos_trades = historical_class_trades[winners_mask]
#         if len(pos_trades) > 0:
#             ewm_avg_win = pos_trades['return_percentage'].ewm(
#                 span=len(pos_trades),
#                 adjust=False
#             ).mean().iloc[-1] / 100.0
#         else:
#             ewm_avg_win = 0.0

#         # Average loss (in decimal form)
#         losers_mask = historical_class_trades['return'] <= 0
#         neg_trades = historical_class_trades[losers_mask]
#         if len(neg_trades) > 0:
#             ewm_avg_loss = abs(
#                 neg_trades['return_percentage'].ewm(
#                     span=len(neg_trades),
#                     adjust=False
#                 ).mean().iloc[-1] 
#                 / 100.0
#             )
#         else:
#             ewm_avg_loss = 0.0

#         if kelly_debug and current_idx < 2000:
#             print(f"Win Probability (EMA): {win_prob_ewm:.3f}")
#             print(f"Average Win (EMA): {ewm_avg_win:.3f}")
#             print(f"Average Loss (EMA): {ewm_avg_loss:.3f}")

#         # If there's literally zero average loss, revert to old fallback
#         if ewm_avg_loss == 0:
#             if kelly_debug and current_idx < 2000:
#                 print("Average loss is 0 - using default kelly = 0.75")
#             return 0.75

#         # Kelly formula: k = p - ( (1 - p) / (avg_win / avg_loss) )
#         kelly_raw = win_prob_ewm - ((1 - win_prob_ewm) / (ewm_avg_win / ewm_avg_loss))

#         # Additionally scale by (1 + avg_return/multiplier)^2, as in original code
#         return_scalar = np.power(1.0 + (avg_return_ewm / self.kelly_multiplier), 2)
#         kelly = kelly_raw * return_scalar

#         # Clip between 0.01 and 1.0
#         kelly = max(0.01, min(kelly, 1.0))

#         if kelly_debug and current_idx < 2000:
#             print(f"Raw Kelly (EMA-based): {kelly_raw:.3f}")
#             print(f"Scaled Kelly: {kelly:.3f}")

#         return kelly


#     def calculate_trailing_metrics(self, market_df: pd.DataFrame, current_idx: int) -> Dict:
#         """Calculate trailing metrics up to current_idx"""
#         metrics = {}
#         cutoff_time = market_df.index[current_idx]
        
#         for pred_class in self.historical_trades['predicted_class'].unique():
#             class_trades = self.historical_trades[
#                 (self.historical_trades['predicted_class'] == pred_class) &
#                 (self.historical_trades['sell_timestamp'] < cutoff_time)
#             ]
            
#             if len(class_trades) == 0:
#                 continue
                
#             daily_returns = class_trades.groupby('buy_timestamp')['return_percentage'].mean()
            
#             # Create series aligned with market data up to current_idx
#             continuous_series = pd.Series(index=market_df.index[:current_idx+1], dtype=float)
#             continuous_series[daily_returns.index] = daily_returns
            
#             continuous_series = continuous_series.ffill()
#             ema = continuous_series.ewm(span=self.ema_period, adjust=False).mean()
            
#             metrics[pred_class] = ema
            
#         return metrics


#     def calculate_atr_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
#         if self.debug:
#             print(f"[DEBUG] Calculating ATR metrics on {len(df)} rows.")

#         df = df.copy()

#         # Ensure the input arrays are of type float64
#         high = df['high_raw'].astype(np.float64).values
#         low = df['low_raw'].astype(np.float64).values
#         close = df['close_raw'].astype(np.float64).values

#         # Compute ATR
#         #atr_28 = talib.ATR(high, low, close, timeperiod=self.atr_period)

#         # Suppose your DataFrame has columns: 'High', 'Low', 'Close'
#         # and you want to ignore overnight gaps. You can call:
            
#         atr_28 = calc_intraday_atr(
#             df,
#             col_high='high_raw',
#             col_low='low_raw',
#             col_close='close_raw',
#             atr_period=self.atr_period_for_stoploss 
#         )
#         # Compute shorter-term SMA
#         close_sma = talib.SMA(close, timeperiod=self.atr_period_for_stoploss)
#         # Compute longer-term SMA
#         close_sma_200 = talib.SMA(close, timeperiod=self.stop_loss_adjust_sma_period)

#         # Ensure no NaNs in SMA calculations by handling edge cases
#         close_above_sma_200 = (close > close_sma_200).astype(int)

#         # Use the SMA in the denominator instead of raw close
#         close_price_atr_28_percent_change = ((atr_28 / close_sma) * 100 * -1.0)

#         # Assign computed metrics back to the DataFrame
#         df['atr_28'] = atr_28
#         df['close_sma'] = close_sma
#         df['close_sma_200'] = close_sma_200
#         df['close_above_sma_200'] = close_above_sma_200
#         df['close_price_atr_28_percent_change'] = close_price_atr_28_percent_change

#         # Drop rows with NaN values in critical columns
#         return df.dropna(subset=['atr_28', 'close_price_atr_28_percent_change'])

#     def process_valid_buys(self, buys: np.ndarray, sells: np.ndarray) -> np.ndarray:
#         if self.debug:
#             print("[DEBUG] Processing valid buys.")

#         valid_buys = buys.copy()
#         last_buy_index = -1
#         last_sell_index = -1

#         for i in range(len(buys)):
#             if buys[i] == 1:
#                 # Check conditions for a valid buy
#                 if last_sell_index == -1 or last_buy_index >= last_sell_index:
#                     if last_buy_index != -1 and (i - last_buy_index <= 1000):
#                         valid_buys[i] = 0
#                     else:
#                         last_buy_index = i
#                 else:
#                     last_buy_index = i

#             if sells[i] == 1:
#                 last_sell_index = i

#         return valid_buys

#     def calculate_stop_losses(
#         self, 
#         close_prices: np.ndarray, 
#         close_price_atr_28_percent_change: np.ndarray, 
#         close_above_sma_200: np.ndarray,
#         valid_buys: np.ndarray, 
#         sells: np.ndarray, 
#         atr_coeff: float,
#         historical_trades: pd.DataFrame,
#         lookback_period: int,
#         use_performance_scaling: bool = True,  # Flag to choose scaling method
#         min_trades_required: int = 10,
#         stop_loss_scale_coeff: float = 1.0,
#         max_scale_up: float = 2.0,
#         min_scale_down: float = 0.5
#     ):
#         if self.debug:
#             scaling_method = "performance-based" if use_performance_scaling else "SMA-based"
#             print(f"[DEBUG] Calculating stop losses using {scaling_method} scaling.")

#         # Initialize arrays
#         sell_stop_loss = np.zeros_like(valid_buys)
#         position_open = False
#         max_price_since_buy = 0.0

#         # Calculate base thresholds
#         base_thresholds = close_price_atr_28_percent_change * atr_coeff

#         # Choose scaling method based on flag
#         if use_performance_scaling:
#             # Performance-based scaling logic with EMA
#             for index in range(len(valid_buys)):
#                 current_price = close_prices[index]
#                 scale_multiplier = 1.0

#                 if historical_trades is not None and not historical_trades.empty:
#                     current_time = historical_trades.index[index]

#                     # Extract all trades before the current time, sorted by sell_timestamp
#                     recent_trades = historical_trades[
#                         historical_trades['sell_timestamp'] < current_time
#                     ].copy()
#                     recent_trades.sort_values('sell_timestamp', inplace=True)

#                     # Take the tail of that set (lookback_period trades) then compute EMA
#                     recent_trades = recent_trades.tail(lookback_period)
                    
#                     if len(recent_trades) >= min_trades_required:
#                         # Compute an exponential moving average of return_percentage
#                         ewm_return = recent_trades['return_percentage'].ewm(
#                             span=len(recent_trades), 
#                             adjust=False
#                         ).mean()

#                         # Use the final EMA value as the "average" return
#                         avg_return_ewm = ewm_return.iloc[-1]

#                         scale_multiplier = 1.0 + (avg_return_ewm * stop_loss_scale_coeff / 100.0)
#                         scale_multiplier = np.clip(scale_multiplier, min_scale_down, max_scale_up)

#                         if self.debug:
#                             print(f"[DEBUG] Index {index}: EMA-based trailing avg return: {avg_return_ewm:.2f}%, "
#                                 f"Scale multiplier: {scale_multiplier:.2f}")

#                 threshold = base_thresholds[index] * scale_multiplier

#                 # Position management logic (unchanged)...
#                 if not position_open and valid_buys[index] == 1:
#                     position_open = True
#                     max_price_since_buy = current_price

#                 if position_open:
#                     if current_price > max_price_since_buy:
#                         max_price_since_buy = current_price

#                     percent_change = (current_price - max_price_since_buy) / max_price_since_buy * 100
#                     if percent_change < threshold:
#                         sell_stop_loss[index] = 1
#                         position_open = False

#                 if position_open and sells[index] == 1:
#                     position_open = False


#         else:
#             # Original SMA-based scaling logic
#             multipliers = np.where(close_above_sma_200 == 1, self.stop_loss_adjust, 1.0)
#             stoploss_thresholds = base_thresholds * multipliers

#             for index in range(len(valid_buys)):
#                 current_price = close_prices[index]
#                 threshold = stoploss_thresholds[index]

#                 if not position_open and valid_buys[index] == 1:
#                     position_open = True
#                     max_price_since_buy = current_price

#                 if position_open:
#                     if current_price > max_price_since_buy:
#                         max_price_since_buy = current_price

#                     percent_change = (current_price - max_price_since_buy) / max_price_since_buy * 100
#                     if percent_change < threshold:
#                         sell_stop_loss[index] = 1
#                         position_open = False

#                 if position_open and sells[index] == 1:
#                     position_open = False

#         return sell_stop_loss

#     def execute_trades(self, timestamps: np.ndarray, close_prices: np.ndarray, 
#             valid_buys: np.ndarray, combined_sells: np.ndarray, 
#             symbol: str, predicted_classes: np.ndarray, 
#             market_df: pd.DataFrame, kelly_debug: bool = False) -> pd.DataFrame:
#         if self.debug:
#             print(f"[DEBUG] Executing trades for symbol: {symbol}")

#         trades_list = []
#         capital = self.initial_capital
#         position_open = False
#         predicted_class_at_buy = None
#         buy_timestamp = None
        
#         # Reset position tracking at start
#         self.current_position_size = 0.0
#         self.current_entry_price = 0.0

#         # Get the highest class number
#         highest_class = max([int(col.split('_')[-1]) for col in market_df.columns if col.startswith('prediction_raw_class_')])
#         highest_class_col = f'prediction_raw_class_{highest_class}'

#         for index in range(len(close_prices)):
#             current_price = close_prices[index]
#             timestamp = timestamps[index]
#             current_time = pd.Timestamp(timestamp).time()

#             # At 15:45, check for special selling logic
#             if current_time.hour == 15 and current_time.minute == 45:
#                 # If a sell signal is active, let the full-sell logic handle closing the position.
#                 if combined_sells[index]:
#                     pass  # Do nothing here so that the full sell logic below executes.
#                 else:
#                     # No sell signal: limit the position size to a maximum of 50% of capital.
#                     max_allowed = capital * self.overnight_position_size
#                     if self.current_position_size > max_allowed:
#                         excess_position = self.current_position_size - max_allowed
#                         profit = (current_price - self.current_entry_price) / self.current_entry_price * excess_position
#                         capital += profit

#                         trade = {
#                             'symbol': symbol,
#                             'buy_timestamp': buy_timestamp,
#                             'sell_timestamp': timestamp,
#                             'buy_price': self.current_entry_price,
#                             'sell_price': current_price,
#                             'position_size': excess_position,  # Partial reduction trade
#                             'return': profit,
#                             'return_percentage': (current_price - self.current_entry_price) / self.current_entry_price * 100,
#                             'capital': capital,
#                             'hold_time_hours': (timestamp - buy_timestamp) / np.timedelta64(1, 'h'),
#                             'predicted_class': predicted_class_at_buy,
#                             'highest_class_probability': market_df.iloc[index][highest_class_col],
#                             'is_partial': True
#                         }
#                         trades_list.append(trade)

#                         self.current_position_size -= excess_position

#                         if self.debug:
#                             print(f"\n[DEBUG] At {timestamp}:")
#                             print(f"Original position size: {self.current_position_size + excess_position:.2f}")
#                             print(f"Excess position reduced: {excess_position:.2f}")
#                             print(f"New position size: {self.current_position_size:.2f}")
#                             print(f"Current capital: {capital:.2f}")

#                         self.historical_trades = pd.DataFrame(trades_list)
                        
#                         # Skip further processing for this timestamp so that a full sell isn't also executed.
#                         continue


#             # Regular buy logic
#             if not position_open and valid_buys[index] == 1:
#                 predicted_class_at_buy = predicted_classes[index]
                
#                 # Calculate Kelly with available historical trades so far
#                 kelly_fraction = self.calculate_kelly_fraction(
#                     pred_class=predicted_class_at_buy, 
#                     current_idx=index, 
#                     market_df=market_df, 
#                     kelly_debug=kelly_debug
#                 )
                
#                 # Open new position
#                 position_open = True
#                 buy_timestamp = timestamp
#                 self.current_entry_price = current_price
#                 self.current_position_size = capital * kelly_fraction
#                 highest_class_prob_at_buy = market_df.iloc[index][highest_class_col]

#             # Regular sell logic
#             elif position_open and combined_sells[index]:
#                 # Close entire remaining position
#                 profit = (current_price - self.current_entry_price) / self.current_entry_price * self.current_position_size
#                 capital += profit
                
#                 trade = {
#                     'symbol': symbol,
#                     'buy_timestamp': buy_timestamp,
#                     'sell_timestamp': timestamp,
#                     'buy_price': self.current_entry_price,
#                     'sell_price': current_price,
#                     'position_size': self.current_position_size,
#                     'return': profit,
#                     'return_percentage': (current_price - self.current_entry_price) / self.current_entry_price * 100,
#                     'capital': capital,
#                     'hold_time_hours': (timestamp - buy_timestamp) / np.timedelta64(1, 'h'),
#                     'predicted_class': predicted_class_at_buy,
#                     'highest_class_probability': highest_class_prob_at_buy,
#                     'is_partial': False  # Complete position close
#                 }
#                 trades_list.append(trade)
                
#                 # Reset position tracking
#                 self.current_position_size = 0.0
#                 self.current_entry_price = 0.0
#                 position_open = False
                
#                 # Update historical trades immediately
#                 self.historical_trades = pd.DataFrame(trades_list)

#             # Update portfolio tracking
#             if len(self.portfolio_timestamps) == 0 or self.portfolio_timestamps[-1] != timestamp:
#                 self.portfolio_timestamps.append(timestamp)
#                 self.portfolio_value.append(capital)

#         # Final update of historical trades
#         if trades_list:
#             self.historical_trades = pd.DataFrame(trades_list)

#         return pd.DataFrame(trades_list) if trades_list else pd.DataFrame(columns=[
#             'symbol', 'buy_timestamp', 'sell_timestamp', 'buy_price', 'sell_price',
#             'position_size', 'return', 'return_percentage', 'capital', 
#             'hold_time_hours', 'predicted_class', 'highest_class_probability', 'is_partial'
#         ])


#     def calculate_metrics(self, trades_df: pd.DataFrame, symbol: str, 
#                         initial_price: float, final_price: float, df: pd.DataFrame) -> Dict:
#         if self.debug:
#             print(f"[DEBUG] Calculating metrics for symbol: {symbol}")

#         if trades_df.empty:
#             # Keep existing empty DataFrame handling...
#             return {
#                 'symbol': symbol,
#                 'total_returns_percentage': 0,
#                 'win_loss_ratio': 0,
#                 'average_percent_return': 0,
#                 'average_hold_time_hours': 0,
#                 'buy_and_hold_return_percentage': ((final_price - initial_price) / initial_price) * 100,
#                 'number_of_trades': 0,
#                 'buy_and_hold_hold_time_hours': (df.index[-1] - df.index[0]).total_seconds() / 3600.0,
#                 'trading_hold_time_hours': 0,
#                 'sharpe_ratio': 0,
#                 'overnight_trades_count': 0,
#                 'overnight_trades_return_percentage': 0,
#                 'partial_trades_count': 0
#             }

#         # Separate complete and partial trades
#         complete_trades = trades_df[~trades_df['is_partial']]
#         partial_trades = trades_df[trades_df['is_partial']]

#         # Calculate metrics for all trades
#         winning_trades = (trades_df['return'] > 0).sum()
#         losing_trades = (trades_df['return'] <= 0).sum()
#         win_loss_ratio = winning_trades / losing_trades if losing_trades > 0 else float('inf')

#         # Calculate overnight trade metrics
#         overnight_trades = trades_df[
#             (trades_df['sell_timestamp'].dt.time == pd.Timestamp('15:45').time()) &
#             trades_df['is_partial']
#         ]

#         # Buy and hold metrics remain the same
#         buy_and_hold_hold_time_hours = (df.index[-1] - df.index[0]).total_seconds() / 3600.0
#         trading_hold_time_hours = trades_df['hold_time_hours'].sum()

#         # Calculate Sharpe Ratio (existing code remains the same)
#         if len(self.portfolio_value) > 1:
#             portfolio_df = pd.DataFrame({
#                 'timestamp': self.portfolio_timestamps,
#                 'capital': self.portfolio_value
#             })
#             portfolio_df.set_index('timestamp', inplace=True)
#             portfolio_df = portfolio_df.sort_index()

#             duplicates_portfolio = portfolio_df.index.duplicated().sum()
#             assert duplicates_portfolio == 0, f"Symbol {symbol} has {duplicates_portfolio} duplicate timestamps in portfolio data."

#             portfolio_daily = portfolio_df.resample('D').ffill()
#             portfolio_daily['daily_return'] = portfolio_daily['capital'].pct_change()
#             portfolio_daily.dropna(inplace=True)

#             mean_daily_return = portfolio_daily['daily_return'].mean()
#             std_daily_return = portfolio_daily['daily_return'].std()
#             sharpe_ratio = (mean_daily_return / std_daily_return * np.sqrt(252)) if std_daily_return != 0 else 0
#         else:
#             sharpe_ratio = 0

#         # Calculate average position sizes (considering both partial and complete trades)
#         avg_position_size = trades_df['position_size'].mean() / self.initial_capital * 100
#         position_size_std = trades_df['position_size'].std() / self.initial_capital * 100
#         avg_kelly = avg_position_size / 100

#         metrics = {
#             'symbol': symbol,
#             'total_returns_percentage': (trades_df['capital'].iloc[-1] - self.initial_capital) / self.initial_capital * 100,
#             'win_loss_ratio': win_loss_ratio,
#             'average_percent_return': trades_df['return_percentage'].mean(),
#             'average_hold_time_hours': trades_df['hold_time_hours'].mean(),
#             'buy_and_hold_return_percentage': ((final_price - initial_price) / initial_price) * 100,
#             'number_of_trades': len(trades_df),
#             'buy_and_hold_hold_time_hours': buy_and_hold_hold_time_hours,
#             'trading_hold_time_hours': trading_hold_time_hours,
#             'sharpe_ratio': sharpe_ratio,

#             # Position sizing metrics
#             'average_position_size': avg_position_size,
#             'position_size_std': position_size_std,
#             'average_kelly_fraction': avg_kelly,

#             # New metrics for partial/overnight trading
#             'complete_trades_count': len(complete_trades),
#             'partial_trades_count': len(partial_trades),
#             'overnight_trades_count': len(overnight_trades),
#             'overnight_trades_return_percentage': overnight_trades['return_percentage'].mean() if not overnight_trades.empty else 0,
#             'average_overnight_position_size': overnight_trades['position_size'].mean() / self.initial_capital * 100 if not overnight_trades.empty else 0,

#             # Per-class metrics (now includes partial trade information)
#             'class_metrics': trades_df.groupby(['predicted_class', 'is_partial']).agg({
#                 'return_percentage': ['mean', 'std'],
#                 'position_size': ['mean', 'std'],
#                 'return': lambda x: (x > 0).mean()  # win rate
#             }).to_dict()
#         }

#         return metrics

#     def get_active_signals(self, current_time, df_trades, debug=False):
#         """
#         Determine which signal set to use based on recent performance.
#         Returns tuple of (buy_col, sell_col)
#         """
#         if df_trades.empty:
#             if debug:
#                 print("[DEBUG] No historical trades, using regular signals")
#             return 'buy_final', 'sell'
        
#         # Grab recent trades and sort them by sell_timestamp
#         recent_trades = df_trades[df_trades['sell_timestamp'] < current_time].copy()
#         recent_trades.sort_values('sell_timestamp', inplace=True)
#         recent_trades = recent_trades.tail(self.lookback_period_for_position_size)

#         # If we have no recent trades, default to 0 for the average
#         if recent_trades.empty:
#             avg_return_ewm = 0
#         else:
#             # Compute an EMA of the return_percentage column
#             ewm_return = recent_trades['return_percentage'].ewm(
#                 span=len(recent_trades),
#                 adjust=False
#             ).mean()
#             # Use the final EMA value
#             avg_return_ewm = ewm_return.iloc[-1]

#         if debug:
#             print(f"\n[DEBUG] Signal Selection at {current_time}:")
#             print(f"Recent trades count: {len(recent_trades)}")
#             print(f"EMA-based average return: {avg_return_ewm:.2f}%")

#         if avg_return_ewm < 0:
#             if debug:
#                 print("Using short signals due to negative performance")
#             return 'buy_final_short', 'sell_short'
#         else:
#             if debug:
#                 print("Using regular signals due to positive/neutral performance")
#             return 'buy_final', 'sell'


#     def backtest_symbol(self, df: pd.DataFrame, symbol: str, atr_coeff: float, kelly_debug: bool = False) -> tuple:
#         if self.debug:
#             print(f"[DEBUG] Backtesting symbol: {symbol}")

#         # Reset historical trades at start of new symbol
#         self.historical_trades = pd.DataFrame(columns=[
#             'symbol', 'buy_timestamp', 'sell_timestamp', 'buy_price', 'sell_price',
#             'position_size', 'return', 'return_percentage', 'capital',
#             'hold_time_hours', 'predicted_class'
#         ])

#         df = self.calculate_atr_metrics(df)
        
#         # Check for both regular and short signal columns
#         if ('buy_final' not in df.columns or 'sell' not in df.columns or 
#             'buy_final_short' not in df.columns or 'sell_short' not in df.columns):
#             raise ValueError(f"Required buy/sell columns (regular and short) missing for symbol {symbol}.")

#         # Initialize arrays
#         timestamps = df.index.to_numpy()
#         buys = np.zeros(len(df), dtype=np.int8)
#         sells = np.zeros(len(df), dtype=np.int8)

#         # Determine buy/sell signals for each timestamp based on performance
#         for i in range(len(df)):
#             buy_col, sell_col = self.get_active_signals(
#                 timestamps[i], 
#                 self.historical_trades, 
#                 debug=self.debug
#             )
#             buys[i] = df[buy_col].iloc[i]
#             sells[i] = df[sell_col].iloc[i]

#         valid_buys = self.process_valid_buys(buys, sells)
        
#         close_prices = df['close_raw'].astype(np.float64).values
#         close_price_atr_28_percent_change = df['close_price_atr_28_percent_change'].astype(np.float64).values
#         close_above_sma_200 = df['close_above_sma_200'].astype(np.int8).values

#         sell_stop_loss = self.calculate_stop_losses(
#             close_prices, 
#             close_price_atr_28_percent_change, 
#             close_above_sma_200, 
#             valid_buys, 
#             sells, 
#             atr_coeff,
#             historical_trades=self.historical_trades,
#             lookback_period=self.lookback_period_for_position_size,
#             use_performance_scaling=self.use_performance_scaling,
#             stop_loss_scale_coeff=self.stop_loss_scale_coeff
#         )

#         combined_sells = np.logical_or(sells == 1, sell_stop_loss == 1)

#         # After processing buys/sells, add:
#         predicted_classes = df['predicted_class_moving_avg'].values

#         trade_data = self.execute_trades(
#             timestamps, close_prices, valid_buys, combined_sells, 
#             symbol, predicted_classes, df, kelly_debug=kelly_debug
#         )

#         trades_df = trade_data if not trade_data.empty else pd.DataFrame()

#         metrics = self.calculate_metrics(
#             trades_df,
#             symbol,
#             close_prices[0],
#             close_prices[-1],
#             df
#         )

#         if self.debug:
#             print(f"[DEBUG] Calculated metrics for {symbol}")
#             print(metrics)

#         df['sell_stop_loss'] = sell_stop_loss
        
#         # Store which signals were used at each timestamp
#         if self.debug:
#             df['signals_used'] = ['regular' if r >= 0 else 'short' 
#                                 for r in trades_df['return_percentage']]

#         # NEW: Store trailing metrics for this symbol
#         if not trades_df.empty:
#             self.class_metrics[symbol] = self.calculate_trailing_metrics(df, len(df)-1)

#         # Collect equity curve
#         if len(self.portfolio_timestamps) > 0 and len(self.portfolio_value) > 0:
#             equity_curve_df = pd.DataFrame({
#                 'timestamp': self.portfolio_timestamps,
#                 'capital': self.portfolio_value
#             })
#             equity_curve_df['symbol'] = symbol
#             equity_curve_df.sort_values('timestamp', inplace=True)
#         else:
#             equity_curve_df = pd.DataFrame()

#         return trades_df, metrics, df, equity_curve_df
    
# def calculate_buy_hold_equity(df, initial_capital):
#     """Calculate buy & hold equity curve using all price changes"""
#     equity_points = []
#     current_capital = initial_capital
#     first_price = df.iloc[0]['close_raw']
    
#     for timestamp, row in df.iterrows():
#         current_price = row['close_raw']
#         current_capital = initial_capital * (current_price / first_price)
#         equity_points.append((timestamp, current_capital))
    
#     return pd.DataFrame(equity_points, columns=['timestamp', 'capital'])

# def calculate_daily_equity(df, initial_capital):
#     daily_returns = []
#     capital = initial_capital
    
#     for date, group in df.groupby(df.index.date):
#         first_price = group.iloc[0]['close_raw']
#         last_price = group.iloc[-1]['close_raw']
#         daily_return = (last_price / first_price) - 1
#         capital *= (1 + daily_return)
#         daily_returns.append((group.index[-1], capital))
    
#     return pd.DataFrame(daily_returns, columns=['timestamp', 'capital'])

# def run_backtest(df: pd.DataFrame, atr_coeff: float = 1.0, initial_capital: float = 10000, 
#                 debug: bool = False, kelly_debug: bool = False,  # Add parameter
#                 atr_period_for_stoploss: int = 28,
#                 stop_loss_adjust: float = 1.0, 
#                 stop_loss_adjust_sma_period: int = 200,
#                 kelly_multiplier: float = 3.0,  
#                 overnight_position_size: float = 0.5,
#                 lookback_period_for_position_size: int = 50,
#                 stop_loss_scale_coeff: float = 25.0,
#                 use_performance_scaling: bool = True,
#                 use_simpler_position_sizing=True,
#                 return_equity_curves: bool = False) -> BacktestResults:
#     if debug:
#         print(f"[DEBUG] Starting run_backtest with atr_coeff={atr_coeff}, initial_capital={initial_capital}")

#     all_trades = []
#     all_metrics = []
#     processed_dfs = []
#     equity_curves = {}

#     # Iterate over each symbol and process them independently
#     symbols = df['symbol'].unique()
#     for symbol in symbols:
#         if debug:
#             print(f"[DEBUG] Processing symbol: {symbol}")

#         # Create a new SymbolBacktester instance for each symbol with the new parameters
#         backtester = SymbolBacktester(
#             initial_capital=initial_capital, 
#             debug=debug,
#             atr_period_for_stoploss=atr_period_for_stoploss,
#             stop_loss_adjust=stop_loss_adjust,
#             stop_loss_adjust_sma_period=stop_loss_adjust_sma_period,
#             lookback_period_for_position_size=lookback_period_for_position_size,
#             stop_loss_scale_coeff=stop_loss_scale_coeff,
#             ema_period=20,
#             use_performance_scaling=use_performance_scaling,
#             kelly_multiplier=kelly_multiplier,
#             overnight_position_size=overnight_position_size,
#             use_simpler_position_sizing=use_simpler_position_sizing
#         )

#         symbol_df = df[df['symbol'] == symbol].copy()
#         symbol_df.sort_index(inplace=True)

#         # Check for duplicate timestamps in input data for this symbol
#         duplicates_input = symbol_df.index.duplicated().sum()
#         assert duplicates_input == 0, f"Symbol {symbol} has {duplicates_input} duplicate timestamps in input data."

#         try:
#             trades_df, metrics, processed_df, equity_curve_df = backtester.backtest_symbol(
#                 symbol_df,
#                 symbol,
#                 atr_coeff,
#                 kelly_debug=kelly_debug  # Add this
#             )
#         except ValueError as e:
#             print(f"Skipping symbol {symbol} due to error: {e}")
#             continue
#         except Exception as e:
#             print(f"An unexpected error occurred for symbol {symbol}: {e}")
#             continue

#         if not trades_df.empty:
#             all_trades.append(trades_df)
#         all_metrics.append(metrics)
#         processed_dfs.append(processed_df)

#         if not equity_curve_df.empty:
#             equity_curves[symbol] = equity_curve_df

#     combined_trades = pd.concat(all_trades) if all_trades else pd.DataFrame()
#     metrics_df = pd.DataFrame(all_metrics) if all_metrics else pd.DataFrame()
#     processed_df = pd.concat(processed_dfs) if processed_dfs else pd.DataFrame()

#     if not metrics_df.empty and 'trading_hold_time_hours' in metrics_df.columns and not metrics_df['trading_hold_time_hours'].empty:
#         mean_hold_time_trading_hours = metrics_df['trading_hold_time_hours'].mean()
#     else:
#         mean_hold_time_trading_hours = 0

#     if not metrics_df.empty and 'buy_and_hold_hold_time_hours' in metrics_df.columns and not metrics_df['buy_and_hold_hold_time_hours'].empty:
#         mean_hold_time_buy_and_hold_hours = metrics_df['buy_and_hold_hold_time_hours'].mean()
#     else:
#         mean_hold_time_buy_and_hold_hours = 0

#     if not metrics_df.empty and 'total_returns_percentage' in metrics_df.columns:
#         average_total_return_percentage = metrics_df['total_returns_percentage'].mean()
#     else:
#         average_total_return_percentage = 0

#     if ('buy_and_hold_return_percentage' in metrics_df.columns and 
#         not metrics_df['buy_and_hold_return_percentage'].empty):
#         average_total_return_percentage_buy_and_hold = metrics_df['buy_and_hold_return_percentage'].mean()
#     else:
#         average_total_return_percentage_buy_and_hold = 0

#     return_per_trading_hour = (average_total_return_percentage / mean_hold_time_trading_hours) if mean_hold_time_trading_hours > 0 else 0
#     return_per_buy_and_hold_hour = (average_total_return_percentage_buy_and_hold / mean_hold_time_buy_and_hold_hours) if mean_hold_time_buy_and_hold_hours > 0 else 0
#     ratio_return_per_trading_hour = (return_per_trading_hour / return_per_buy_and_hold_hour) if return_per_buy_and_hold_hour > 0 else 0

#     if 'sharpe_ratio' in metrics_df.columns and not metrics_df['sharpe_ratio'].empty:
#         mean_sharpe_ratio = metrics_df['sharpe_ratio'].mean()
#     else:
#         mean_sharpe_ratio = 0

#     overall_metrics = {
#         'total_trades': len(combined_trades),
#         'average_total_return_percentage': average_total_return_percentage,
#         'average_trade_return': combined_trades['return_percentage'].mean() if not combined_trades.empty else 0,
#         'average_hold_time': combined_trades['hold_time_hours'].mean() if not combined_trades.empty else 0,
#         'winning_trades_percentage': (combined_trades['return'] > 0).mean() * 100 if not combined_trades.empty else 0,
#         'total_hold_time_buy_and_hold_hours': mean_hold_time_buy_and_hold_hours,
#         'total_hold_time_trading_hours': mean_hold_time_trading_hours,
#         'average_total_return_percentage_buy_and_hold': average_total_return_percentage_buy_and_hold,
#         'return_per_trading_hour': return_per_trading_hour,
#         'return_per_buy_and_hold_hour': return_per_buy_and_hold_hour,
#         'ratio_return_per_trading_hour': ratio_return_per_trading_hour,
#         'mean_sharpe_ratio': mean_sharpe_ratio
#     }

#     equity_curve_plots = []  # New list to store the plots
    
#     if return_equity_curves:
#         for symbol, equity_df in equity_curves.items():
#             symbol_df = df[df['symbol'] == symbol]
            
#             # Calculate curves
#             buy_hold_df = calculate_buy_hold_equity(symbol_df, initial_capital)
#             daily_equity_df = calculate_daily_equity(symbol_df, initial_capital)
            
#             fig, ax = plt.subplots(figsize=(10, 6))
            
#             # Plot all curves
#             ax.plot(equity_df['timestamp'], equity_df['capital'], 
#                     label='Strategy', color='blue')
#             ax.plot(buy_hold_df['timestamp'], buy_hold_df['capital'],
#                     label='Buy & Hold', color='green')
#             ax.plot(daily_equity_df['timestamp'], daily_equity_df['capital'],
#                     label='Daily Buy & Hold', color='red')
            
#             ax.set_title(f'Equity Curves - {symbol}')
#             ax.set_xlabel('Date')
#             ax.set_ylabel('Capital')
#             ax.legend()
#             plt.xticks(rotation=45)
#             plt.tight_layout()
#             equity_curve_plots.append(fig)

#     return BacktestResults(
#         trades_df=combined_trades,
#         summary_metrics=overall_metrics,
#         symbol_metrics=metrics_df,
#         equity_curves=equity_curves,
#         equity_curve_plots=equity_curve_plots if return_equity_curves else None  # Add new field
#     )




















#     #####################
#         # GARBAGE COLLECTION  
#         #####################
#         cleanup_memory()
        
#         #############################
#         # SPLIT DATA FOR PREDICTIONS
#         ##############################
#         # test data only 
#         df_features_master_for_predictions_test  = prepare_test_data(df_features_master_for_predictions, test_data_begin_timestamp)

#         # split into deciles 
#         (decile_1, decile_2, decile_3, decile_4, decile_5,
#         decile_6, decile_7, decile_8, decile_9, decile_10,
#         full_test_set) = split_into_deciles(df_features_master_for_predictions_test, verbose=True)

#         #############################
#         # ADD PREDICTIONS
#         ##############################
#         # df_predictions_decile_1, fig_ens_1 = process_predictions_in_batches(decile_1, batch_size=100000, model_path=main_model_path, debug=False, include_confusion_matrix=True, time_period="Decile 1", use_ensemble=use_ensemble)
#         # df_predictions_decile_2, fig_ens_2 = process_predictions_in_batches(decile_2, batch_size=100000, model_path=main_model_path, debug=False, include_confusion_matrix=True, time_period="Decile 2", use_ensemble=use_ensemble)
#         # df_predictions_decile_3, fig_ens_3 = process_predictions_in_batches(decile_3, batch_size=100000, model_path=main_model_path, debug=False, include_confusion_matrix=True, time_period="Decile 3", use_ensemble=use_ensemble)
#         # df_predictions_decile_4, fig_ens_4 = process_predictions_in_batches(decile_4, batch_size=100000, model_path=main_model_path, debug=False, include_confusion_matrix=True, time_period="Decile 4", use_ensemble=use_ensemble)
#         # df_predictions_decile_5, fig_ens_5 = process_predictions_in_batches(decile_5, batch_size=100000, model_path=main_model_path, debug=False, include_confusion_matrix=True, time_period="Decile 5", use_ensemble=use_ensemble)
#         # df_predictions_decile_6, fig_ens_6 = process_predictions_in_batches(decile_6, batch_size=100000, model_path=main_model_path, debug=False, include_confusion_matrix=True, time_period="Decile 6", use_ensemble=use_ensemble)
#         # df_predictions_decile_7, fig_ens_7 = process_predictions_in_batches(decile_7, batch_size=100000, model_path=main_model_path, debug=False, include_confusion_matrix=True, time_period="Decile 7", use_ensemble=use_ensemble)
#         # df_predictions_decile_8, fig_ens_8 = process_predictions_in_batches(decile_8, batch_size=100000, model_path=main_model_path, debug=False, include_confusion_matrix=True, time_period="Decile 8", use_ensemble=use_ensemble)
#         # df_predictions_decile_9, fig_ens_9 = process_predictions_in_batches(decile_9, batch_size=100000, model_path=main_model_path, debug=False, include_confusion_matrix=True, time_period="Decile 9", use_ensemble=use_ensemble)
#         # df_predictions_decile_10, fig_ens_10 = process_predictions_in_batches(decile_10, batch_size=100000, model_path=main_model_path, debug=False, include_confusion_matrix=True, time_period="Decile 10", use_ensemble=use_ensemble)
#         df_predictions_full_test, fig_ens_full_test = process_predictions_in_batches(full_test_set, batch_size=100000, model_path=main_model_path, debug=False, include_confusion_matrix=True, time_period="Full Test Set", use_ensemble=use_ensemble)


#         all_ensemble_figs = []
#         # all_ensemble_figs.append(fig_ens_1)
#         # all_ensemble_figs.append(fig_ens_2)
#         # all_ensemble_figs.append(fig_ens_3)
#         # all_ensemble_figs.append(fig_ens_4)
#         # all_ensemble_figs.append(fig_ens_5)
#         # all_ensemble_figs.append(fig_ens_6)
#         # all_ensemble_figs.append(fig_ens_7)
#         # all_ensemble_figs.append(fig_ens_8)
#         # all_ensemble_figs.append(fig_ens_9)
#         # all_ensemble_figs.append(fig_ens_10)
#         all_ensemble_figs.append(fig_ens_full_test)

#         #############################
#         # ADD BUY AND SELL SIGNALS 
#         ##############################
#         prediction_dfs = {
#             # 'decile_1': df_predictions_decile_1,
#             # 'decile_2': df_predictions_decile_2,
#             # 'decile_3': df_predictions_decile_3,
#             # 'decile_4': df_predictions_decile_4,
#             # 'decile_5': df_predictions_decile_5,
#             # 'decile_6': df_predictions_decile_6,
#             # 'decile_7': df_predictions_decile_7,
#             # 'decile_8': df_predictions_decile_8,
#             # 'decile_9': df_predictions_decile_9,
#             # 'decile_10': df_predictions_decile_10,
#             'full_test_set': df_predictions_full_test
#         }

#         df_with_signals = {
#             name: process_all_symbols(
#                 df=prediction_df,  
#                 hma_period = hma_period_for_class_spread,
#                 feature_suffix="",
#                 classes_for_spread=classes_for_spread,
#                 buy_signal_n_classes=num_classes_per_side,
#                 sell_signal_n_classes=num_classes_per_side,
#                 debug= False,
#                 use_class_spread=use_class_spread,
#                 use_adjusted_hma=True,
#                 hma_adjust_ma_period=stop_loss_adjust_sma_period, # just using the same period
#                 hma_adjust=hma_adjust,
#                 buy_time_cutoff=buy_cut_off,
#                 apply_930_logic=apply_930_logic,
#                 sell_at_end_of_day=sell_at_end_of_day,
#                 morning_hma_adjustment=morning_hma_adjustment 
#             )
#             for name, prediction_df in prediction_dfs.items()
#         }

#         # Unpack results into individual variables
#         # df_with_signals_decile_1 = df_with_signals['decile_1']
#         # df_with_signals_decile_2 = df_with_signals['decile_2']
#         # df_with_signals_decile_3 = df_with_signals['decile_3']
#         # df_with_signals_decile_4 = df_with_signals['decile_4']
#         # df_with_signals_decile_5 = df_with_signals['decile_5']
#         # df_with_signals_decile_6 = df_with_signals['decile_6']
#         # df_with_signals_decile_7 = df_with_signals['decile_7']
#         # df_with_signals_decile_8 = df_with_signals['decile_8']
#         # df_with_signals_decile_9 = df_with_signals['decile_9']
#         # df_with_signals_decile_10 = df_with_signals['decile_10']
#         df_with_signals_full_test_set = df_with_signals['full_test_set'] 

#         #############################
#         # BACKTESTING
#         ##############################
#         dfs = {
#             # 'decile_1': df_with_signals_decile_1,
#             # 'decile_2': df_with_signals_decile_2,
#             # 'decile_3': df_with_signals_decile_3,
#             # 'decile_4': df_with_signals_decile_4,
#             # 'decile_5': df_with_signals_decile_5,
#             # 'decile_6': df_with_signals_decile_6,
#             # 'decile_7': df_with_signals_decile_7,
#             # 'decile_8': df_with_signals_decile_8,
#             # 'decile_9': df_with_signals_decile_9,
#             # 'decile_10': df_with_signals_decile_10,
#             'full_test_set': df_with_signals_full_test_set
#         }

#         results = {
#             name: run_backtest(
#                 df=signal_df,  # Using the processed DataFrame with signals
#                 atr_coeff=atr_coeff_for_stoploss,
#                 initial_capital=10000,
#                 debug= False,
#                 kelly_debug= False,
#                 atr_period_for_stoploss=atr_period_for_stoploss,
#                 stop_loss_adjust=stop_loss_adjust,
#                 stop_loss_adjust_sma_period=stop_loss_adjust_sma_period,
#                 kelly_multiplier= kelly_multiplier,
#                 overnight_position_size=overnight_position_size, 
#                 lookback_period_for_position_size=lookback_period_for_position_size, 
#                 stop_loss_scale_coeff=stop_loss_scale_coeff,
#                 use_performance_scaling=use_performance_scaling,
#                 return_equity_curves=(name == 'full_test_set')
#             )
#             for name, signal_df in dfs.items()  # Changed variable name to be clearer
#         }

#         # results_decile_1 = results['decile_1']
#         # results_decile_2 = results['decile_2']
#         # results_decile_3 = results['decile_3']
#         # results_decile_4 = results['decile_4']
#         # results_decile_5 = results['decile_5']
#         # results_decile_6 = results['decile_6']
#         # results_decile_7 = results['decile_7']
#         # results_decile_8 = results['decile_8']
#         # results_decile_9 = results['decile_9']
#         # results_decile_10 = results['decile_10']
#         results_full_test_set = results['full_test_set']



# ######################
# # LABELS (MULTI-CLASS, SIMILAR TO KAYVON'S APPROACH)
# ######################

# # Set debug parameters
# debug_start_date = '2024-05-13'  # Modify these dates as needed
# debug_end_date = '2024-05-20'
# debug_symbol = 'QQQ'  # Symbol to debug
# debugging_mode = False

# # Drop rows where 'close' is NaN without resetting the index
# ticker_df_adjusted = ticker_df_adjusted.dropna(subset=['close'])

# import numpy as np
# import pandas as pd

# num_classes_per_side = num_classes_per_side

# ticker_df_adjusted['label'] = -999

# # Get debug indices if debugging
# if debugging_mode and symbol == debug_symbol:
#     date_mask = (ticker_df_adjusted.index >= debug_start_date) & \
#                 (ticker_df_adjusted.index <= debug_end_date)
#     debug_indices = np.where(date_mask)[0]
#     if len(debug_indices) > 0:
#         print(f"\nDebugging for {debug_symbol} in date range: {debug_start_date} to {debug_end_date}")
#         print(f"Number of samples in debug range: {len(debug_indices)}")

# def print_sample_debug(idx, date, close_prices, atr_percent_changes, boundaries_positive, 
#                     boundaries_negative, percent_changes, look_ahead_window, num_classes_per_side):
#     # Only print debug info if current symbol matches debug symbol
#     if symbol != debug_symbol:
#         return
        
#     print(f"\nAnalyzing {debug_symbol} sample at {date}:")
#     print(f"Close price: {close_prices[idx]:.2f}")
#     print(f"ATR %: {atr_percent_changes[idx]:.2f}")
    
#     print("\nBarrier Calculation:")
#     print(f"1. Base ATR % = {atr_percent_changes[idx]:.2f}")
#     print(f"2. Upper coefficient = {upper_coeff_smooth}")
#     print(f"3. Lower coefficient = {lower_coeff_smooth}")
#     print(f"4. Max upper barrier = ATR % * upper_coeff = {atr_percent_changes[idx]:.2f} * {upper_coeff_smooth} = {atr_percent_changes[idx] * upper_coeff_smooth:.2f}%")
#     print(f"5. Max lower barrier = ATR % * lower_coeff = {atr_percent_changes[idx]:.2f} * {lower_coeff_smooth} = {atr_percent_changes[idx] * lower_coeff_smooth:.2f}%")
#     print(f"6. Class barriers are calculated as fractions (1/{num_classes_per_side} to 1) of max barriers")
    
#     print("\nBarriers:")
#     for cls in range(num_classes_per_side):
#         fraction = (cls + 1) / num_classes_per_side
#         print(f"Class {cls+1} ({fraction:.2f} of max): {boundaries_positive[idx, cls]:.2f}% / {boundaries_negative[idx, cls]:.2f}%")
    
#     print("\nFuture price changes:")
#     for t in range(look_ahead_window):
#         if not np.isnan(percent_changes[idx, t]):
#             print(f"t+{t+1}: {percent_changes[idx, t]:.2f}%")
#         else:
#             print(f"t+{t+1}: NaN")


# # Prepare data
# #buy_signals = ticker_df_adjusted['buy'].to_numpy()
# close_prices = ticker_df_adjusted['close_sav_gol'].to_numpy()
# atr_percent_changes = np.abs(
#     ticker_df_adjusted['atr_percent_change_for_labeling_sav_gol'].to_numpy()
# )

# # Handle NaN values using recommended methods
# atr_percent_changes_series = pd.Series(atr_percent_changes)
# atr_percent_changes_filled = atr_percent_changes_series.ffill().bfill()  # Forward fill then backward fill
# atr_percent_changes = atr_percent_changes_filled.to_numpy()

# n = len(close_prices)

# # Calculate barriers for smoothed data
# upper_barrier_smooth = atr_percent_changes * upper_coeff_smooth
# lower_barrier_smooth = -atr_percent_changes * lower_coeff_smooth
# abs_lower_barrier_smooth = np.abs(lower_barrier_smooth)

# # Prepare shifted prices matrix for the look-ahead window
# shifted_prices = np.full((n, look_ahead_window), np.nan)

# for shift in range(1, look_ahead_window + 1):
#     shifted = np.roll(close_prices, -shift)
#     shifted_prices[:, shift - 1] = shifted

# # Mask out invalid future prices
# mask = (
#     np.arange(n)[:, None] + np.arange(1, look_ahead_window + 1)
# ) > n - 1
# shifted_prices[mask] = np.nan

# # Compute percent changes
# percent_changes = (
#     (shifted_prices - close_prices[:, None])
#     / close_prices[:, None]
#     * 100
# )

# # Compute class boundaries
# coefficients_positive = np.linspace(1/num_classes_per_side, 1, num_classes_per_side)
# coefficients_negative = np.linspace(1/num_classes_per_side, 1, num_classes_per_side)

# boundaries_positive = upper_barrier_smooth[:, None] * coefficients_positive[None, :]
# boundaries_negative = lower_barrier_smooth[:, None] * coefficients_negative[None, :]

# # Initialize labels
# labels = np.full(n, -999, dtype=int)
# max_class = num_classes_per_side

# # Get max class barriers
# pos_boundary_max = boundaries_positive[:, -1]  # Max class barrier
# neg_boundary_max = boundaries_negative[:, -1]  # Max negative class barrier

# # Check for max class barrier hits
# pos_hit_max = percent_changes >= pos_boundary_max[:, None]
# neg_hit_max = percent_changes <= neg_boundary_max[:, None]

# # Assign max class labels where appropriate
# pos_hit_max_any = np.any(pos_hit_max, axis=1)
# neg_hit_max_any = np.any(neg_hit_max, axis=1)
# labels[pos_hit_max_any] = max_class
# labels[neg_hit_max_any] = -max_class

# # Debug section for max class hits
# if debugging_mode and symbol == debug_symbol:
#     date_mask = (ticker_df_adjusted.index >= debug_start_date) & \
#                 (ticker_df_adjusted.index <= debug_end_date)
    
#     # Debug positive max hits
#     debug_pos = pos_hit_max_any & date_mask
#     if np.any(debug_pos):
#         for idx in np.where(debug_pos)[0]:
#             date = ticker_df_adjusted.index[idx]
#             print(f"\nMax positive class hit at {date}:")
#             print_sample_debug(idx, date, close_prices, atr_percent_changes, 
#                             boundaries_positive, boundaries_negative, percent_changes,
#                             look_ahead_window, num_classes_per_side)
#             print(f"Assigned Label: {max_class}")
    
#     # Debug negative max hits
#     debug_neg = neg_hit_max_any & date_mask
#     if np.any(debug_neg):
#         for idx in np.where(debug_neg)[0]:
#             date = ticker_df_adjusted.index[idx]
#             print(f"\nMax negative class hit at {date}:")
#             print_sample_debug(idx, date, close_prices, atr_percent_changes,
#                             boundaries_positive, boundaries_negative, percent_changes,
#                             look_ahead_window, num_classes_per_side)
#             print(f"Assigned Label: {-max_class}")

# # For remaining samples, use final price change
# remaining_mask = labels == -999

# if np.any(remaining_mask):
#     # Get final price changes (t+5)
#     final_changes = percent_changes[:, -1]
    
#     # Iterate through classes (largest to smallest) for remaining samples
#     for cls in range(max_class - 1, 0, -1):  # Exclude max_class as it's already handled
#         pos_boundary = boundaries_positive[:, cls-1]
#         neg_boundary = boundaries_negative[:, cls-1]
        
#         # Check which samples exceed current barriers
#         pos_samples = (final_changes >= pos_boundary) & remaining_mask
#         neg_samples = (final_changes <= neg_boundary) & remaining_mask
        
#         # Assign labels and update remaining mask
#         labels[pos_samples] = cls
#         labels[neg_samples] = -cls
#         remaining_mask = labels == -999
        
#         if debugging_mode and symbol == debug_symbol:
#             date_mask = (ticker_df_adjusted.index >= debug_start_date) & \
#                         (ticker_df_adjusted.index <= debug_end_date)
            
#             debug_pos = pos_samples & date_mask
#             debug_neg = neg_samples & date_mask
            
#             if np.any(debug_pos):
#                 for idx in np.where(debug_pos)[0]:
#                     date = ticker_df_adjusted.index[idx]
#                     print(f"\nClass {cls} positive assignment at {date}:")
#                     print_sample_debug(idx, date, close_prices, atr_percent_changes,
#                                     boundaries_positive, boundaries_negative, percent_changes,
#                                     look_ahead_window, num_classes_per_side)
#                     print(f"Final change: {final_changes[idx]:.2f}%")
#                     print(f"Assigned Label: {cls}")
            
#             if np.any(debug_neg):
#                 for idx in np.where(debug_neg)[0]:
#                     date = ticker_df_adjusted.index[idx]
#                     print(f"\nClass {cls} negative assignment at {date}:")
#                     print_sample_debug(idx, date, close_prices, atr_percent_changes,
#                                     boundaries_positive, boundaries_negative, percent_changes,
#                                     look_ahead_window, num_classes_per_side)
#                     print(f"Final change: {final_changes[idx]:.2f}%")
#                     print(f"Assigned Label: {-cls}")

#     # Assign smallest class to any remaining samples with valid final changes
#     remaining_samples = remaining_mask & ~np.isnan(final_changes)
#     labels[remaining_samples & (final_changes > 0)] = 1
#     labels[remaining_samples & (final_changes < 0)] = -1
    
#     if debugging_mode and symbol == debug_symbol:
#         date_mask = (ticker_df_adjusted.index >= debug_start_date) & \
#                     (ticker_df_adjusted.index <= debug_end_date)
#         debug_remaining = remaining_samples & date_mask
        
#         if np.any(debug_remaining):
#             for idx in np.where(debug_remaining)[0]:
#                 date = ticker_df_adjusted.index[idx]
#                 print(f"\nSmallest class assignment at {date}:")
#                 print_sample_debug(idx, date, close_prices, atr_percent_changes,
#                                 boundaries_positive, boundaries_negative, percent_changes,
#                                 look_ahead_window, num_classes_per_side)
#                 print(f"Final change: {final_changes[idx]:.2f}%")
#                 print(f"Assigned Label: {1 if final_changes[idx] > 0 else -1}")

# # Handle remaining invalid samples (those with NaN final changes)
# labels[remaining_mask & np.isnan(final_changes)] = -100

# # Prepare label assignment mask based on buy signals
# # label_assigned = np.zeros(n, dtype=bool)
# # buy_indices = np.where(buy_signals == 1)[0]

# # for i in buy_indices:
# #     end_idx = min(i + labeling_window, n)
# #     label_assigned[i:end_idx] = True

# # # Final labels assignment
# # final_labels = np.where(label_assigned, labels, -999)

# # Update the DataFrame with the final labels
# ticker_df_adjusted['label'] = labels

# if debugging_mode and symbol == debug_symbol:
#     date_mask = (ticker_df_adjusted.index >= debug_start_date) & \
#                 (ticker_df_adjusted.index <= debug_end_date)
#     debug_labels = labels[date_mask]
#     print("\nFinal Label Distribution in Debug Range:")
#     unique_labels, counts = np.unique(debug_labels, return_counts=True)
#     for label, count in zip(unique_labels, counts):
#         print(f"  Label {label}: {count}")


        # import pandas as pd
        # import numpy as np
        # import xgboost as xgb
        # from sklearn.metrics import confusion_matrix
        # import plotly.graph_objects as go
        # import torch
        # from torch import nn
        
        # def plot_confusion_matrix(
        #     model,
        #     X_test,
        #     y_test,
        #     num_classes,
        #     model_name,
        #     is_xgboost=False,
        #     is_lightgbm=False,
        #     is_tabnet=False,
        #     is_pytorch_nn=False,
        #     device=torch.device('cpu')  # Default device
        # ):
        #     """
        #     Create and display a confusion matrix for the given model.
        
        #     Parameters:
        #     - model: The trained model (XGBoost, CatBoost, Random Forest, LightGBM, TabNet, or PyTorch NN)
        #     - X_test: Test features (DataFrame or NumPy array, depending on the model)
        #     - y_test: True test labels
        #     - num_classes: Number of classes in the target
        #     - model_name: String name of the model for the plot title
        #     - is_xgboost: Boolean flag for XGBoost-specific handling
        #     - is_lightgbm: Boolean flag for LightGBM-specific handling
        #     - is_tabnet: Boolean flag for TabNet-specific handling
        #     - is_pytorch_nn: Boolean flag for PyTorch NN-specific handling
        #     - device: PyTorch device (CPU or GPU) for PyTorch NN
        #     """
        #     # 1. Generate predicted probabilities depending on the model type
        #     if is_xgboost:
        #         # XGBoost requires DMatrix conversion
        #         dtest = xgb.DMatrix(X_test, enable_categorical=True)
        #         y_pred_probs = model.predict(dtest)
        #     elif is_lightgbm:
        #         # LightGBM can use best iteration if available
        #         y_pred_probs = model.predict(X_test, num_iteration=model.best_iteration)
        #     elif is_tabnet:
        #         # TabNet expects a NumPy array (DataFrame causes KeyError indexing)
        #         if isinstance(X_test, pd.DataFrame):
        #             X_test_np = X_test.values
        #         else:
        #             X_test_np = X_test
        #         y_pred_probs = model.predict_proba(X_test_np)
        #     elif is_pytorch_nn:
        #         # Ensure the model is in evaluation mode
        #         model.eval()
                
        #         # Convert X_test to a PyTorch tensor
        #         if isinstance(X_test, pd.DataFrame):
        #             X_test_np = X_test.values
        #         else:
        #             X_test_np = X_test
        #         X_tensor = torch.tensor(X_test_np, dtype=torch.float32).to(device)
                
        #         with torch.no_grad():
        #             outputs = model(X_tensor)
        #             # Apply softmax to get probabilities
        #             y_pred_probs = torch.softmax(outputs, dim=1).cpu().numpy()
        #     else:
        #         # CatBoost, Random Forest, and other models typically handle DataFrames or arrays
        #         y_pred_probs = model.predict_proba(X_test)
        
        #     # 2. Handle binary classification where predict_proba might return a 1D array
        #     if y_pred_probs.ndim == 1:
        #         # Convert shape from (n_samples,) to (n_samples, 2)
        #         y_pred_probs = np.vstack([1 - y_pred_probs, y_pred_probs]).T
        
        #     # 3. Get predicted classes via argmax
        #     y_pred = np.argmax(y_pred_probs, axis=1)
        
        #     # 4. Compute confusion matrix
        #     cm = confusion_matrix(y_test, y_pred, labels=range(num_classes))
        
        #     # 5. Define labels (assuming numeric class labels 0..num_classes-1)
        #     labels = list(range(num_classes))
        
        #     # 6. Create Plotly heatmap of the confusion matrix
        #     fig = go.Figure(data=go.Heatmap(
        #         z=cm,
        #         x=labels,
        #         y=labels,
        #         text=cm,  # Display cell counts as text
        #         texttemplate="%{text}",
        #         textfont={"size": 12},
        #         hoverongaps=False,
        #         colorscale='Blues',
        #         showscale=True
        #     ))
        
        #     # 7. Customize layout
        #     fig.update_layout(
        #         title=f'Confusion Matrix - {model_name}',
        #         title_x=0.5,
        #         width=900,
        #         height=800,
        #         xaxis_title="Predicted Label",
        #         yaxis_title="True Label",
        #         xaxis=dict(
        #             tickmode='linear',
        #             ticktext=labels,
        #             tickvals=labels,
        #         ),
        #         yaxis=dict(
        #             tickmode='linear',
        #             ticktext=labels,
        #             tickvals=labels,
        #             autorange='reversed'
        #         )
        #     )
        
        #     # 8. Show the plot
        #     fig.show()

            
        # # ------------------------------------------------------------------------------
        # # Example usage assuming you have the following models trained and loaded:
        # #  bst_xgb      -> XGBoost model
        # #  cat_model    -> CatBoost model
        # #  rf_model     -> Random Forest model
        # #  lgb_model    -> LightGBM model
        # #  tabnet_model -> TabNet model
        # #  nn_model     -> PyTorch Neural Network model
        # #  X_test       -> Test set features
        # #  y_test       -> Test set labels (true labels)
        # #  num_classes  -> Number of classes in the classification problem
        # #  show_confusion_matrix -> Boolean flag to control if we plot
        # # ------------------------------------------------------------------------------
        
        # if show_confusion_matrix:
        #     # XGBoost Confusion Matrix
        #     print("\nGenerating XGBoost Confusion Matrix...")
        #     plot_confusion_matrix(
        #         model=bst_xgb,
        #         X_test=X_test,
        #         y_test=y_test,
        #         num_classes=num_classes,
        #         model_name="XGBoost",
        #         is_xgboost=True
        #     )
            
        #     # CatBoost Confusion Matrix
        #     print("\nGenerating CatBoost Confusion Matrix...")
        #     plot_confusion_matrix(
        #         model=cat_model,
        #         X_test=X_test,
        #         y_test=y_test,
        #         num_classes=num_classes,
        #         model_name="CatBoost"
        #     )
            
        #     # Random Forest Confusion Matrix
        #     print("\nGenerating Random Forest Confusion Matrix...")
        #     plot_confusion_matrix(
        #         model=rf_model,
        #         X_test=X_test,
        #         y_test=y_test,
        #         num_classes=num_classes,
        #         model_name="Random Forest"
        #     )
            
        #     # LightGBM Confusion Matrix
        #     print("\nGenerating LightGBM Confusion Matrix...")
        #     plot_confusion_matrix(
        #         model=lgb_model,
        #         X_test=X_test,
        #         y_test=y_test,
        #         num_classes=num_classes,
        #         model_name="LightGBM",
        #         is_lightgbm=True
        #     )

        #     # Extra Trees Confusion Matrix
        #     print("\nGenerating Extra Trees Confusion Matrix...")
        #     plot_confusion_matrix(
        #         model=extra_trees_model,
        #         X_test=X_test,
        #         y_test=y_test,
        #         num_classes=num_classes,
        #         model_name="Extra Trees"
        #     )
            
        #     # Rotation Forest Confusion Matrix
        #     # print("\nGenerating Rotation Forest Confusion Matrix...")
        #     # plot_confusion_matrix(
        #     #     model=rotation_forest_model,
        #     #     X_test=X_test,
        #     #     y_test=y_test,
        #     #     num_classes=num_classes,
        #     #     model_name="Rotation Forest"
        #     # )
            
        #     # ERTBoost Confusion Matrix
        #     print("\nGenerating ERTBoost Confusion Matrix...")
        #     plot_confusion_matrix(
        #         model=ertboost_model,
        #         X_test=X_test,
        #         y_test=y_test,
        #         num_classes=num_classes,
        #         model_name="ERTBoost"
        #     )
                
        #     # PyTorch Neural Network Confusion Matrix
        #     print("\nGenerating PyTorch Neural Network Confusion Matrix...")
            
        #     # Define the NeuralNet class (ensure it's defined in your script)
        #     class NeuralNet(nn.Module):
        #         def __init__(self, input_size, hidden_sizes, num_classes, dropout_p=0.3):
        #             super(NeuralNet, self).__init__()
        #             layers = []
        #             previous_size = input_size
        #             for hidden_size in hidden_sizes:
        #                 layers.append(nn.Linear(previous_size, hidden_size))
        #                 layers.append(nn.BatchNorm1d(hidden_size))
        #                 layers.append(nn.ReLU())
        #                 layers.append(nn.Dropout(dropout_p))
        #                 previous_size = hidden_size
        #             layers.append(nn.Linear(previous_size, num_classes))
        #             self.network = nn.Sequential(*layers)
                
        #         def forward(self, x):
        #             return self.network(x)
            
        #     # Initialize the PyTorch Neural Network model
        #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #     print(f"Using device: {device}")
        #     input_size = X_train.shape[1]  # Should match the number of features
        #     hidden_sizes = [128, 64]       # Changed to match training architecture
        #     num_classes = num_classes      # Should match the number of classes
        #     dropout_p = 0.3               # Must match training
            
        #     nn_model = NeuralNet(input_size, hidden_sizes, num_classes, dropout_p).to(device)
            
        #     # Load the PyTorch model info and state dictionary
        #     try:
        #         # First load the model info file (contains metadata and path to state dict)
        #         nn_model_info = joblib.load(ensemble_info['pytorch_nn_path'])
                
        #         # Get the path to the actual model state dictionary
        #         nn_model_state_path = nn_model_info['model_state_path']
                
        #         # Load the state dictionary into the model
        #         nn_model.load_state_dict(torch.load(nn_model_state_path, map_location=device))
        #         nn_model.eval()  # Set to evaluation mode
        #         print("Successfully loaded PyTorch model state.")
                
        #     except FileNotFoundError as e:
        #         print(f"Error: Could not find model file: {e}")
        #     except Exception as e:
        #         print(f"Error loading PyTorch model: {e}")
            
        #     # Plot confusion matrix for PyTorch Neural Network
        #     plot_confusion_matrix(
        #         model=nn_model,
        #         X_test=X_test,
        #         y_test=y_test,
        #         num_classes=num_classes,
        #         model_name="PyTorch Neural Network",
        #         is_pytorch_nn=True,
        #         device=device
        #     )


        # ###############
        # # FEATURE IMPORTANCE 
        # ################

        # if show_xgboost_feature_importance:
        #     import pandas as pd
        #     import plotly.express as px
        #     import xgboost as xgb
            
        #     # Load the XGBoost model info
        #     xgb_model_info = joblib.load(ensemble_info['xgboost_path'])
            
        #     # Get the actual model from the info
        #     bst = xgb_model_info['model']
            
        #     # 1. Extract feature importances for 'weight', 'gain', and 'cover'
        #     importance_weight = bst.get_score(importance_type='weight')
        #     importance_gain = bst.get_score(importance_type='gain')
        #     importance_cover = bst.get_score(importance_type='cover')
            
        #     # 2. Convert each importance type to a DataFrame
        #     importance_df_weight = pd.DataFrame({
        #         'Feature': list(importance_weight.keys()),
        #         'Weight': list(importance_weight.values())
        #     })
            
        #     importance_df_gain = pd.DataFrame({
        #         'Feature': list(importance_gain.keys()),
        #         'Gain': list(importance_gain.values())
        #     })
            
        #     importance_df_cover = pd.DataFrame({
        #         'Feature': list(importance_cover.keys()),
        #         'Cover': list(importance_cover.values())
        #     })
            
        #     # 3. Sort by each importance type and select the top 60 features
        #     importance_df_weight_sorted = importance_df_weight.sort_values(by='Weight', ascending=False).head(60)
        #     importance_df_gain_sorted = importance_df_gain.sort_values(by='Gain', ascending=False).head(60)
        #     importance_df_cover_sorted = importance_df_cover.sort_values(by='Cover', ascending=False).head(60)
            
        #     #Sort gain importance in ascending order and select the worst 50 features
        #     importance_df_gain_worst_50 = importance_df_gain.sort_values(by='Gain', ascending=True).head(50)
            
        #     # Extract the feature names of the worst 50 gain importance features
        #     worst_50_gain_features = importance_df_gain_worst_50['Feature'].tolist()
            
        #     # Function to create a plot with sub-title and adjusted x-axis font size
        #     def create_feature_importance_plot(df, x, y, title_main, subtitle):
        #         fig = px.bar(
        #             df, 
        #             x=x, 
        #             y=y, 
        #             title=f"{title_main}<br><span style='font-size:12px;'>({subtitle})</span>",
        #             labels={x: x, y: y},
        #             height=600, 
        #             width=1000
        #         )
                
        #         fig.update_layout(
        #             xaxis_title=x,
        #             yaxis_title=y,
        #             xaxis_tickangle=-45,
        #             xaxis_tickfont=dict(size=8),  # Adjust the font size here
        #             title={
        #                 'text': f"{title_main}<br><span style='font-size:12px;'>({subtitle})</span>",
        #                 'y':0.95,
        #                 'x':0.5,
        #                 'xanchor': 'center',
        #                 'yanchor': 'top'
        #             },
        #             font=dict(size=12)
        #         )
                
        #         return fig
            
        #     # 4. Plot the top 60 features by Weight
        #     subtitle_weight = "Number of times feature is used in splits."
        #     fig_weight = create_feature_importance_plot(
        #         importance_df_weight_sorted, 
        #         x='Feature', 
        #         y='Weight', 
        #         title_main='Top 60 Features by Weight Importance',
        #         subtitle=subtitle_weight
        #     )
            
        #     # 5. Plot the top 60 features by Gain
        #     subtitle_gain = "Average gain of splits which use the feature."
        #     fig_gain = create_feature_importance_plot(
        #         importance_df_gain_sorted, 
        #         x='Feature', 
        #         y='Gain', 
        #         title_main='Top 60 Features by Gain Importance',
        #         subtitle=subtitle_gain
        #     )
            
        #     # 6. Plot the top 60 features by Cover
        #     subtitle_cover = "Number of data points split using this feature."
        #     fig_cover = create_feature_importance_plot(
        #         importance_df_cover_sorted, 
        #         x='Feature', 
        #         y='Cover', 
        #         title_main='Top 60 Features by Cover Importance',
        #         subtitle=subtitle_cover
        #     )
            
        #     #7. Display all three plots
        #     fig_weight.show()
        #     fig_gain.show()
        #     fig_cover.show()
                

        # ###############
        # # ROC AUC CURVES
        # ################
        
        # import pandas as pd
        # import numpy as np
        # import plotly.graph_objs as go
        # import xgboost as xgb
        # from sklearn.preprocessing import label_binarize
        # from sklearn.metrics import roc_curve, auc
        # import torch
        # import torch.nn.functional as F
        
        # def plot_multiclass_roc_auc(y_test, X_test, models_dict, device='cpu'):
        #     """
        #     Plot Multiclass ROC AUC for various models.
            
        #     Parameters:
        #     - y_test: True labels for the test set (array-like)
        #     - X_test: Features for the test set (DataFrame or NumPy array)
        #     - models_dict: Dictionary containing models
        #     - device: Device to run PyTorch model on ('cpu' or 'cuda')
        #     """
        #     for model_name, model in models_dict.items():
        #         print(f"\nCalculating ROC AUC for {model_name.upper()}")
                
        #         # --- 1. Handle predictions based on the model type ---
        #         if model_name == 'xgboost':
        #             # Convert to DMatrix for XGBoost
        #             dtest = xgb.DMatrix(X_test, enable_categorical=True)
        #             y_test_pred_probs = model.predict(dtest)
        #             n_classes = y_test_pred_probs.shape[1]
        #             classes = np.arange(n_classes)
                
        #         elif model_name == 'lightgbm':
        #             # Use best iteration if available
        #             y_test_pred_probs = model.predict(X_test, num_iteration=model.best_iteration)
        #             n_classes = y_test_pred_probs.shape[1]
        #             classes = np.arange(n_classes)
                
        #         elif model_name == 'tabnet':
        #             # TabNet requires NumPy array instead of DataFrame
        #             X_test_np = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
        #             y_test_pred_probs = model.predict_proba(X_test_np)
        #             n_classes = y_test_pred_probs.shape[1]
        #             classes = np.arange(n_classes)
                
        #         elif model_name == 'pytorch_nn':
        #             # PyTorch Neural Network
        #             model.eval()  # Set model to evaluation mode
        #             X_test_np = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
        #             X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32).to(device)
                    
        #             with torch.no_grad():
        #                 outputs = model(X_test_tensor)
        #                 # Apply softmax to get probabilities
        #                 y_test_pred_probs = F.softmax(outputs, dim=1).cpu().numpy()
                    
        #             n_classes = y_test_pred_probs.shape[1]
        #             classes = np.arange(n_classes)
                
        #         else:
        #             # CatBoost, Random Forest, or other sklearn-like models
        #             if hasattr(model, 'classes_'):
        #                 classes = model.classes_
        #             elif hasattr(model, 'estimators') and hasattr(model, 'classes_'):
        #                 # For ensemble models, ensure classes_ is available
        #                 classes = model.classes_
        #             else:
        #                 raise AttributeError(f"The model '{model_name}' does not have a 'classes_' attribute.")
                    
        #             n_classes = len(classes)
        #             y_test_pred_probs = model.predict_proba(X_test)
                
        #         # --- 2. Binarize the labels for multi-class ROC ---
        #         # Ensure that classes are sorted and match between y_test and model.classes_
        #         y_test_binarized = label_binarize(y_test, classes=classes)
        #         if y_test_binarized.shape[1] != n_classes:
        #             print(f"Warning: Number of classes in y_test_binarized ({y_test_binarized.shape[1]}) does not match n_classes ({n_classes}).")
                
        #         # --- 3. Plot the ROC curves ---
        #         roc_auc_dict = {}
        #         fig = go.Figure()
                
        #         for i in range(n_classes):
        #             fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_test_pred_probs[:, i])
        #             roc_auc = auc(fpr, tpr)
        #             roc_auc_dict[i] = roc_auc
                    
        #             fig.add_trace(go.Scatter(
        #                 x=fpr, 
        #                 y=tpr, 
        #                 mode='lines',
        #                 name=f'Class {classes[i]} (AUC = {roc_auc:.2f})'
        #             ))
                
        #         # --- 4. Print average AUC ---
        #         average_roc_auc = np.mean(list(roc_auc_dict.values()))
        #         print(f"{model_name.upper()} Average ROC AUC: {average_roc_auc:.4f}")
                
        #         # --- 5. Add diagonal line for random classifier ---
        #         fig.add_trace(go.Scatter(
        #             x=[0, 1],
        #             y=[0, 1],
        #             mode='lines',
        #             name='Random Classifier',
        #             line=dict(color='black', dash='dash')
        #         ))
                
        #         # --- 6. Figure layout ---
        #         fig.update_layout(
        #             title=f'Multiclass ROC Curves - {model_name.upper()}',
        #             xaxis_title='False Positive Rate',
        #             yaxis_title='True Positive Rate',
        #             legend_title='Class',
        #             width=700,
        #             height=500
        #         )
                
        #         fig.show()

        
        
        # # ------------------------------------------------------------------------------
        
        # # Create dictionary of models
        # models = {
        #     'xgboost': bst_xgb,
        #     'catboost': cat_model,
        #     'randomforest': rf_model,
        #     'lightgbm': lgb_model,
        #     # 'tabnet': tabnet_model,  # Uncomment if TabNet is used
        #     'pytorch_nn': nn_model,  # Add PyTorch Neural Network model
        #     'extratrees': extra_trees_model,
        #     #'rotationforest': rotation_forest_model,
        #     'ertboost': ertboost_model
        # }
        
        # # Generate ROC curves for all models
        # if show_roc_curves:
        #     plot_multiclass_roc_auc(y_test, X_test, models, device=device)





# else:
# # --------------------------------------------------
# # Decide which model to use based on `use_cat_boost`
# # --------------------------------------------------

#     if use_cat_boost:

#         ############################
#         # CATBOOST TRAINING SNIPPET
#         ############################
#         from catboost import CatBoostClassifier
        
#         # Map your XGBoost parameters to CatBoost equivalents where possible
#         cat_params = {
#             'iterations': num_epochs,             # corresponds to num_boost_round in XGBoost
#             'depth': max_depth,                   # max_depth
#             'learning_rate': learning_rate,       # learning_rate
#             #'subsample': subsample,               # subsample
#             'rsm': colsample_bytree,              # colsample_bytree is called rsm in CatBoost
#             'l2_leaf_reg': lambda_p,              # lambda -> CatBoost's L2 reg
#             # There is no direct 'alpha' equivalent in CatBoost
#             # For min_child_weight we can use min_data_in_leaf:
#             'min_data_in_leaf': min_child_weight,
#             #'random_seed': 42,
#             'eval_metric': 'MultiClass',          # equivalent to mlogloss in XGB for multiclass
#             'loss_function': 'MultiClass'         # handles multi-class classification
#         }

#         # Initialize and train CatBoostClassifier
#         cat_model = CatBoostClassifier(**cat_params)
#         cat_model.fit(
#             X_train,
#             y_train,
#             sample_weight=sample_weights,          # pass sample weights if you have them
#             eval_set=(X_test, y_test),
#             verbose=False                          # turn off detailed CatBoost printing
#         )

#         # Example of creating a model_info dictionary to store key info
#         model_info = {
#             'model_type': 'catboost',
#             'model': cat_model,
#             'params': cat_params,
#         }

#         # Save the CatBoost model info via joblib
#         import joblib
#         joblib.dump(model_info, main_model_path)  # main_model_path is your desired save location

#         # Optional: If you want to save the raw CatBoost model file separately:
#         # cat_model.save_model("catboost_model.cbm")

#         bst = cat_model

#     else:
        
#         ###############
#         # Convert the datasets to DMatrix
#         ################
        
#         # Convert the datasets to DMatrix
#         dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights, enable_categorical=True)
#         dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)
        
#         # Verify the columns in the final feature sets
#         # print("\n")
#         # print("Columns in X_train:")
#         # print(X_train.columns)
#         # print("\n")
#         # print("Columns in X_test:")
#         # print("\n")
#         # print(X_test.columns)
        
#         ###############
#         # SET XGBOOST PARAMS 
#         ################
        
#         # Update XGBoost parameters for multiclass classification
#         num_classes = y_train.nunique()
#         params = {
#             'objective': 'multi:softprob',  # Use 'multi:softprob' for probability outputs
#             'eval_metric': 'mlogloss',
#             'num_class': num_classes,
#             'max_depth': max_depth,
#             'learning_rate': learning_rate,
#             'subsample': subsample,
#             'colsample_bytree': colsample_bytree,
#             'lambda': lambda_p,
#             'alpha': alpha,
#             'min_child_weight': min_child_weight,
#             #'seed': 42,
#             #'random_state': 42
#         }
        
#         # Watchlist to observe performance on both train and test datasets
#         watchlist = [(dtrain, 'train'), (dtest, 'eval')]
        
#         from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
#     ###############
#     # TRAIN THE MODEL
#     ################
    
#     # Initialize the custom callback
#     callback = MulticlassMetricsCallback(print_interval=print_interval)
    
#     # Train the XGBoost model with the custom callback
#     if early_stopping_enabled:
#         bst = xgb.train(
#             params, 
#             dtrain, 
#             num_boost_round=num_epochs, 
#             evals=watchlist, 
#             early_stopping_rounds=early_stopping_rounds, 
#             callbacks=[callback], 
#             verbose_eval=False
#         )
#     else:
#         bst = xgb.train(
#             params, 
#             dtrain, 
#             num_boost_round=num_epochs, 
#             evals=watchlist, 
#             callbacks=[callback], 
#             verbose_eval=False
#         )
