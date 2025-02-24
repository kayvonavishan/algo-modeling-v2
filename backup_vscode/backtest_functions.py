import numpy as np
import pandas as pd
import talib
from dataclasses import dataclass
from typing import Dict, List, Optional
from my_functions import calc_intraday_atr
import matplotlib.pyplot as plt 
from matplotlib.figure import Figure

@dataclass
class BacktestResults:
    """Container for backtest results"""
    trades_df: pd.DataFrame
    summary_metrics: Dict
    symbol_metrics: pd.DataFrame
    equity_curves: Dict[str, pd.DataFrame]
    equity_curve_plots: Optional[List[Figure]] = None

class SymbolBacktester:
    def __init__(self, initial_capital: float = 10000, 
                atr_period_for_stoploss: int = 28,  # Changed from atr_period
                stop_loss_adjust: float = 1.0, 
                stop_loss_adjust_sma_period: int = 200, 
                lookback_period_for_position_size: int = 50,  # For Kelly calculation
                ema_period: int = 20,
                kelly_multiplier: float = 3.0,  # Add this parameter
                debug: bool = False,
                overnight_position_size: float = 0.5,
                use_simpler_position_sizing: bool = True,
                use_performance_scaling: bool = True,
                stop_loss_scale_coeff: float = 25.0,
                is_model_type_short=False
                ):
        self.initial_capital = initial_capital
        self.atr_period_for_stoploss = atr_period_for_stoploss  # Updated variable name
        self.stop_loss_adjust = stop_loss_adjust
        self.stop_loss_adjust_sma_period = stop_loss_adjust_sma_period
        self.portfolio_timestamps = []
        self.portfolio_value = []
        self.debug = debug

        self.lookback_period_for_position_size = lookback_period_for_position_size
        self.ema_period = ema_period
        self.kelly_multiplier = kelly_multiplier
        self.class_metrics = {}
        self.current_position_size = 0.0
        self.current_entry_price = 0.0
        self.overnight_position_size = overnight_position_size
        self.use_simpler_position_sizing = use_simpler_position_sizing
        self.use_performance_scaling = use_performance_scaling
        self.stop_loss_scale_coeff = stop_loss_scale_coeff
        self.is_model_type_short = is_model_type_short
        # Initialize historical_trades with all expected columns
        self.historical_trades = pd.DataFrame(columns=[
            'symbol', 'buy_timestamp', 'sell_timestamp', 'buy_price', 'sell_price',
            'position_size', 'return', 'return_percentage', 'capital', 
            'hold_time_hours', 'predicted_class', 'is_partial'
        ])

    import numpy as np
    import pandas as pd

    def calculate_kelly_fraction(self, pred_class: int, current_idx: int, market_df: pd.DataFrame, 
                                kelly_debug: bool = False) -> float:
        """
        Calculate Kelly fraction based on historical performance with option for simpler position sizing.
        This version uses an exponential moving average (EMA) instead of a simple average for both the 
        simpler sizing and the Kelly-based sizing logic.
        """

        # Basic checks remain the same
        if 'predicted_class' not in self.historical_trades.columns:
            if self.debug or kelly_debug:
                print("[WARNING] 'predicted_class' column missing in self.historical_trades. Returning default Kelly=0.5")
            return 0.5

        if self.historical_trades.empty:
            if self.debug or kelly_debug:
                print("[WARNING] 'historical_trades' is empty. Returning default Kelly=0.25")
            return 0.25

        # Get historical trades for this class, limited to lookback
        cutoff_time = market_df.index[current_idx]
        historical_class_trades = self.historical_trades[
            (self.historical_trades['predicted_class'] == pred_class) & 
            (self.historical_trades['sell_timestamp'] < cutoff_time)
        ].tail(self.lookback_period_for_position_size)

        if kelly_debug and current_idx < 2000:
            print(f"\nPosition Sizing Calculation for row {current_idx}:")
            print(f"Predicted Class: {pred_class}")
            print(f"Number of historical trades: {len(historical_class_trades)}")
        
        if len(historical_class_trades) < 10:
            if kelly_debug and current_idx < 2000:
                print("Insufficient historical trades - using default = 0.25")
            return 0.25

        # Sort by 'sell_timestamp' so EMA follows chronological order
        historical_class_trades = historical_class_trades.sort_values('sell_timestamp')

        # ============ Simpler Position Sizing (EMA) ============
        # Instead of simple average over last N trades, we take the final value
        # of the EMA on return_percentage.
        ewm_return = historical_class_trades['return_percentage'].ewm(
            span=len(historical_class_trades), 
            adjust=False
        ).mean()
        # This is the final EMA-based "average" return for the class
        avg_return_ewm = ewm_return.iloc[-1] / 100.0  # convert % to decimal

        if self.use_simpler_position_sizing:
            # If negative, position_size=0.01; otherwise 1.0
            if avg_return_ewm < 0:
                position_size = 0.01
            else:
                position_size = 1.0

            if kelly_debug and current_idx < 2000:
                print("Using simpler position sizing:")
                print(f"EMA-based Average Return: {avg_return_ewm:.4f}")
                print(f"Position Size: {position_size:.2f}")

            return position_size

        # ============ Kelly-based Sizing (Using EMA) ============
        # win_prob: 0 or 1 for each trade, then an EMA
        winners_mask = historical_class_trades['return'] > 0
        # 0/1 series the same length as historical_class_trades
        win_01_series = winners_mask.astype(float)
        ewm_win_prob = win_01_series.ewm(
            span=len(win_01_series), 
            adjust=False
        ).mean()
        win_prob_ewm = ewm_win_prob.iloc[-1]  # fraction in [0..1]

        # Average win (in decimal form)
        pos_trades = historical_class_trades[winners_mask]
        if len(pos_trades) > 0:
            ewm_avg_win = pos_trades['return_percentage'].ewm(
                span=len(pos_trades),
                adjust=False
            ).mean().iloc[-1] / 100.0
        else:
            ewm_avg_win = 0.0

        # Average loss (in decimal form)
        losers_mask = historical_class_trades['return'] <= 0
        neg_trades = historical_class_trades[losers_mask]
        if len(neg_trades) > 0:
            ewm_avg_loss = abs(
                neg_trades['return_percentage'].ewm(
                    span=len(neg_trades),
                    adjust=False
                ).mean().iloc[-1] 
                / 100.0
            )
        else:
            ewm_avg_loss = 0.0

        if kelly_debug and current_idx < 2000:
            print(f"Win Probability (EMA): {win_prob_ewm:.3f}")
            print(f"Average Win (EMA): {ewm_avg_win:.3f}")
            print(f"Average Loss (EMA): {ewm_avg_loss:.3f}")

        # If there's literally zero average loss, revert to old fallback
        if ewm_avg_loss == 0:
            if kelly_debug and current_idx < 2000:
                print("Average loss is 0 - using default kelly = 0.75")
            return 0.75

        # Kelly formula: k = p - ( (1 - p) / (avg_win / avg_loss) )
        kelly_raw = win_prob_ewm - ((1 - win_prob_ewm) / (ewm_avg_win / ewm_avg_loss))

        # Additionally scale by (1 + avg_return/multiplier)^2, as in original code
        return_scalar = np.power(1.0 + (avg_return_ewm / self.kelly_multiplier), 2)
        kelly = kelly_raw * return_scalar

        # Clip between 0.01 and 1.0
        kelly = max(0.01, min(kelly, 1.0))

        if kelly_debug and current_idx < 2000:
            print(f"Raw Kelly (EMA-based): {kelly_raw:.3f}")
            print(f"Scaled Kelly: {kelly:.3f}")

        return kelly


    def calculate_trailing_metrics(self, market_df: pd.DataFrame, current_idx: int) -> Dict:
        """Calculate trailing metrics up to current_idx"""
        metrics = {}
        cutoff_time = market_df.index[current_idx]
        
        for pred_class in self.historical_trades['predicted_class'].unique():
            class_trades = self.historical_trades[
                (self.historical_trades['predicted_class'] == pred_class) &
                (self.historical_trades['sell_timestamp'] < cutoff_time)
            ]
            
            if len(class_trades) == 0:
                continue
                
            daily_returns = class_trades.groupby('buy_timestamp')['return_percentage'].mean()
            
            # Create series aligned with market data up to current_idx
            continuous_series = pd.Series(index=market_df.index[:current_idx+1], dtype=float)
            continuous_series[daily_returns.index] = daily_returns
            
            continuous_series = continuous_series.ffill()
            ema = continuous_series.ewm(span=self.ema_period, adjust=False).mean()
            
            metrics[pred_class] = ema
            
        return metrics


    def calculate_atr_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.debug:
            print(f"[DEBUG] Calculating ATR metrics on {len(df)} rows.")

        df = df.copy()

        # Ensure the input arrays are of type float64
        high = df['high_raw'].astype(np.float64).values
        low = df['low_raw'].astype(np.float64).values
        close = df['close_raw'].astype(np.float64).values

        # Compute ATR
        #atr_28 = talib.ATR(high, low, close, timeperiod=self.atr_period)

        # Suppose your DataFrame has columns: 'High', 'Low', 'Close'
        # and you want to ignore overnight gaps. You can call:
            
        atr_28 = calc_intraday_atr(
            df,
            col_high='high_raw',
            col_low='low_raw',
            col_close='close_raw',
            atr_period=self.atr_period_for_stoploss 
        )
        # Compute shorter-term SMA
        close_sma = talib.SMA(close, timeperiod=self.atr_period_for_stoploss)
        # Compute longer-term SMA
        close_sma_200 = talib.SMA(close, timeperiod=self.stop_loss_adjust_sma_period)

        # Ensure no NaNs in SMA calculations by handling edge cases
        close_above_sma_200 = (close > close_sma_200).astype(int)

        # Use the SMA in the denominator instead of raw close
        close_price_atr_28_percent_change = ((atr_28 / close_sma) * 100 * -1.0)

        # Assign computed metrics back to the DataFrame
        df['atr_28'] = atr_28
        df['close_sma'] = close_sma
        df['close_sma_200'] = close_sma_200
        df['close_above_sma_200'] = close_above_sma_200
        df['close_price_atr_28_percent_change'] = close_price_atr_28_percent_change

        # Drop rows with NaN values in critical columns
        return df.dropna(subset=['atr_28', 'close_price_atr_28_percent_change'])

    def process_valid_buys(self, buys: np.ndarray, sells: np.ndarray) -> np.ndarray:
        if self.debug:
            print("[DEBUG] Processing valid buys.")

        valid_buys = buys.copy()
        last_buy_index = -1
        last_sell_index = -1

        for i in range(len(buys)):
            if buys[i] == 1:
                # Check conditions for a valid buy
                if last_sell_index == -1 or last_buy_index >= last_sell_index:
                    if last_buy_index != -1 and (i - last_buy_index <= 1000):
                        valid_buys[i] = 0
                    else:
                        last_buy_index = i
                else:
                    last_buy_index = i

            if sells[i] == 1:
                last_sell_index = i

        return valid_buys
    
    def calculate_single_stoploss_threshold(
        self,
        index: int,
        current_price: float,
        close_prices: np.ndarray, 
        close_price_atr_28_percent_change: np.ndarray,
        close_above_sma_200: np.ndarray,
        atr_coeff: float,
        market_df: pd.DataFrame,
        lookback_period: int,
        use_performance_scaling: bool = True,
        min_trades_required: int = 10,
        stop_loss_scale_coeff: float = 1.0,
        max_scale_up: float = 2.0,
        min_scale_down: float = 0.5,
        #is_model_type_short: bool = False
    ) -> float:
        """
        Return the stop‐loss threshold (in % from highest price since buy) 
        for the *current* index.
        """
        base_threshold = close_price_atr_28_percent_change[index] * atr_coeff

        if use_performance_scaling:
            # --- Performance-based scaling: use exponential moving average of return_percentage.
            scale_multiplier = 1.0

            # We need the "current time" to filter historical trades up to now
            current_time = market_df.index[index]  # or timestamps[index] if you prefer
            if self.historical_trades is not None and not self.historical_trades.empty:
                # Filter to trades strictly before "current_time"
                recent_trades = self.historical_trades[
                    self.historical_trades['sell_timestamp'] < current_time
                ].copy()
                recent_trades.sort_values('sell_timestamp', inplace=True)

                # Take the tail of that set (lookback_period trades) then compute EMA
                recent_trades = recent_trades.tail(lookback_period)

                if len(recent_trades) >= min_trades_required:
                    ewm_return = recent_trades['return_percentage'].ewm(
                        span=len(recent_trades), adjust=False
                    ).mean()
                    avg_return_ewm = ewm_return.iloc[-1]

                    scale_multiplier = 1.0 + (avg_return_ewm * stop_loss_scale_coeff / 100.0)
                    scale_multiplier = np.clip(scale_multiplier, min_scale_down, max_scale_up)

                    if self.debug:
                        print(f"[DEBUG] StopLoss @ idx={index}: EMA-based trailing avg return={avg_return_ewm:.2f}%, "
                              f"scale_multiplier={scale_multiplier:.2f}")

            threshold = base_threshold * scale_multiplier

        else:
            # --- SMA-based scaling logic (your original approach).
            if not self.is_model_type_short:
                multiplier = self.stop_loss_adjust if close_above_sma_200[index] == 1 else 1.0
            else:
                multiplier = self.stop_loss_adjust if close_above_sma_200[index] != 1 else 1.0

            threshold = base_threshold * multiplier

        return threshold

    def execute_trades(self, timestamps: np.ndarray, close_prices: np.ndarray, 
            valid_buys: np.ndarray, sells: np.ndarray, 
            symbol: str, predicted_classes: np.ndarray,
            close_price_atr_28_percent_change,
            close_above_sma_200,
            atr_coeff,
            market_df: pd.DataFrame, kelly_debug: bool = False) -> pd.DataFrame:
        if self.debug:
            print(f"[DEBUG] Executing trades for symbol: {symbol}")

        trades_list = []
        capital = self.initial_capital
        position_open = False
        predicted_class_at_buy = None
        buy_timestamp = None
        
        # Reset position tracking at start
        self.current_position_size = 0.0
        self.current_entry_price = 0.0

        # Get the highest class number
        highest_class = max([int(col.split('_')[-1]) for col in market_df.columns if col.startswith('prediction_raw_class_')])
        highest_class_col = f'prediction_raw_class_{highest_class}'

        #stop loss tracking
        stoploss_threshold_pct = -np.inf
        percent_change = 0.0

        for index in range(len(close_prices)):
            current_price = close_prices[index]
            timestamp = timestamps[index]
            current_time = pd.Timestamp(timestamp).time()
            

            if position_open:
                if current_price > max_price_since_buy:
                    max_price_since_buy = current_price
                percent_change = (current_price - max_price_since_buy) / max_price_since_buy * 100

            # At 15:45, check for special selling logic
            if current_time.hour == 15 and current_time.minute == 45:
                # If a sell signal is active, let the full-sell logic handle closing the position.
                if sells[index] or percent_change < stoploss_threshold_pct:
                    pass  # Do nothing here so that the full sell logic below executes.
                else:
                    # No sell signal: limit the position size to a maximum of 50% of capital.
                    max_allowed = capital * self.overnight_position_size
                    if self.current_position_size > max_allowed:
                        excess_position = self.current_position_size - max_allowed
                        profit = (current_price - self.current_entry_price) / self.current_entry_price * excess_position
                        capital += profit

                        trade = {
                            'symbol': symbol,
                            'buy_timestamp': buy_timestamp,
                            'sell_timestamp': timestamp,
                            'buy_price': self.current_entry_price,
                            'sell_price': current_price,
                            'position_size': excess_position,  # Partial reduction trade
                            'return': profit,
                            'return_percentage': (current_price - self.current_entry_price) / self.current_entry_price * 100,
                            'capital': capital,
                            'hold_time_hours': (timestamp - buy_timestamp) / np.timedelta64(1, 'h'),
                            'predicted_class': predicted_class_at_buy,
                            'highest_class_probability': market_df.iloc[index][highest_class_col],
                            'is_partial': True
                        }
                        trades_list.append(trade)

                        self.current_position_size -= excess_position

                        if self.debug:
                            print(f"\n[DEBUG] At {timestamp}:")
                            print(f"Original position size: {self.current_position_size + excess_position:.2f}")
                            print(f"Excess position reduced: {excess_position:.2f}")
                            print(f"New position size: {self.current_position_size:.2f}")
                            print(f"Current capital: {capital:.2f}")

                        self.historical_trades = pd.DataFrame(trades_list)
                        
                        # Skip further processing for this timestamp so that a full sell isn't also executed.
                        continue


            # Regular buy logic
            if not position_open and valid_buys[index] == 1:
                predicted_class_at_buy = predicted_classes[index]
                
                # Calculate Kelly with available historical trades so far
                kelly_fraction = self.calculate_kelly_fraction(
                    pred_class=predicted_class_at_buy, 
                    current_idx=index, 
                    market_df=market_df, 
                    kelly_debug=kelly_debug
                )
                
                # Open new position
                position_open = True
                buy_timestamp = timestamp
                self.current_entry_price = current_price
                self.current_position_size = capital * kelly_fraction
                highest_class_prob_at_buy = market_df.iloc[index][highest_class_col]
                max_price_since_buy = current_price
                stoploss_threshold_pct = self.calculate_single_stoploss_threshold(
                    index=index,
                    current_price=current_price,
                    close_prices=close_prices,
                    close_price_atr_28_percent_change=close_price_atr_28_percent_change,
                    close_above_sma_200=close_above_sma_200,
                    atr_coeff=atr_coeff,
                    market_df=market_df,
                    lookback_period=self.lookback_period_for_position_size,
                    use_performance_scaling=self.use_performance_scaling,
                    stop_loss_scale_coeff=self.stop_loss_scale_coeff,
                    #is_model_type_short=False  # or True if you are in short mode
                )


            # Regular sell logic
            elif position_open and (sells[index] or percent_change < stoploss_threshold_pct):
                # Close entire remaining position
                profit = (current_price - self.current_entry_price) / self.current_entry_price * self.current_position_size
                capital += profit
                
                trade = {
                    'symbol': symbol,
                    'buy_timestamp': buy_timestamp,
                    'sell_timestamp': timestamp,
                    'buy_price': self.current_entry_price,
                    'sell_price': current_price,
                    'position_size': self.current_position_size,
                    'return': profit,
                    'return_percentage': (current_price - self.current_entry_price) / self.current_entry_price * 100,
                    'capital': capital,
                    'hold_time_hours': (timestamp - buy_timestamp) / np.timedelta64(1, 'h'),
                    'predicted_class': predicted_class_at_buy,
                    'highest_class_probability': highest_class_prob_at_buy,
                    'is_partial': False  # Complete position close
                }
                trades_list.append(trade)
                
                # Reset position tracking
                self.current_position_size = 0.0
                self.current_entry_price = 0.0
                position_open = False

                #reset stop loss tracking
                stoploss_threshold_pct = -np.inf
                percent_change = 0.0
                
                # Update historical trades immediately
                self.historical_trades = pd.DataFrame(trades_list)

            # Update portfolio tracking
            if len(self.portfolio_timestamps) == 0 or self.portfolio_timestamps[-1] != timestamp:
                self.portfolio_timestamps.append(timestamp)
                self.portfolio_value.append(capital)

        # Final update of historical trades
        if trades_list:
            self.historical_trades = pd.DataFrame(trades_list)

        return pd.DataFrame(trades_list) if trades_list else pd.DataFrame(columns=[
            'symbol', 'buy_timestamp', 'sell_timestamp', 'buy_price', 'sell_price',
            'position_size', 'return', 'return_percentage', 'capital', 
            'hold_time_hours', 'predicted_class', 'highest_class_probability', 'is_partial'
        ])


    def execute_trades_short_only(self, timestamps: np.ndarray, close_prices: np.ndarray, 
                valid_buys: np.ndarray, sells: np.ndarray, 
                symbol: str, predicted_classes: np.ndarray, 
                close_price_atr_28_percent_change,
                close_above_sma_200,
                atr_coeff,
                #is_model_type_short,
                market_df: pd.DataFrame, kelly_debug: bool = False) -> pd.DataFrame:
        if self.debug:
            print(f"[DEBUG] Executing short trades for symbol: {symbol}")

        trades_list = []
        capital = self.initial_capital
        position_open = False
        predicted_class_at_entry = None
        # We still use 'buy_timestamp' for the entry time (i.e. when we open the short)
        entry_timestamp = None

        # Reset position tracking at start
        self.current_position_size = 0.0
        self.current_entry_price = 0.0

        # Get the highest class number from the market dataframe columns
        highest_class = max([int(col.split('_')[-1]) for col in market_df.columns if col.startswith('prediction_raw_class_')])
        highest_class_col = f'prediction_raw_class_{highest_class}'

        #stop loss tracking
        stoploss_threshold_pct = -np.inf
        percent_change = 0.0

        for index in range(len(close_prices)):
            current_price = close_prices[index]
            timestamp = timestamps[index]
            current_time = pd.Timestamp(timestamp).time()

            if position_open:
                if current_price < min_price_since_entry:
                    min_price_since_entry = current_price
                percent_change = (min_price_since_entry - current_price) / min_price_since_entry * 100

            # At 15:45, check for special cover logic for short positions:
            # If it's 15:45 and no buy (cover) signal is active, then partially cover the short position 
            # if its absolute size exceeds the allowed maximum.
            if current_time.hour == 15 and current_time.minute == 45:
                if valid_buys[index] or percent_change < stoploss_threshold_pct:
                    # A cover signal is active; allow the regular covering logic below.
                    pass
                else:
                    max_allowed = capital * self.overnight_position_size
                    # For a short, current_position_size is negative.
                    if abs(self.current_position_size) > max_allowed:
                        excess_position = abs(self.current_position_size) - max_allowed
                        # Profit for a short trade: (entry_price - current_price) / entry_price * position size (absolute)
                        profit = (self.current_entry_price - current_price) / self.current_entry_price * excess_position
                        capital += profit

                        trade = {
                            'symbol': symbol,
                            'buy_timestamp': entry_timestamp,      # Short entry time
                            'sell_timestamp': timestamp,           # Cover (partial) time
                            'buy_price': self.current_entry_price,   # Short entry price
                            'sell_price': current_price,           # Cover price
                            'position_size': excess_position,      # Partial position covered (absolute value)
                            'return': profit,
                            'return_percentage': (self.current_entry_price - current_price) / self.current_entry_price * 100,
                            'capital': capital,
                            'hold_time_hours': (timestamp - entry_timestamp) / np.timedelta64(1, 'h'),
                            'predicted_class': predicted_class_at_entry,
                            'highest_class_probability': market_df.iloc[index][highest_class_col],
                            'is_partial': True
                        }
                        trades_list.append(trade)

                        # Reduce the short position (remember: it's stored as negative)
                        self.current_position_size = -max_allowed

                        if self.debug:
                            print(f"\n[DEBUG] At {timestamp}:")
                            print(f"Original short position size (abs): {abs(self.current_position_size) + excess_position:.2f}")
                            print(f"Excess short position covered: {excess_position:.2f}")
                            print(f"New short position size (abs): {max_allowed:.2f}")
                            print(f"Current capital: {capital:.2f}")

                        self.historical_trades = pd.DataFrame(trades_list)
                        # Skip further processing for this timestamp so that a full cover isn't also executed.
                        continue

            # Regular short entry logic:
            # Open a short position when no position is open and a sell signal is triggered.
            if not position_open and sells[index] == 1:
                predicted_class_at_entry = predicted_classes[index]

                # Calculate Kelly fraction based on available historical trades so far
                kelly_fraction = self.calculate_kelly_fraction(
                    pred_class=predicted_class_at_entry, 
                    current_idx=index, 
                    market_df=market_df, 
                    kelly_debug=kelly_debug
                )

                # Open a new short position.
                position_open = True
                entry_timestamp = timestamp  # Using 'buy_timestamp' to record the short entry time.
                self.current_entry_price = current_price
                # For shorts, we store the position size as negative.
                self.current_position_size = - (capital * kelly_fraction)
                highest_class_prob_at_entry = market_df.iloc[index][highest_class_col]
                min_price_since_entry = current_price
                stoploss_threshold_pct = self.calculate_single_stoploss_threshold(
                    index=index,
                    current_price=current_price,
                    close_prices=close_prices,
                    close_price_atr_28_percent_change=close_price_atr_28_percent_change,
                    close_above_sma_200=close_above_sma_200,
                    atr_coeff=atr_coeff,
                    market_df=market_df,
                    lookback_period=self.lookback_period_for_position_size,
                    use_performance_scaling=self.use_performance_scaling,
                    stop_loss_scale_coeff=self.stop_loss_scale_coeff,
                    #is_model_type_short=is_model_type_short 
                )

            # Regular cover (exit) logic:
            # Close the short position (cover) when in a short and a buy signal is triggered.
            elif position_open and (valid_buys[index] or percent_change < stoploss_threshold_pct):
                # Calculate profit: for a short, profit is (entry_price - cover_price)
                profit = (self.current_entry_price - current_price) / self.current_entry_price * abs(self.current_position_size)
                capital += profit

                trade = {
                    'symbol': symbol,
                    'buy_timestamp': entry_timestamp,       # Short entry timestamp
                    'sell_timestamp': timestamp,            # Cover timestamp
                    'buy_price': self.current_entry_price,    # Short entry price
                    'sell_price': current_price,            # Cover price
                    'position_size': abs(self.current_position_size),
                    'return': profit,
                    'return_percentage': (self.current_entry_price - current_price) / self.current_entry_price * 100,
                    'capital': capital,
                    'hold_time_hours': (timestamp - entry_timestamp) / np.timedelta64(1, 'h'),
                    'predicted_class': predicted_class_at_entry,
                    'highest_class_probability': highest_class_prob_at_entry,
                    'is_partial': False  # Complete cover of the short position
                }
                trades_list.append(trade)

                # Reset position tracking
                self.current_position_size = 0.0
                self.current_entry_price = 0.0
                position_open = False

                # Update historical trades immediately
                self.historical_trades = pd.DataFrame(trades_list)

            # Update portfolio tracking (timestamps and capital)
            if len(self.portfolio_timestamps) == 0 or self.portfolio_timestamps[-1] != timestamp:
                self.portfolio_timestamps.append(timestamp)
                self.portfolio_value.append(capital)

        # Final update of historical trades
        if trades_list:
            self.historical_trades = pd.DataFrame(trades_list)

        # Return DataFrame using the same column names as before.
        columns = [
            'symbol', 'buy_timestamp', 'sell_timestamp', 'buy_price', 'sell_price',
            'position_size', 'return', 'return_percentage', 'capital', 
            'hold_time_hours', 'predicted_class', 'highest_class_probability', 'is_partial'
        ]

        return pd.DataFrame(trades_list) if trades_list else pd.DataFrame(columns=columns)


    def calculate_metrics(self, trades_df: pd.DataFrame, symbol: str, 
                        initial_price: float, final_price: float, df: pd.DataFrame) -> Dict:
        if self.debug:
            print(f"[DEBUG] Calculating metrics for symbol: {symbol}")

        if trades_df.empty:
            # Keep existing empty DataFrame handling...
            return {
                'symbol': symbol,
                'total_returns_percentage': 0,
                'win_loss_ratio': 0,
                'average_percent_return': 0,
                'average_hold_time_hours': 0,
                'buy_and_hold_return_percentage': ((final_price - initial_price) / initial_price) * 100,
                'number_of_trades': 0,
                'buy_and_hold_hold_time_hours': (df.index[-1] - df.index[0]).total_seconds() / 3600.0,
                'trading_hold_time_hours': 0,
                'sharpe_ratio': 0,
                'overnight_trades_count': 0,
                'overnight_trades_return_percentage': 0,
                'partial_trades_count': 0
            }

        # Separate complete and partial trades
        complete_trades = trades_df[~trades_df['is_partial']]
        partial_trades = trades_df[trades_df['is_partial']]

        # Calculate metrics for all trades
        winning_trades = (trades_df['return'] > 0).sum()
        losing_trades = (trades_df['return'] <= 0).sum()
        win_loss_ratio = winning_trades / losing_trades if losing_trades > 0 else float('inf')

        # Calculate overnight trade metrics
        overnight_trades = trades_df[
            (trades_df['sell_timestamp'].dt.time == pd.Timestamp('15:45').time()) &
            trades_df['is_partial']
        ]

        # Buy and hold metrics remain the same
        buy_and_hold_hold_time_hours = (df.index[-1] - df.index[0]).total_seconds() / 3600.0
        trading_hold_time_hours = trades_df['hold_time_hours'].sum()

        # Calculate Sharpe Ratio (existing code remains the same)
        if len(self.portfolio_value) > 1:
            portfolio_df = pd.DataFrame({
                'timestamp': self.portfolio_timestamps,
                'capital': self.portfolio_value
            })
            portfolio_df.set_index('timestamp', inplace=True)
            portfolio_df = portfolio_df.sort_index()

            duplicates_portfolio = portfolio_df.index.duplicated().sum()
            assert duplicates_portfolio == 0, f"Symbol {symbol} has {duplicates_portfolio} duplicate timestamps in portfolio data."

            portfolio_daily = portfolio_df.resample('D').ffill()
            portfolio_daily['daily_return'] = portfolio_daily['capital'].pct_change()
            portfolio_daily.dropna(inplace=True)

            mean_daily_return = portfolio_daily['daily_return'].mean()
            std_daily_return = portfolio_daily['daily_return'].std()
            sharpe_ratio = (mean_daily_return / std_daily_return * np.sqrt(252)) if std_daily_return != 0 else 0
        else:
            sharpe_ratio = 0

        # Calculate average position sizes (considering both partial and complete trades)
        avg_position_size = trades_df['position_size'].mean() / self.initial_capital * 100
        position_size_std = trades_df['position_size'].std() / self.initial_capital * 100
        avg_kelly = avg_position_size / 100

        metrics = {
            'symbol': symbol,
            'total_returns_percentage': (trades_df['capital'].iloc[-1] - self.initial_capital) / self.initial_capital * 100,
            'win_loss_ratio': win_loss_ratio,
            'average_percent_return': trades_df['return_percentage'].mean(),
            'average_hold_time_hours': trades_df['hold_time_hours'].mean(),
            'buy_and_hold_return_percentage': ((final_price - initial_price) / initial_price) * 100,
            'number_of_trades': len(trades_df),
            'buy_and_hold_hold_time_hours': buy_and_hold_hold_time_hours,
            'trading_hold_time_hours': trading_hold_time_hours,
            'sharpe_ratio': sharpe_ratio,

            # Position sizing metrics
            'average_position_size': avg_position_size,
            'position_size_std': position_size_std,
            'average_kelly_fraction': avg_kelly,

            # New metrics for partial/overnight trading
            'complete_trades_count': len(complete_trades),
            'partial_trades_count': len(partial_trades),
            'overnight_trades_count': len(overnight_trades),
            'overnight_trades_return_percentage': overnight_trades['return_percentage'].mean() if not overnight_trades.empty else 0,
            'average_overnight_position_size': overnight_trades['position_size'].mean() / self.initial_capital * 100 if not overnight_trades.empty else 0,

            # Per-class metrics (now includes partial trade information)
            'class_metrics': trades_df.groupby(['predicted_class', 'is_partial']).agg({
                'return_percentage': ['mean', 'std'],
                'position_size': ['mean', 'std'],
                'return': lambda x: (x > 0).mean()  # win rate
            }).to_dict()
        }

        return metrics

    def get_active_signals(self, current_time, df_trades, debug=False):
        """
        Determine which signal set to use based on recent performance.
        Returns tuple of (buy_col, sell_col)
        """
        if df_trades.empty:
            if debug:
                print("[DEBUG] No historical trades, using regular signals")
            return 'buy_final', 'sell'
        
        # Grab recent trades and sort them by sell_timestamp
        recent_trades = df_trades[df_trades['sell_timestamp'] < current_time].copy()
        recent_trades.sort_values('sell_timestamp', inplace=True)
        recent_trades = recent_trades.tail(self.lookback_period_for_position_size)

        # If we have no recent trades, default to 0 for the average
        if recent_trades.empty:
            avg_return_ewm = 0
        else:
            # Compute an EMA of the return_percentage column
            ewm_return = recent_trades['return_percentage'].ewm(
                span=len(recent_trades),
                adjust=False
            ).mean()
            # Use the final EMA value
            avg_return_ewm = ewm_return.iloc[-1]

        if debug:
            print(f"\n[DEBUG] Signal Selection at {current_time}:")
            print(f"Recent trades count: {len(recent_trades)}")
            print(f"EMA-based average return: {avg_return_ewm:.2f}%")

        if avg_return_ewm < 0:
            if debug:
                print("Using short signals due to negative performance")
            return 'buy_final_fast', 'sell_fast'
        else:
            if debug:
                print("Using regular signals due to positive/neutral performance")
            return 'buy_final', 'sell'


    def backtest_symbol(self, df: pd.DataFrame, symbol: str, atr_coeff: float, kelly_debug: bool = False) -> tuple:
        if self.debug:
            print(f"[DEBUG] Backtesting symbol: {symbol}")

        # Reset historical trades at start of new symbol
        self.historical_trades = pd.DataFrame(columns=[
            'symbol', 'buy_timestamp', 'sell_timestamp', 'buy_price', 'sell_price',
            'position_size', 'return', 'return_percentage', 'capital',
            'hold_time_hours', 'predicted_class'
        ])

        df = self.calculate_atr_metrics(df)
        
        # Check for both regular and short signal columns
        if ('buy_final' not in df.columns or 'sell' not in df.columns or 
            'buy_final_fast' not in df.columns or 'sell_fast' not in df.columns):
            raise ValueError(f"Required buy/sell columns (regular and short) missing for symbol {symbol}.")

        # Initialize arrays
        timestamps = df.index.to_numpy()
        buys = np.zeros(len(df), dtype=np.int8)
        sells = np.zeros(len(df), dtype=np.int8)

        # Determine buy/sell signals for each timestamp based on performance
        for i in range(len(df)):
            buy_col, sell_col = self.get_active_signals(
                timestamps[i], 
                self.historical_trades, 
                debug=self.debug,
                #is_model_type_short=is_model_type_short
            )
            buys[i] = df[buy_col].iloc[i]
            sells[i] = df[sell_col].iloc[i]

        valid_buys = self.process_valid_buys(buys, sells)
        
        close_prices = df['close_raw'].astype(np.float64).values
        close_price_atr_28_percent_change = df['close_price_atr_28_percent_change'].astype(np.float64).values
        close_above_sma_200 = df['close_above_sma_200'].astype(np.int8).values

        # After processing buys/sells, add:
        predicted_classes = df['predicted_class_moving_avg'].values

        if not self.is_model_type_short:
            trade_data = self.execute_trades(
                timestamps, close_prices, valid_buys, sells, 
                symbol, predicted_classes,
                close_price_atr_28_percent_change=close_price_atr_28_percent_change,
                close_above_sma_200=close_above_sma_200,
                atr_coeff=atr_coeff,
                market_df=df, kelly_debug=kelly_debug
            )
        else:
            trade_data = self.execute_trades_short_only(
                timestamps, close_prices, valid_buys, sells, 
                symbol, predicted_classes, 
                close_price_atr_28_percent_change=close_price_atr_28_percent_change,
                close_above_sma_200=close_above_sma_200,
                atr_coeff=atr_coeff,
                market_df=df, kelly_debug=kelly_debug
            )

        trades_df = trade_data if not trade_data.empty else pd.DataFrame()

        metrics = self.calculate_metrics(
            trades_df,
            symbol,
            close_prices[0],
            close_prices[-1],
            df
        )

        if self.debug:
            print(f"[DEBUG] Calculated metrics for {symbol}")
            print(metrics)

        #df['sell_stop_loss'] = sell_stop_loss
        
        # Store which signals were used at each timestamp
        if self.debug:
            df['signals_used'] = ['regular' if r >= 0 else 'short' 
                                for r in trades_df['return_percentage']]

        # NEW: Store trailing metrics for this symbol
        if not trades_df.empty:
            self.class_metrics[symbol] = self.calculate_trailing_metrics(df, len(df)-1)

        # Collect equity curve
        if len(self.portfolio_timestamps) > 0 and len(self.portfolio_value) > 0:
            equity_curve_df = pd.DataFrame({
                'timestamp': self.portfolio_timestamps,
                'capital': self.portfolio_value
            })
            equity_curve_df['symbol'] = symbol
            equity_curve_df.sort_values('timestamp', inplace=True)
        else:
            equity_curve_df = pd.DataFrame()

        return trades_df, metrics, df, equity_curve_df
    
def calculate_buy_hold_equity(df, initial_capital):
    """Calculate buy & hold equity curve using all price changes"""
    equity_points = []
    current_capital = initial_capital
    first_price = df.iloc[0]['close_raw']
    
    for timestamp, row in df.iterrows():
        current_price = row['close_raw']
        current_capital = initial_capital * (current_price / first_price)
        equity_points.append((timestamp, current_capital))
    
    return pd.DataFrame(equity_points, columns=['timestamp', 'capital'])

def calculate_daily_equity(df, initial_capital):
    daily_returns = []
    capital = initial_capital
    
    for date, group in df.groupby(df.index.date):
        first_price = group.iloc[0]['close_raw']
        last_price = group.iloc[-1]['close_raw']
        daily_return = (last_price / first_price) - 1
        capital *= (1 + daily_return)
        daily_returns.append((group.index[-1], capital))
    
    return pd.DataFrame(daily_returns, columns=['timestamp', 'capital'])

def run_backtest(df: pd.DataFrame, atr_coeff: float = 1.0, initial_capital: float = 10000, 
                debug: bool = False, kelly_debug: bool = False,  # Add parameter
                atr_period_for_stoploss: int = 28,
                stop_loss_adjust: float = 1.0, 
                stop_loss_adjust_sma_period: int = 200,
                kelly_multiplier: float = 3.0,  
                overnight_position_size: float = 0.5,
                lookback_period_for_position_size: int = 50,
                stop_loss_scale_coeff: float = 25.0,
                use_performance_scaling: bool = True,
                use_simpler_position_sizing=True,
                return_equity_curves: bool = False,
                is_model_type_short=False) -> BacktestResults:
    if debug:
        print(f"[DEBUG] Starting run_backtest with atr_coeff={atr_coeff}, initial_capital={initial_capital}")

    all_trades = []
    all_metrics = []
    processed_dfs = []
    equity_curves = {}

    # Iterate over each symbol and process them independently
    symbols = df['symbol'].unique()
    for symbol in symbols:
        if debug:
            print(f"[DEBUG] Processing symbol: {symbol}")

        # Create a new SymbolBacktester instance for each symbol with the new parameters
        backtester = SymbolBacktester(
            initial_capital=initial_capital, 
            debug=debug,
            atr_period_for_stoploss=atr_period_for_stoploss,
            stop_loss_adjust=stop_loss_adjust,
            stop_loss_adjust_sma_period=stop_loss_adjust_sma_period,
            lookback_period_for_position_size=lookback_period_for_position_size,
            stop_loss_scale_coeff=stop_loss_scale_coeff,
            ema_period=20,
            use_performance_scaling=use_performance_scaling,
            kelly_multiplier=kelly_multiplier,
            overnight_position_size=overnight_position_size,
            use_simpler_position_sizing=use_simpler_position_sizing,
            is_model_type_short=is_model_type_short
        )

        symbol_df = df[df['symbol'] == symbol].copy()
        symbol_df.sort_index(inplace=True)

        # Check for duplicate timestamps in input data for this symbol
        duplicates_input = symbol_df.index.duplicated().sum()
        assert duplicates_input == 0, f"Symbol {symbol} has {duplicates_input} duplicate timestamps in input data."

        try:
            trades_df, metrics, processed_df, equity_curve_df = backtester.backtest_symbol(
                symbol_df,
                symbol,
                atr_coeff,
                kelly_debug=kelly_debug,  # Add this
            )
        except ValueError as e:
            print(f"Skipping symbol {symbol} due to error: {e}")
            continue
        except Exception as e:
            print(f"An unexpected error occurred for symbol {symbol}: {e}")
            continue

        if not trades_df.empty:
            all_trades.append(trades_df)
        all_metrics.append(metrics)
        processed_dfs.append(processed_df)

        if not equity_curve_df.empty:
            equity_curves[symbol] = equity_curve_df

    combined_trades = pd.concat(all_trades) if all_trades else pd.DataFrame()
    metrics_df = pd.DataFrame(all_metrics) if all_metrics else pd.DataFrame()
    processed_df = pd.concat(processed_dfs) if processed_dfs else pd.DataFrame()

    if not metrics_df.empty and 'trading_hold_time_hours' in metrics_df.columns and not metrics_df['trading_hold_time_hours'].empty:
        mean_hold_time_trading_hours = metrics_df['trading_hold_time_hours'].mean()
    else:
        mean_hold_time_trading_hours = 0

    if not metrics_df.empty and 'buy_and_hold_hold_time_hours' in metrics_df.columns and not metrics_df['buy_and_hold_hold_time_hours'].empty:
        mean_hold_time_buy_and_hold_hours = metrics_df['buy_and_hold_hold_time_hours'].mean()
    else:
        mean_hold_time_buy_and_hold_hours = 0

    if not metrics_df.empty and 'total_returns_percentage' in metrics_df.columns:
        average_total_return_percentage = metrics_df['total_returns_percentage'].mean()
    else:
        average_total_return_percentage = 0

    if ('buy_and_hold_return_percentage' in metrics_df.columns and 
        not metrics_df['buy_and_hold_return_percentage'].empty):
        average_total_return_percentage_buy_and_hold = metrics_df['buy_and_hold_return_percentage'].mean()
    else:
        average_total_return_percentage_buy_and_hold = 0

    return_per_trading_hour = (average_total_return_percentage / mean_hold_time_trading_hours) if mean_hold_time_trading_hours > 0 else 0
    return_per_buy_and_hold_hour = (average_total_return_percentage_buy_and_hold / mean_hold_time_buy_and_hold_hours) if mean_hold_time_buy_and_hold_hours > 0 else 0
    ratio_return_per_trading_hour = (return_per_trading_hour / return_per_buy_and_hold_hour) if return_per_buy_and_hold_hour > 0 else 0

    if 'sharpe_ratio' in metrics_df.columns and not metrics_df['sharpe_ratio'].empty:
        mean_sharpe_ratio = metrics_df['sharpe_ratio'].mean()
    else:
        mean_sharpe_ratio = 0

    overall_metrics = {
        'total_trades': len(combined_trades),
        'average_total_return_percentage': average_total_return_percentage,
        'average_trade_return': combined_trades['return_percentage'].mean() if not combined_trades.empty else 0,
        'average_hold_time': combined_trades['hold_time_hours'].mean() if not combined_trades.empty else 0,
        'winning_trades_percentage': (combined_trades['return'] > 0).mean() * 100 if not combined_trades.empty else 0,
        'total_hold_time_buy_and_hold_hours': mean_hold_time_buy_and_hold_hours,
        'total_hold_time_trading_hours': mean_hold_time_trading_hours,
        'average_total_return_percentage_buy_and_hold': average_total_return_percentage_buy_and_hold,
        'return_per_trading_hour': return_per_trading_hour,
        'return_per_buy_and_hold_hour': return_per_buy_and_hold_hour,
        'ratio_return_per_trading_hour': ratio_return_per_trading_hour,
        'mean_sharpe_ratio': mean_sharpe_ratio
    }

    equity_curve_plots = []  # New list to store the plots
    
    if return_equity_curves:
        for symbol, equity_df in equity_curves.items():
            symbol_df = df[df['symbol'] == symbol]
            
            # Calculate curves
            buy_hold_df = calculate_buy_hold_equity(symbol_df, initial_capital)
            daily_equity_df = calculate_daily_equity(symbol_df, initial_capital)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot all curves
            ax.plot(equity_df['timestamp'], equity_df['capital'], 
                    label='Strategy', color='blue')
            ax.plot(buy_hold_df['timestamp'], buy_hold_df['capital'],
                    label='Buy & Hold', color='green')
            ax.plot(daily_equity_df['timestamp'], daily_equity_df['capital'],
                    label='Daily Buy & Hold', color='red')
            
            ax.set_title(f'Equity Curves - {symbol}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Capital')
            ax.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            equity_curve_plots.append(fig)

    return BacktestResults(
        trades_df=combined_trades,
        summary_metrics=overall_metrics,
        symbol_metrics=metrics_df,
        equity_curves=equity_curves,
        equity_curve_plots=equity_curve_plots if return_equity_curves else None  # Add new field
    )