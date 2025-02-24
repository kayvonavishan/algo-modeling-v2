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
                stop_loss_scale_coeff: float = 25.0
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
        # Initialize historical_trades with all expected columns
        self.historical_trades = pd.DataFrame(columns=[
            'symbol', 'buy_timestamp', 'sell_timestamp', 'buy_price', 'sell_price',
            'position_size', 'return', 'return_percentage', 'capital', 
            'hold_time_hours', 'predicted_class', 'is_partial',
            'trailing_ema_return_long', 'trailing_ema_return_short'
        ])

    import numpy as np
    import pandas as pd

    def get_trailing_ema_return(self, current_idx: int, market_df: pd.DataFrame, is_short: bool) -> float:
        """
        Calculate the trailing EMA return for the given position type (long or short) using historical trades.
        This logic is similar to the one used in calculate_kelly_fraction but groups only by position type.
        
        Args:
            current_idx: Current index in the market data.
            market_df: Full market dataframe.
            is_short: True for short positions, False for long positions.
        
        Returns:
            float: The trailing EMA return (as a decimal value, e.g. 0.05 for 5%).
        """
        cutoff_time = market_df.index[current_idx]
        # Filter historical trades by position type (ignoring predicted class)
        trades_pos = self.historical_trades[
            (self.historical_trades['is_short'] == is_short) &
            (self.historical_trades['sell_timestamp'] < cutoff_time)
        ].tail(self.lookback_period_for_position_size)
        
        # If there are insufficient trades, use all available trades of this position type.
        if len(trades_pos) < 10:
            trades_pos = self.historical_trades[
                (self.historical_trades['is_short'] == is_short) &
                (self.historical_trades['sell_timestamp'] < cutoff_time)
            ].tail(self.lookback_period_for_position_size)
            if trades_pos.empty:
                # If still empty, return a default (you can adjust this default as needed)
                return 0.0

        trades_pos = trades_pos.sort_values('sell_timestamp')
        ewm_return = trades_pos['return_percentage'].ewm(span=len(trades_pos), adjust=False).mean()
        avg_return_ewm = ewm_return.iloc[-1] / 100.0  # converting percentage to decimal
        return avg_return_ewm


    def calculate_kelly_fraction(self, pred_class: int, current_idx: int, market_df: pd.DataFrame,
                                kelly_debug: bool = False, is_short: bool = False) -> float:
        """
        Calculate Kelly fraction based on historical performance, but for now grouping only by 
        position type (long/short) rather than by predicted class.
        
        Args:
            pred_class: The predicted class for this trade (currently not used in filtering)
            current_idx: Current index in the market data
            market_df: Full market dataframe
            kelly_debug: Whether to print debug information
            is_short: Whether this calculation is for a short position
        
        Returns:
            float: Kelly fraction (position size as percentage of capital)
        """
        # Basic validation checks
        if 'predicted_class' not in self.historical_trades.columns:
            if self.debug or kelly_debug:
                print("[WARNING] 'predicted_class' column missing in historical_trades. Returning default Kelly=0.5")
            return 0.5

        if self.historical_trades.empty:
            if self.debug or kelly_debug:
                print("[WARNING] historical_trades is empty. Returning default Kelly=0.25")
            return 0.25

        cutoff_time = market_df.index[current_idx]
        # Originally, we filtered by predicted class as well:
        # historical_class_trades = self.historical_trades[
        #     (self.historical_trades['predicted_class'] == pred_class) & 
        #     (self.historical_trades['is_short'] == is_short) &
        #     (self.historical_trades['sell_timestamp'] < cutoff_time)
        # ].tail(self.lookback_period_for_position_size)
        
        # New version: Group only by position type (long/short)
        historical_class_trades = self.historical_trades[
            (self.historical_trades['is_short'] == is_short) &
            (self.historical_trades['sell_timestamp'] < cutoff_time)
        ].tail(self.lookback_period_for_position_size)

        if kelly_debug and current_idx < 2000:
            # The following line is commented out since we are no longer grouping by predicted class:
            # print(f"Predicted Class: {pred_class}")
            print(f"Position Type: {'Short' if is_short else 'Long'}")
            print(f"Number of historical trades (grouped only by position): {len(historical_class_trades)}")

        # If insufficient history for this position type, try to use all trades for the position type
        if len(historical_class_trades) < 10:
            # The original fallback that filtered by predicted class is commented out:
            # all_class_trades = self.historical_trades[
            #     (self.historical_trades['predicted_class'] == pred_class) &
            #     (self.historical_trades['sell_timestamp'] < cutoff_time)
            # ].tail(self.lookback_period_for_position_size)
            #
            # Instead, we use all trades for the position type:
            all_position_trades = self.historical_trades[
                (self.historical_trades['is_short'] == is_short) &
                (self.historical_trades['sell_timestamp'] < cutoff_time)
            ].tail(self.lookback_period_for_position_size)
            
            if len(all_position_trades) < 10:
                if kelly_debug and current_idx < 2000:
                    print("Insufficient historical trades in all categories - using default = 0.25")
                return 0.25
            
            if kelly_debug and current_idx < 2000:
                print(f"Using all {len(all_position_trades)} trades of same position type")
            historical_class_trades = all_position_trades

        # Sort by sell_timestamp for proper EMA calculation
        historical_class_trades = historical_class_trades.sort_values('sell_timestamp')

        # ============ Simpler Position Sizing (EMA) ============
        ewm_return = historical_class_trades['return_percentage'].ewm(
            span=len(historical_class_trades),
            adjust=False
        ).mean()
        avg_return_ewm = ewm_return.iloc[-1] / 100.0  # convert % to decimal

        if self.use_simpler_position_sizing:
            # Determine position size based on return ranges
            if avg_return_ewm < 0:
                position_size = 0.01
            elif avg_return_ewm < 0.05:  # 0 to 5%
                position_size = 0.05
            elif avg_return_ewm < 0.10:  # 5% to 10%
                position_size = 0.15
            elif avg_return_ewm < 0.15:  # 10% to 15%
                position_size = 0.30
            else:  # > 15%
                position_size = 1.0

            if kelly_debug and current_idx < 2000:
                print("\nUsing simpler position sizing:")
                print(f"Position Type: {'Short' if is_short else 'Long'}")
                print(f"EMA-based Average Return: {avg_return_ewm:.4f}")
                print(f"Position Size: {position_size:.2f}")
                print(f"Return Range: {avg_return_ewm * 100:.2f}%")

            return position_size

        # ============ Kelly-based Sizing (Using EMA) ============
        # Calculate win probability using EMA
        winners_mask = historical_class_trades['return'] > 0
        win_01_series = winners_mask.astype(float)
        ewm_win_prob = win_01_series.ewm(
            span=len(win_01_series),
            adjust=False
        ).mean()
        win_prob_ewm = ewm_win_prob.iloc[-1]  # fraction in [0..1]

        # Calculate average win size using EMA
        pos_trades = historical_class_trades[winners_mask]
        ewm_avg_win = (
            pos_trades['return_percentage'].ewm(
                span=len(pos_trades),
                adjust=False
            ).mean().iloc[-1] / 100.0 if len(pos_trades) > 0 else 0.0
        )

        # Calculate average loss size using EMA
        losers_mask = historical_class_trades['return'] <= 0
        neg_trades = historical_class_trades[losers_mask]
        ewm_avg_loss = (
            abs(neg_trades['return_percentage'].ewm(
                span=len(neg_trades),
                adjust=False
            ).mean().iloc[-1] / 100.0) if len(neg_trades) > 0 else 0.0
        )

        if kelly_debug and current_idx < 2000:
            print(f"Position Type: {'Short' if is_short else 'Long'}")
            print(f"Win Probability (EMA): {win_prob_ewm:.3f}")
            print(f"Average Win (EMA): {ewm_avg_win:.3f}")
            print(f"Average Loss (EMA): {ewm_avg_loss:.3f}")

        # Handle edge case where there's no average loss
        if ewm_avg_loss == 0:
            if kelly_debug and current_idx < 2000:
                print("Average loss is 0 - using default kelly = 0.75")
            return 0.75

        # Kelly formula (remains unchanged)
        kelly_raw = win_prob_ewm - ((1 - win_prob_ewm) / (ewm_avg_win / ewm_avg_loss))
        return_scalar = np.power(1.0 + (avg_return_ewm / self.kelly_multiplier), 2)
        kelly = kelly_raw * return_scalar

        # Clip between 0.01 and 1.0
        kelly = max(0.01, min(kelly, 1.0))

        if kelly_debug and current_idx < 2000:
            print(f"Raw Kelly (EMA-based): {kelly_raw:.3f}")
            print(f"Scaled Kelly: {kelly:.3f}")

        return kelly



    def calculate_trailing_metrics(self, market_df: pd.DataFrame, current_idx: int) -> Dict:
        """
        Calculate trailing metrics up to current_idx, with separate tracking for long and short positions.
        
        Returns:
            Dict with structure:
            {
                'long': {
                    pred_class: pd.Series of EMAs
                },
                'short': {
                    pred_class: pd.Series of EMAs
                },
                'combined': {
                    pred_class: pd.Series of EMAs
                }
            }
        """
        metrics = {
            'long': {},
            'short': {},
            'combined': {}
        }
        
        cutoff_time = market_df.index[current_idx]
        
        # First, calculate combined metrics (original logic)
        for pred_class in self.historical_trades['predicted_class'].unique():
            class_trades = self.historical_trades[
                (self.historical_trades['predicted_class'] == pred_class) &
                (self.historical_trades['sell_timestamp'] < cutoff_time)
            ]
            
            if len(class_trades) == 0:
                continue
                
            daily_returns = class_trades.groupby('buy_timestamp')['return_percentage'].mean()
            
            # Create series aligned with market data up to current_idx
            continuous_series = pd.Series(
                index=market_df.index[:current_idx+1], 
                dtype=float
            )
            continuous_series[daily_returns.index] = daily_returns
            continuous_series = continuous_series.ffill()
            
            # Calculate EMA for combined positions
            ema = continuous_series.ewm(
                span=self.ema_period, 
                adjust=False
            ).mean()
            
            metrics['combined'][pred_class] = ema
            
            # Now calculate separate metrics for long and short positions
            for position_type, is_short in [('long', False), ('short', True)]:
                position_trades = class_trades[class_trades['is_short'] == is_short]
                
                if len(position_trades) == 0:
                    continue
                    
                position_daily_returns = position_trades.groupby('buy_timestamp')['return_percentage'].mean()
                
                # Create series aligned with market data
                position_continuous_series = pd.Series(
                    index=market_df.index[:current_idx+1], 
                    dtype=float
                )
                position_continuous_series[position_daily_returns.index] = position_daily_returns
                position_continuous_series = position_continuous_series.ffill()
                
                # Calculate EMA for this position type
                position_ema = position_continuous_series.ewm(
                    span=self.ema_period, 
                    adjust=False
                ).mean()
                
                metrics[position_type][pred_class] = position_ema
                
                if self.debug:
                    avg_return = position_ema.mean() if not position_ema.empty else 0
                    print(f"\n[DEBUG] Trailing metrics for {position_type} positions, class {pred_class}:")
                    print(f"Number of trades: {len(position_trades)}")
                    print(f"Average EMA return: {avg_return:.2f}%")
        
        # Add performance comparison metrics
        metrics['performance_comparison'] = {}
        for pred_class in metrics['combined'].keys():
            if pred_class in metrics['long'] and pred_class in metrics['short']:
                # Calculate relative performance between long and short
                long_ema = metrics['long'][pred_class]
                short_ema = metrics['short'][pred_class]
                
                # Only compare where we have both long and short data
                valid_indices = long_ema.notna() & short_ema.notna()
                if valid_indices.any():
                    relative_performance = pd.Series(
                        index=long_ema[valid_indices].index,
                        data=long_ema[valid_indices] - short_ema[valid_indices]
                    )
                    
                    metrics['performance_comparison'][pred_class] = {
                        'relative_performance': relative_performance,
                        'long_better': relative_performance > 0,
                        'short_better': relative_performance < 0
                    }
                    
                    if self.debug:
                        better_strategy = 'LONG' if relative_performance.mean() > 0 else 'SHORT'
                        print(f"\n[DEBUG] For class {pred_class}, {better_strategy} performing better")
                        print(f"Average performance difference: {abs(relative_performance.mean()):.2f}%")
        
        # Add volatility metrics
        metrics['volatility'] = {}
        for position_type in ['long', 'short', 'combined']:
            metrics['volatility'][position_type] = {}
            for pred_class, ema_series in metrics[position_type].items():
                if not ema_series.empty:
                    # Calculate rolling standard deviation of EMA
                    rolling_std = ema_series.rolling(
                        window=self.ema_period,
                        min_periods=1
                    ).std()
                    
                    metrics['volatility'][position_type][pred_class] = rolling_std
                    
                    if self.debug:
                        avg_vol = rolling_std.mean()
                        print(f"\n[DEBUG] {position_type.capitalize()} position volatility for class {pred_class}:")
                        print(f"Average volatility: {avg_vol:.2f}%")
        
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

    def calculate_stop_losses(
        self, 
        close_prices: np.ndarray, 
        close_price_atr_28_percent_change: np.ndarray, 
        close_above_sma_200: np.ndarray,
        valid_buys: np.ndarray, 
        sells: np.ndarray, 
        atr_coeff: float,
        historical_trades: pd.DataFrame,
        lookback_period: int,
        use_performance_scaling: bool = True,
        min_trades_required: int = 10,
        stop_loss_scale_coeff: float = 1.0,
        max_scale_up: float = 2.0,
        min_scale_down: float = 0.5
    ):
        """
        Calculate stop losses for both long and short positions, maintaining separate
        performance tracking for each position type.
        """
        if self.debug:
            scaling_method = "performance-based" if use_performance_scaling else "SMA-based"
            print(f"[DEBUG] Calculating stop losses using {scaling_method} scaling")

        # Initialize arrays
        sell_stop_loss = np.zeros_like(valid_buys)
        long_position_open = False
        short_position_open = False
        max_price_since_entry = 0.0
        min_price_since_entry = float('inf')

        # Calculate base thresholds
        base_thresholds = close_price_atr_28_percent_change * atr_coeff

        # Performance-based scaling logic
        if use_performance_scaling and historical_trades is not None and not historical_trades.empty:
            for index in range(len(valid_buys)):
                current_price = close_prices[index]
                
                # Determine position type and calculate appropriate scale multiplier
                scale_multiplier = 1.0

                if long_position_open:
                    position_type = False  # False = long
                elif short_position_open:
                    position_type = True   # True = short
                else:
                    position_type = None

                if position_type is not None:
                    # Handle the case where we don't have any timestamps yet
                    try:
                        current_time = historical_trades.index[index]
                    except IndexError:
                        current_time = None
                    
                    if current_time is not None:
                        # Get recent trades for the current position type
                        recent_trades = historical_trades[
                            (historical_trades['sell_timestamp'] < current_time) &
                            (historical_trades['is_short'] == position_type)
                        ].copy()
                        
                        if not recent_trades.empty:
                            recent_trades.sort_values('sell_timestamp', inplace=True)
                            recent_trades = recent_trades.tail(lookback_period)

                            if len(recent_trades) >= min_trades_required:
                                # Calculate EMA of returns for this position type
                                ewm_return = recent_trades['return_percentage'].ewm(
                                    span=len(recent_trades), 
                                    adjust=False
                                ).mean()
                                avg_return_ewm = ewm_return.iloc[-1]

                                scale_multiplier = 1.0 + (avg_return_ewm * stop_loss_scale_coeff / 100.0)
                                scale_multiplier = np.clip(scale_multiplier, min_scale_down, max_scale_up)

                                if self.debug:
                                    print(f"[DEBUG] Index {index}: Position={'Short' if position_type else 'Long'}, "
                                        f"EMA return: {avg_return_ewm:.2f}%, Scale: {scale_multiplier:.2f}")

                threshold = base_thresholds[index] * scale_multiplier

                # Long position management
                if not long_position_open and not short_position_open and valid_buys[index] == 1:
                    long_position_open = True
                    max_price_since_entry = current_price

                # Short position management
                elif not short_position_open and not long_position_open and sells[index] == 1:
                    short_position_open = True
                    min_price_since_entry = current_price

                # Update tracking prices and check stops
                if long_position_open:
                    if current_price > max_price_since_entry:
                        max_price_since_entry = current_price

                    # Long stop loss - triggered when price falls below threshold
                    percent_change = (current_price - max_price_since_entry) / max_price_since_entry * 100
                    if percent_change < threshold:
                        sell_stop_loss[index] = 1
                        long_position_open = False

                elif short_position_open:
                    if current_price < min_price_since_entry:
                        min_price_since_entry = current_price

                    # Short stop loss - triggered when price rises above threshold
                    percent_change = (current_price - min_price_since_entry) / min_price_since_entry * 100
                    if percent_change > abs(threshold):  # Note: using abs() since threshold is negative
                        sell_stop_loss[index] = 1
                        short_position_open = False

                # Handle regular position closes
                if (long_position_open and sells[index] == 1):
                    long_position_open = False
                elif (short_position_open and valid_buys[index] == 1):
                    short_position_open = False

        else:
            # Original SMA-based scaling logic (no historical trades needed)
            multipliers = np.where(close_above_sma_200 == 1, self.stop_loss_adjust, 1.0)
            stoploss_thresholds = base_thresholds * multipliers

            for index in range(len(valid_buys)):
                current_price = close_prices[index]
                threshold = stoploss_thresholds[index]

                # Long position management
                if not long_position_open and not short_position_open and valid_buys[index] == 1:
                    long_position_open = True
                    max_price_since_entry = current_price

                # Short position management
                elif not short_position_open and not long_position_open and sells[index] == 1:
                    short_position_open = True
                    min_price_since_entry = current_price

                # Track prices and check stops
                if long_position_open:
                    if current_price > max_price_since_entry:
                        max_price_since_entry = current_price

                    percent_change = (current_price - max_price_since_entry) / max_price_since_entry * 100
                    if percent_change < threshold:
                        sell_stop_loss[index] = 1
                        long_position_open = False

                elif short_position_open:
                    if current_price < min_price_since_entry:
                        min_price_since_entry = current_price

                    percent_change = (current_price - min_price_since_entry) / min_price_since_entry * 100
                    if percent_change > abs(threshold):
                        sell_stop_loss[index] = 1
                        short_position_open = False

                # Handle regular position closes
                if (long_position_open and sells[index] == 1):
                    long_position_open = False
                elif (short_position_open and valid_buys[index] == 1):
                    short_position_open = False

        return sell_stop_loss

    def execute_trades(self, timestamps: np.ndarray, close_prices: np.ndarray, 
                        valid_buys: np.ndarray, sells: np.ndarray, combined_sells: np.ndarray, 
                        symbol: str, predicted_classes: np.ndarray, 
                        market_df: pd.DataFrame, sell_stop_loss: np.ndarray,
                        kelly_debug: bool = False) -> pd.DataFrame:
        if self.debug:
            print(f"[DEBUG] Executing trades for symbol: {symbol}")

        trades_list = []
        capital = self.initial_capital
        position_open = False
        is_short = False  # current position type
        predicted_class_at_entry = None
        entry_timestamp = None

        # Reset position tracking
        self.current_position_size = 0.0
        self.current_entry_price = 0.0

        # Determine the highest prediction class column (for metrics later)
        highest_class = max([int(col.split('_')[-1]) 
                            for col in market_df.columns 
                            if col.startswith('prediction_raw_class_')])
        highest_class_col = f'prediction_raw_class_{highest_class}'

        for index in range(len(close_prices)):
            current_price = close_prices[index]
            timestamp = timestamps[index]
            current_time = pd.Timestamp(timestamp).time()
            current_timestamp = pd.Timestamp(timestamp)

            # ===== Full Liquidation on Friday 15:45 =====
            if current_timestamp.weekday() == 4 and current_time.hour == 15 and current_time.minute == 45:
                if position_open:
                    # Fully close any open position (long or short)
                    if is_short:
                        profit = (self.current_entry_price - current_price) / self.current_entry_price * abs(self.current_position_size)
                    else:
                        profit = (current_price - self.current_entry_price) / self.current_entry_price * self.current_position_size
                    capital += profit
                    trade = {
                        'symbol': symbol,
                        'buy_timestamp': entry_timestamp,
                        'sell_timestamp': timestamp,
                        'is_short': is_short,
                        'buy_price': self.current_entry_price,
                        'sell_price': current_price,
                        'position_size': abs(self.current_position_size),
                        'return': profit,
                        'return_percentage': (
                            (self.current_entry_price - current_price) / self.current_entry_price * 100
                            if is_short else
                            (current_price - self.current_entry_price) / self.current_entry_price * 100
                        ),
                        'capital': capital,
                        'hold_time_hours': (timestamp - entry_timestamp) / np.timedelta64(1, 'h'),
                        'predicted_class': predicted_class_at_entry,
                        'highest_class_probability': market_df.iloc[index][highest_class_col],
                        'is_partial': False,
                        'trailing_ema_return_long': self.get_trailing_ema_return(index, market_df, is_short=False),
                        'trailing_ema_return_short': self.get_trailing_ema_return(index, market_df, is_short=True)
                    }
                    trades_list.append(trade)
                    if self.debug:
                        print(f"[DEBUG] Friday 15:45 Full Liquidation at {timestamp}:")
                        print(f"Liquidated full position of size {abs(self.current_position_size):.2f}, Profit: {profit:.2f}, Capital: {capital:.2f}")
                    # Reset position tracking variables
                    self.current_position_size = 0.0
                    self.current_entry_price = 0.0
                    position_open = False
                # Skip further processing for this bar
                continue

            # ===== End Friday 15:45 liquidation check =====

            # Define local signal booleans (using the model's raw signals and stop loss values)
            model_buy = (valid_buys[index] == 1)
            model_sell = (sells[index] == 1)
            # Here we assume stop_loss_signal is provided via the sell_stop_loss array already computed.
            stop_loss_signal = (sell_stop_loss[index] == 1)

            # (Then follow your existing reversal and entry/exit logic...)
            if position_open:
                reversal_flag = False
                reversal_action = None
                exit_flag = False

                if not is_short:  # currently long
                    if model_sell:
                        reversal_flag = True
                        reversal_action = 'long_to_short'
                        exit_flag = True
                    elif (not model_sell) and (sell_stop_loss[index] == 1):
                        exit_flag = True  # normal exit due to stop loss; no reversal
                else:  # currently short
                    if model_buy:
                        reversal_flag = True
                        reversal_action = 'short_to_long'
                        exit_flag = True
                    elif (not model_buy) and (sell_stop_loss[index] == 1):
                        exit_flag = True  # normal exit due to stop loss; no reversal

                if exit_flag:
                    # Close the existing position
                    if is_short:
                        profit = (self.current_entry_price - current_price) / self.current_entry_price * abs(self.current_position_size)
                    else:
                        profit = (current_price - self.current_entry_price) / self.current_entry_price * self.current_position_size
                    capital += profit

                    trade = {
                        'symbol': symbol,
                        'buy_timestamp': entry_timestamp,
                        'sell_timestamp': timestamp,
                        'is_short': is_short,
                        'buy_price': self.current_entry_price,
                        'sell_price': current_price,
                        'position_size': abs(self.current_position_size),
                        'return': profit,
                        'return_percentage': (
                            (self.current_entry_price - current_price) / self.current_entry_price * 100
                            if is_short else
                            (current_price - self.current_entry_price) / self.current_entry_price * 100
                        ),
                        'capital': capital,
                        'hold_time_hours': (timestamp - entry_timestamp) / np.timedelta64(1, 'h'),
                        'predicted_class': predicted_class_at_entry,
                        'highest_class_probability': market_df.iloc[index][highest_class_col],
                        'is_partial': False,
                        'trailing_ema_return_long': self.get_trailing_ema_return(index, market_df, is_short=False),
                        'trailing_ema_return_short': self.get_trailing_ema_return(index, market_df, is_short=True)
                    }
                    trades_list.append(trade)
                    self.historical_trades = pd.DataFrame(trades_list)

                    # Reset current trade variables
                    self.current_position_size = 0.0
                    self.current_entry_price = 0.0
                    position_open = False

                    if reversal_flag:
                        # Immediately open the reversed trade using the model signal.
                        predicted_class_at_entry = predicted_classes[index]
                        if reversal_action == 'long_to_short':
                            kelly_fraction = self.calculate_kelly_fraction(
                                pred_class=predicted_class_at_entry,
                                current_idx=index,
                                market_df=market_df,
                                kelly_debug=kelly_debug,
                                is_short=True
                            )
                            position_open = True
                            is_short = True
                            entry_timestamp = timestamp
                            self.current_entry_price = current_price
                            self.current_position_size = -capital * kelly_fraction
                        elif reversal_action == 'short_to_long':
                            kelly_fraction = self.calculate_kelly_fraction(
                                pred_class=predicted_class_at_entry,
                                current_idx=index,
                                market_df=market_df,
                                kelly_debug=kelly_debug,
                                is_short=False
                            )
                            position_open = True
                            is_short = False
                            entry_timestamp = timestamp
                            self.current_entry_price = current_price
                            self.current_position_size = capital * kelly_fraction
            else:
                # No position is open: Check for fresh entries.
                if model_buy:
                    predicted_class_at_entry = predicted_classes[index]
                    kelly_fraction = self.calculate_kelly_fraction(
                        pred_class=predicted_class_at_entry,
                        current_idx=index,
                        market_df=market_df,
                        kelly_debug=kelly_debug,
                        is_short=False
                    )
                    position_open = True
                    is_short = False
                    entry_timestamp = timestamp
                    self.current_entry_price = current_price
                    self.current_position_size = capital * kelly_fraction
                    if self.debug:
                        print(f"[DEBUG] Opening LONG position at {timestamp}, price: {current_price}")
                elif model_sell:
                    predicted_class_at_entry = predicted_classes[index]
                    kelly_fraction = self.calculate_kelly_fraction(
                        pred_class=predicted_class_at_entry,
                        current_idx=index,
                        market_df=market_df,
                        kelly_debug=kelly_debug,
                        is_short=True
                    )
                    position_open = True
                    is_short = True
                    entry_timestamp = timestamp
                    self.current_entry_price = current_price
                    self.current_position_size = -capital * kelly_fraction
                    if self.debug:
                        print(f"[DEBUG] Opening SHORT position at {timestamp}, price: {current_price}")

            # Portfolio tracking (update capital over time)
            if len(self.portfolio_timestamps) == 0 or self.portfolio_timestamps[-1] != timestamp:
                self.portfolio_timestamps.append(timestamp)
                self.portfolio_value.append(capital)

        # Final update to historical trades
        if trades_list:
            self.historical_trades = pd.DataFrame(trades_list)
            return pd.DataFrame(trades_list)
        return pd.DataFrame(columns=[
            'symbol', 'buy_timestamp', 'sell_timestamp', 'is_short',
            'buy_price', 'sell_price', 'position_size', 'return',
            'return_percentage', 'capital', 'hold_time_hours',
            'predicted_class', 'highest_class_probability', 'is_partial',
            'trailing_ema_return_long', 'trailing_ema_return_short'
        ])




    def calculate_metrics(self, trades_df: pd.DataFrame, symbol: str, 
                        initial_price: float, final_price: float, df: pd.DataFrame) -> Dict:
        """
        Calculate trading metrics with separate tracking for long and short positions.
        """
        if self.debug:
            print(f"[DEBUG] Calculating metrics for symbol: {symbol}")

        if trades_df.empty:
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
                'partial_trades_count': 0,
                # New metrics for long/short
                'long_trades_count': 0,
                'short_trades_count': 0,
                'long_returns_percentage': 0,
                'short_returns_percentage': 0,
                'long_win_rate': 0,
                'short_win_rate': 0
            }

        # Separate complete and partial trades
        complete_trades = trades_df[~trades_df['is_partial']]
        partial_trades = trades_df[trades_df['is_partial']]

        # Separate long and short trades
        long_trades = trades_df[~trades_df['is_short']]
        short_trades = trades_df[trades_df['is_short']]

        # Calculate win rates for each position type
        long_wins = (long_trades['return'] > 0).sum()
        long_losses = (long_trades['return'] <= 0).sum()
        short_wins = (short_trades['return'] > 0).sum()
        short_losses = (short_trades['return'] <= 0).sum()

        long_win_rate = long_wins / len(long_trades) if len(long_trades) > 0 else 0
        short_win_rate = short_wins / len(short_trades) if len(short_trades) > 0 else 0

        # Overall win/loss ratio combining both types
        total_wins = long_wins + short_wins
        total_losses = long_losses + short_losses
        win_loss_ratio = total_wins / total_losses if total_losses > 0 else float('inf')

        # Calculate overnight trade metrics
        overnight_trades = trades_df[
            (trades_df['sell_timestamp'].dt.time == pd.Timestamp('15:45').time()) &
            trades_df['is_partial']
        ]

        # Separate overnight trades by position type
        long_overnight = overnight_trades[~overnight_trades['is_short']]
        short_overnight = overnight_trades[overnight_trades['is_short']]

        # Calculate Sharpe Ratio
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

        # Calculate average position sizes
        avg_position_size_long = long_trades['position_size'].mean() / self.initial_capital * 100 if not long_trades.empty else 0
        avg_position_size_short = short_trades['position_size'].mean() / self.initial_capital * 100 if not short_trades.empty else 0
        position_size_std_long = long_trades['position_size'].std() / self.initial_capital * 100 if not long_trades.empty else 0
        position_size_std_short = short_trades['position_size'].std() / self.initial_capital * 100 if not short_trades.empty else 0

        metrics = {
            'symbol': symbol,
            'total_returns_percentage': (trades_df['capital'].iloc[-1] - self.initial_capital) / self.initial_capital * 100,
            'win_loss_ratio': win_loss_ratio,
            'average_percent_return': trades_df['return_percentage'].mean(),
            'average_hold_time_hours': trades_df['hold_time_hours'].mean(),
            'buy_and_hold_return_percentage': ((final_price - initial_price) / initial_price) * 100,
            'number_of_trades': len(trades_df),
            'buy_and_hold_hold_time_hours': (df.index[-1] - df.index[0]).total_seconds() / 3600.0,
            'trading_hold_time_hours': trades_df['hold_time_hours'].sum(),
            'sharpe_ratio': sharpe_ratio,

            # Long position metrics
            'long_trades_count': len(long_trades),
            'long_returns_percentage': long_trades['return_percentage'].mean() if not long_trades.empty else 0,
            'long_win_rate': long_win_rate,
            'avg_position_size_long': avg_position_size_long,
            'position_size_std_long': position_size_std_long,

            # Short position metrics
            'short_trades_count': len(short_trades),
            'short_returns_percentage': short_trades['return_percentage'].mean() if not short_trades.empty else 0,
            'short_win_rate': short_win_rate,
            'avg_position_size_short': avg_position_size_short,
            'position_size_std_short': position_size_std_short,

            # Overnight metrics
            'overnight_trades_count': len(overnight_trades),
            'overnight_trades_return_percentage': overnight_trades['return_percentage'].mean() if not overnight_trades.empty else 0,
            'long_overnight_count': len(long_overnight),
            'short_overnight_count': len(short_overnight),
            'long_overnight_return': long_overnight['return_percentage'].mean() if not long_overnight.empty else 0,
            'short_overnight_return': short_overnight['return_percentage'].mean() if not short_overnight.empty else 0,

            # Class-based metrics with long/short separation
            'class_metrics': trades_df.groupby(['predicted_class', 'is_short', 'is_partial']).agg({
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
            return 'buy_final_short', 'sell_short'
        else:
            if debug:
                print("Using regular signals due to positive/neutral performance")
            return 'buy_final', 'sell'


    def backtest_symbol(self, df: pd.DataFrame, symbol: str, atr_coeff: float, kelly_debug: bool = False) -> tuple:
        """
        Backtest a single symbol with support for both long and short positions.
        """
        if self.debug:
            print(f"[DEBUG] Backtesting symbol: {symbol}")
            print(f"[DEBUG] Data shape: {df.shape}")
            print(f"[DEBUG] Columns available: {df.columns}")
            print(f"[DEBUG] Buy signals count: {df['buy_final'].sum()}")
            print(f"[DEBUG] Sell signals count: {df['sell'].sum()}")

        # Reset historical trades at start of new symbol
        self.historical_trades = pd.DataFrame(columns=[
            'symbol', 'buy_timestamp', 'sell_timestamp', 'buy_price', 'sell_price',
            'position_size', 'return', 'return_percentage', 'capital',
            'hold_time_hours', 'predicted_class', 'is_short', 'is_partial',
            'trailing_ema_return_long', 'trailing_ema_return_short'
        ])

        # Validate required columns
        required_columns = ['buy_final', 'sell']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Required columns missing for symbol {symbol}: {missing_columns}")

        # Validate price columns
        price_columns = ['close_raw', 'high_raw', 'low_raw']
        missing_price_cols = [col for col in price_columns if col not in df.columns]
        if missing_price_cols:
            raise ValueError(f"Required price columns missing for symbol {symbol}: {missing_price_cols}")

        # Validate prediction column
        if 'predicted_class_moving_avg' not in df.columns:
            raise ValueError(f"Required prediction column 'predicted_class_moving_avg' missing for symbol {symbol}")

        # Calculate ATR metrics
        df = self.calculate_atr_metrics(df)

        # Initialize arrays
        timestamps = df.index.to_numpy()
        buys = np.zeros(len(df), dtype=np.int8)
        sells = np.zeros(len(df), dtype=np.int8)

        # First get the base signals
        for i in range(len(df)):
            buys[i] = df['buy_final'].iloc[i]
            sells[i] = df['sell'].iloc[i]

        # Process valid buys
        valid_buys = self.process_valid_buys(buys, sells)

        # Track current position state
        position_state = {
            'long_open': False,
            'short_open': False,
            'last_close_was_stop': False,
            'waiting_for_signal': False
        }

        # Now process signals based on position state
        for i in range(len(df)):
            current_time = timestamps[i]
            
            # Update position state based on signals
            if valid_buys[i] == 1:
                position_state['long_open'] = True
                position_state['short_open'] = False
            elif sells[i] == 1:
                position_state['long_open'] = False
                position_state['short_open'] = True
            
            if position_state['waiting_for_signal']:
                if position_state['last_close_was_stop']:
                    if position_state['long_open']:
                        valid_buys[i] = 0
                        sells[i] = df['sell'].iloc[i]
                    elif position_state['short_open']:
                        valid_buys[i] = df['buy_final'].iloc[i]
                        sells[i] = 0
                position_state['waiting_for_signal'] = False

            if self.debug and (valid_buys[i] or sells[i]):
                print(f"\n[DEBUG] At {current_time}:")
                print(f"Position state: {position_state}")
                print(f"Signals - Buy: {valid_buys[i]}, Sell: {sells[i]}")
        
        # Prepare price data
        close_prices = df['close_raw'].astype(np.float64).values
        close_price_atr_28_percent_change = df['close_price_atr_28_percent_change'].astype(np.float64).values
        close_above_sma_200 = df['close_above_sma_200'].astype(np.int8).values

        # Calculate stop losses with position type awareness
        sell_stop_loss = self.calculate_stop_losses(
            close_prices, 
            close_price_atr_28_percent_change, 
            close_above_sma_200, 
            valid_buys, 
            sells, 
            atr_coeff,
            historical_trades=self.historical_trades,
            lookback_period=self.lookback_period_for_position_size,
            use_performance_scaling=self.use_performance_scaling,
            stop_loss_scale_coeff=self.stop_loss_scale_coeff
        )

        # Update position state based on stop losses
        for i in range(len(df)):
            if sell_stop_loss[i] == 1:
                position_state['last_close_was_stop'] = True
                position_state['waiting_for_signal'] = True
                if position_state['long_open']:
                    position_state['long_open'] = False
                elif position_state['short_open']:
                    position_state['short_open'] = False

        # Combined sells now include stop losses
        combined_sells = np.logical_or(sells == 1, sell_stop_loss == 1)

        # Get predicted classes for position sizing
        predicted_classes = df['predicted_class_moving_avg'].values

        # Execute trades with enhanced position tracking
        trade_data = self.execute_trades(
            timestamps, close_prices, valid_buys, sells, combined_sells, 
            symbol, predicted_classes, df, sell_stop_loss, kelly_debug=kelly_debug
        )

        trades_df = trade_data if not trade_data.empty else pd.DataFrame()

        # Calculate metrics with position type awareness
        metrics = self.calculate_metrics(
            trades_df,
            symbol,
            close_prices[0],
            close_prices[-1],
            df
        )

        if self.debug:
            print(f"\n[DEBUG] Calculated metrics for {symbol}")
            print("Position-specific metrics:")
            if 'long_trades_count' in metrics:
                print(f"Long trades: {metrics['long_trades_count']}")
                print(f"Long returns: {metrics['long_returns_percentage']:.2f}%")
            if 'short_trades_count' in metrics:
                print(f"Short trades: {metrics['short_trades_count']}")
                print(f"Short returns: {metrics['short_returns_percentage']:.2f}%")

        # Add signal information to DataFrame
        df['sell_stop_loss'] = sell_stop_loss
        df['valid_buys'] = valid_buys
        df['combined_sells'] = combined_sells
        
        # Add position type tracking columns with improved logic
        df['is_long_position'] = False
        df['is_short_position'] = False
        current_position = None
        
        for i in range(len(df)):
            if valid_buys[i]:
                current_position = 'long'
            elif (combined_sells[i] or sell_stop_loss[i]) and current_position == 'long':
                current_position = None
            elif sells[i]:
                current_position = 'short'
            elif (valid_buys[i] or sell_stop_loss[i]) and current_position == 'short':
                current_position = None
                
            df.iloc[i, df.columns.get_loc('is_long_position')] = current_position == 'long'
            df.iloc[i, df.columns.get_loc('is_short_position')] = current_position == 'short'

        # Create equity curve with improved position tracking
        if len(self.portfolio_timestamps) > 0 and len(self.portfolio_value) > 0:
            equity_curve_df = pd.DataFrame({
                'timestamp': self.portfolio_timestamps,
                'capital': self.portfolio_value,
                'symbol': symbol
            })
            equity_curve_df.sort_values('timestamp', inplace=True)
            
            # Create separate equity curves for long and short positions
            long_capital = self.initial_capital
            short_capital = self.initial_capital
            long_values = []
            short_values = []
            
            for i, trade in trades_df.iterrows():
                if trade['is_short']:
                    short_capital += trade['return']
                    short_values.append((trade['sell_timestamp'], short_capital))
                else:
                    long_capital += trade['return']
                    long_values.append((trade['sell_timestamp'], long_capital))
            
            # Create separate DataFrames for long and short positions
            long_equity_df = pd.DataFrame(long_values, columns=['timestamp', 'capital']) if long_values else pd.DataFrame()
            short_equity_df = pd.DataFrame(short_values, columns=['timestamp', 'capital']) if short_values else pd.DataFrame()
            
            # Forward fill the values using ffill() instead of fillna(method='ffill')
            if not long_equity_df.empty:
                long_equity_df = long_equity_df.set_index('timestamp').reindex(equity_curve_df['timestamp']).ffill()
            if not short_equity_df.empty:
                short_equity_df = short_equity_df.set_index('timestamp').reindex(equity_curve_df['timestamp']).ffill()

            # Combine into single equity curve DataFrame with position type
            equity_curve_df['long_capital'] = long_equity_df['capital'] if not long_equity_df.empty else self.initial_capital
            equity_curve_df['short_capital'] = short_equity_df['capital'] if not short_equity_df.empty else self.initial_capital
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

def run_backtest(df: pd.DataFrame, 
                atr_coeff: float = 1.0, 
                initial_capital: float = 10000, 
                debug: bool = False, 
                kelly_debug: bool = False,
                atr_period_for_stoploss: int = 28,
                stop_loss_adjust: float = 1.0, 
                stop_loss_adjust_sma_period: int = 200,
                kelly_multiplier: float = 3.0,  
                overnight_position_size: float = 0.5,
                lookback_period_for_position_size: int = 50,
                stop_loss_scale_coeff: float = 25.0,
                use_performance_scaling: bool = True,
                use_simpler_position_sizing: bool = True,
                return_equity_curves: bool = False) -> BacktestResults:
    """
    Run backtest with support for both long and short positions.
    """
    if debug:
        print(f"[DEBUG] Starting run_backtest with atr_coeff={atr_coeff}, initial_capital={initial_capital}")

    all_trades = []
    all_metrics = []
    processed_dfs = []
    equity_curves = {}
    
    # Track performance by position type
    position_type_performance = {
        'long': {'trades': [], 'metrics': []},
        'short': {'trades': [], 'metrics': []}
    }

    symbols = df['symbol'].unique()
    for symbol in symbols:
        if debug:
            print(f"\n[DEBUG] Processing symbol: {symbol}")

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
            use_simpler_position_sizing=use_simpler_position_sizing
        )

        symbol_df = df[df['symbol'] == symbol].copy()
        symbol_df.sort_index(inplace=True)

        # Check for duplicate timestamps
        duplicates_input = symbol_df.index.duplicated().sum()
        assert duplicates_input == 0, f"Symbol {symbol} has {duplicates_input} duplicate timestamps in input data."

        try:
            trades_df, metrics, processed_df, equity_curve_df = backtester.backtest_symbol(
                symbol_df,
                symbol,
                atr_coeff,
                kelly_debug=kelly_debug
            )
            
            if not trades_df.empty:
                # Split trades by position type
                long_trades = trades_df[~trades_df['is_short']]
                short_trades = trades_df[trades_df['is_short']]
                
                if not long_trades.empty:
                    position_type_performance['long']['trades'].append(long_trades)
                if not short_trades.empty:
                    position_type_performance['short']['trades'].append(short_trades)
                
                all_trades.append(trades_df)

            all_metrics.append(metrics)
            processed_dfs.append(processed_df)

            if not equity_curve_df.empty:
                equity_curves[symbol] = equity_curve_df

        except ValueError as e:
            print(f"Skipping symbol {symbol} due to error: {e}")
            continue
        except Exception as e:
            print(f"An unexpected error occurred for symbol {symbol}: {e}")
            continue

    # Combine results
    combined_trades = pd.concat(all_trades) if all_trades else pd.DataFrame()
    metrics_df = pd.DataFrame(all_metrics) if all_metrics else pd.DataFrame()
    processed_df = pd.concat(processed_dfs) if processed_dfs else pd.DataFrame()

    # Calculate overall performance metrics
    overall_metrics = {
        'total_trades': len(combined_trades),
        'long_trades': len(combined_trades[~combined_trades['is_short']]) if not combined_trades.empty else 0,
        'short_trades': len(combined_trades[combined_trades['is_short']]) if not combined_trades.empty else 0,
        'average_total_return_percentage': metrics_df['total_returns_percentage'].mean() if not metrics_df.empty else 0,
        'average_long_return': metrics_df['long_returns_percentage'].mean() if 'long_returns_percentage' in metrics_df else 0,
        'average_short_return': metrics_df['short_returns_percentage'].mean() if 'short_returns_percentage' in metrics_df else 0,
        'long_win_rate': metrics_df['long_win_rate'].mean() if 'long_win_rate' in metrics_df else 0,
        'short_win_rate': metrics_df['short_win_rate'].mean() if 'short_win_rate' in metrics_df else 0,
        'mean_sharpe_ratio': metrics_df['sharpe_ratio'].mean() if 'sharpe_ratio' in metrics_df else 0,
    }

    # Calculate position-specific metrics
    for position_type in ['long', 'short']:
        position_trades = pd.concat(position_type_performance[position_type]['trades']) \
                         if position_type_performance[position_type]['trades'] else pd.DataFrame()
        
        if not position_trades.empty:
            avg_hold_time = position_trades['hold_time_hours'].mean()
            avg_position_size = position_trades['position_size'].mean() / initial_capital * 100
            win_rate = (position_trades['return'] > 0).mean() * 100
            
            overall_metrics.update({
                f'{position_type}_avg_hold_time': avg_hold_time,
                f'{position_type}_avg_position_size': avg_position_size,
                f'{position_type}_win_rate': win_rate,
                f'{position_type}_total_return': position_trades['return_percentage'].sum(),
                f'{position_type}_avg_return_per_trade': position_trades['return_percentage'].mean(),
                f'{position_type}_return_std': position_trades['return_percentage'].std(),
                f'{position_type}_max_drawdown': calculate_max_drawdown(position_trades['return_percentage']),
                f'{position_type}_profit_factor': calculate_profit_factor(position_trades['return'])
            })

    # Generate equity curves if requested
    equity_curve_plots = []
    if return_equity_curves:
        for symbol, equity_df in equity_curves.items():
            symbol_df = df[df['symbol'] == symbol]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot overall equity curve
            ax.plot(equity_df['timestamp'], equity_df['capital'], 
                label='Combined', color='blue', alpha=0.7)
            
            # Plot position-specific curves
            ax.plot(equity_df['timestamp'], equity_df['long_capital'],
                label='Long Only', color='green', alpha=0.5)
            ax.plot(equity_df['timestamp'], equity_df['short_capital'],
                label='Short Only', color='red', alpha=0.5)
            
            title = f'Equity Curves - {symbol}\n'
            if 'long_trades_count' in metrics and 'short_trades_count' in metrics:
                title += f'Long Trades: {metrics["long_trades_count"]} | Short Trades: {metrics["short_trades_count"]}'
            ax.set_title(title)
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
        equity_curve_plots=equity_curve_plots if return_equity_curves else None
    )

def calculate_max_drawdown(returns):
    """Calculate maximum drawdown from a series of returns."""
    cumulative = (1 + returns/100).cumprod()
    rolling_max = cumulative.expanding(min_periods=1).max()
    drawdowns = (cumulative - rolling_max) / rolling_max * 100
    return drawdowns.min()

def calculate_profit_factor(returns):
    """Calculate profit factor (gross profits / gross losses)."""
    profits = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    return profits / losses if losses != 0 else float('inf')


