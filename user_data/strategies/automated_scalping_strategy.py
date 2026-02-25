# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Optional, Union

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,  # @informative decorator
    # Hyperopt Parameters
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    # timeframe helpers
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    # Strategy helper functions
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
from technical import qtpylib


class AutomatedScalpingStrategy(IStrategy):
    """
    Automated Scalping Strategy v4.2 - Python/Freqtrade Version
    Converted from JavaScript Gunbot strategy
    High-frequency cascading filter system with multi-timeframe trend filtering
    """
    
    # Strategy interface version
    INTERFACE_VERSION = 3
    
    # Can this strategy go short?
    can_short: bool = True
    
    # Minimal ROI designed for the strategy
    minimal_roi = {
        "60": 0.01,
        "30": 0.02,
        "0": 0.04,
    }
    
    # Optimal stoploss designed for the strategy
    stoploss = -0.10
    
    # Trailing stoploss
    trailing_stop = False
    
    # Optimal timeframe for the strategy
    timeframe = "5m"
    
    # Run "populate_indicators()" only for new candle
    process_only_new_candles = True
    
    # These values can be overridden in the config
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    
    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200
    
    # Optional order type mapping
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }
    
    # Optional order time in force
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}
    
    plot_config = {
        "main_plot": {
            "ema_8": {"color": "blue"},
            "ema_13": {"color": "orange"},
            "ema_21": {"color": "red"},
            "vwap": {"color": "purple"},
        },
        "subplots": {
            "WaveTrend": {
                "wt1": {"color": "green"},
                "wt2": {"color": "red"},
                "wt_overbought": {"color": "gray", "overlay": True},
                "wt_oversold": {"color": "gray", "overlay": True},
            },
            "RSI": {
                "rsi": {"color": "yellow"},
                "rsi_overbought": {"color": "red", "overlay": True},
                "rsi_oversold": {"color": "green", "overlay": True},
            },
            "ADX": {
                "adx": {"color": "white"},
                "adx_threshold": {"color": "gray", "overlay": True},
            },
            "Volume": {
                "volume_ratio": {"color": "cyan"},
            },
        },
    }
    
    # Strategy parameters (from JavaScript CONFIG)
    # HTF Trend Filter
    htf_adx_period = IntParameter(10, 20, default=14, space="buy", optimize=True)
    htf_adx_threshold = IntParameter(20, 30, default=25, space="buy", optimize=True)
    htf_range_adx_offset = IntParameter(3, 8, default=5, space="buy", optimize=True)
    
    # LTF Entry Engine - WaveTrend
    wt_channel_length = IntParameter(8, 15, default=10, space="buy", optimize=True)
    wt_avg_length = IntParameter(18, 25, default=21, space="buy", optimize=True)
    wt_overbought = IntParameter(50, 70, default=60, space="sell", optimize=True)
    wt_oversold = IntParameter(-70, -50, default=-60, space="sell", optimize=True)
    wt_entry_long_max = IntParameter(10, 20, default=15, space="buy", optimize=True)
    wt_entry_long_min = IntParameter(-30, -20, default=-25, space="buy", optimize=True)
    wt_entry_short_max = IntParameter(20, 30, default=25, space="buy", optimize=True)
    wt_entry_short_min = IntParameter(-20, -10, default=-15, space="buy", optimize=True)
    
    # RSI
    rsi_period = IntParameter(10, 20, default=14, space="buy", optimize=True)
    rsi_threshold = IntParameter(40, 50, default=45, space="buy", optimize=True)
    rsi_long_max = IntParameter(60, 70, default=65, space="buy", optimize=True)
    rsi_short_min = IntParameter(30, 40, default=35, space="buy", optimize=True)
    
    # EMA Stack
    ema_periods = [8, 13, 21]
    
    # VWAP
    vwap_acceptance_candles = IntParameter(1, 5, default=1, space="buy", optimize=True)
    
    # Volume Veto
    volume_veto_period = IntParameter(15, 25, default=20, space="buy", optimize=True)
    volume_veto_multiplier = DecimalParameter(2.0, 3.0, default=2.5, space="buy", optimize=True)
    
    # Risk Management
    leverage = IntParameter(5, 20, default=10, space="buy", optimize=True)
    risk_per_trade_percent = DecimalParameter(0.3, 1.0, default=0.5, space="buy", optimize=True)
    min_rr_ratio = DecimalParameter(1.0, 2.0, default=1.5, space="buy", optimize=True)
    slippage_cost_percent = DecimalParameter(0.03, 0.08, default=0.05, space="buy", optimize=True)
    commission_percent = DecimalParameter(0.03, 0.06, default=0.04, space="buy", optimize=True)
    weekly_drawdown_limit = IntParameter(8, 15, default=10, space="buy", optimize=True)
    total_drawdown_limit = IntParameter(12, 18, default=15, space="buy", optimize=True)
    
    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        """
        return []
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame
        """
        
        # EMA Stack
        for period in self.ema_periods:
            dataframe[f'ema_{period}'] = ta.EMA(dataframe, timeperiod=period)
        
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)
        
        # ADX for HTF trend (using current timeframe for now as proxy)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=self.htf_adx_period.value)
        
        # Volume analysis
        dataframe['volume_sma'] = ta.SMA(dataframe['volume'], timeperiod=self.volume_veto_period.value)
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_sma']
        
        # Bollinger Bands for range detection
        dataframe['bb_upper'], dataframe['bb_middle'], dataframe['bb_lower'] = ta.BBANDS(
            dataframe['close'], timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0
        )
        dataframe['bb_percent'] = (dataframe['close'] - dataframe['bb_lower']) / (dataframe['bb_upper'] - dataframe['bb_lower'])
        
        # VWAP
        dataframe['vwap'] = qtpylib.rolling_vwap(dataframe, window=14)
        
        # WaveTrend Implementation - LOOSENED
        ap = (dataframe['close'] + dataframe['high'] + dataframe['low']) / 3
        dataframe['esa'] = ta.EMA(ap, timeperiod=self.wt_channel_length.value)
        
        # d = ema(abs(ap - esa), channel_length)
        dataframe['ap_esa_diff'] = abs(ap - dataframe['esa'])
        dataframe['d'] = ta.EMA(dataframe['ap_esa_diff'], timeperiod=self.wt_channel_length.value)
        
        # ci = (ap - esa) / (0.015 * d)
        dataframe['ci'] = (ap - dataframe['esa']) / (0.015 * dataframe['d'])
        
        # wt1 = ema(ci, avg_length)
        dataframe['wt1'] = ta.EMA(dataframe['ci'], timeperiod=self.wt_avg_length.value)
        
        # wt2 = sma(wt1, 4)
        dataframe['wt2'] = ta.SMA(dataframe['wt1'], timeperiod=4)
        
        # WaveTrend cross and zones
        dataframe['wt_cross_up'] = (dataframe['wt1'].shift(1) <= dataframe['wt2'].shift(1)) & (dataframe['wt1'] > dataframe['wt2'])
        dataframe['wt_cross_down'] = (dataframe['wt1'].shift(1) >= dataframe['wt2'].shift(1)) & (dataframe['wt1'] < dataframe['wt2'])
        dataframe['wt_oversold'] = dataframe['wt1'] < self.wt_oversold.value
        dataframe['wt_overbought'] = dataframe['wt1'] > self.wt_overbought.value
        dataframe['wt_near_zero_long'] = (dataframe['wt1'] < self.wt_entry_long_max.value) & (dataframe['wt1'] > self.wt_entry_long_min.value)
        dataframe['wt_near_zero_short'] = (dataframe['wt1'] < self.wt_entry_short_max.value) & (dataframe['wt1'] > self.wt_entry_short_min.value)
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        """
        
        # Conditions setup
        conditions_long = []
        conditions_short = []
        
        # 1. HTF Trend Filter (ADX-based)
        adx_threshold = self.htf_adx_threshold.value
        range_adx_offset = self.htf_range_adx_offset.value
        
        # Trend mode: ADX > threshold
        trend_mode = dataframe['adx'] > adx_threshold
        # Range mode: ADX < (threshold - offset)
        range_mode = dataframe['adx'] < (adx_threshold - range_adx_offset)
        
        # 2. EMA Stack Alignment
        ema_stack_aligned_long = (
            (dataframe['ema_8'] > dataframe['ema_13']) & 
            (dataframe['ema_13'] > dataframe['ema_21'])
        )
        ema_stack_aligned_short = (
            (dataframe['ema_8'] < dataframe['ema_13']) & 
            (dataframe['ema_13'] < dataframe['ema_21'])
        )
        
        # 3. RSI Logic
        rsi_overbought = dataframe['rsi'] > self.rsi_long_max.value
        rsi_oversold = dataframe['rsi'] < self.rsi_short_min.value
        rsi_near_threshold_long = dataframe['rsi'] < self.rsi_threshold.value
        rsi_near_threshold_short = dataframe['rsi'] > (100 - self.rsi_threshold.value)
        
        # 4. Volume Veto
        volume_veto = dataframe['volume_ratio'] < self.volume_veto_multiplier.value
        
        # LONG ENTRY CONDITIONS
        if trend_mode.iloc[-1]:  # Trend mode
            conditions_long.append(
                ema_stack_aligned_long &  # Trend alignment
                dataframe['wt_near_zero_long'] &       # WaveTrend in entry zone
                rsi_near_threshold_long & # RSI not overbought
                ~rsi_overbought &         # RSI below overbought
                volume_veto &              # Volume not excessive
                dataframe['wt_cross_up']               # WaveTrend cross up
            )
        else:  # Range mode
            conditions_long.append(
                (dataframe['bb_percent'] < 0.2) &  # Near lower Bollinger Band
                rsi_near_threshold_long &
                ~rsi_overbought &
                volume_veto &
                dataframe['wt_cross_up']
            )
        
        # SHORT ENTRY CONDITIONS
        if trend_mode.iloc[-1]:  # Trend mode
            conditions_short.append(
                ema_stack_aligned_short &  # Trend alignment
                dataframe['wt_near_zero_short'] &       # WaveTrend in entry zone
                rsi_near_threshold_short & # RSI not oversold
                ~rsi_oversold &            # RSI above oversold
                volume_veto &              # Volume not excessive
                dataframe['wt_cross_down']              # WaveTrend cross down
            )
        else:  # Range mode
            conditions_short.append(
                (dataframe['bb_percent'] > 0.8) &  # Near upper Bollinger Band
                rsi_near_threshold_short &
                ~rsi_oversold &
                volume_veto &
                dataframe['wt_cross_down']
            )
        
        # Combine conditions
        if conditions_long:
            dataframe.loc[conditions_long[0], 'enter_long'] = 1
        
        if conditions_short:
            dataframe.loc[conditions_short[0], 'enter_short'] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        """
        
        # Exit conditions
        conditions_exit_long = []
        conditions_exit_short = []
        
        # RSI exits
        rsi_overbought = dataframe['rsi'] > self.rsi_long_max.value
        rsi_oversold = dataframe['rsi'] < self.rsi_short_min.value
        
        # WaveTrend exits
        wt_oversold = dataframe['wt1'] < self.wt_oversold.value
        wt_overbought = dataframe['wt1'] > self.wt_overbought.value
        
        # EMA stack reversal
        ema_stack_reversal_long = dataframe['ema_8'] < dataframe['ema_13']
        ema_stack_reversal_short = dataframe['ema_8'] > dataframe['ema_13']
        
        # LONG EXIT CONDITIONS
        conditions_exit_long.append(
            (rsi_overbought | ema_stack_reversal_long | wt_overbought)
        )
        
        # SHORT EXIT CONDITIONS
        conditions_exit_short.append(
            (rsi_oversold | ema_stack_reversal_short | wt_oversold)
        )
        
        # Combine conditions
        if conditions_exit_long:
            dataframe.loc[conditions_exit_long[0], 'exit_long'] = 1
        
        if conditions_exit_short:
            dataframe.loc[conditions_exit_short[0], 'exit_short'] = 1
        
        return dataframe