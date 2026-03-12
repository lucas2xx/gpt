"""
Strategy Backtester - Level-Based Trend Trading
================================================
Tests Lucas's strategy using:
1. MTF Trends script (converted from Pine Script)
2. Hourly Levels script (converted from Pine Script)
3. Pullback + breakout entries with 3R targets

Author: Built for Lucas
Date: March 2026
"""

import numpy as np
import pandas as pd
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
import requests
import time as time_module
from typing import List, Dict, Tuple, Optional

# ============================================
# CONFIGURATION
# ============================================

class Config:
    # Polygon API
    POLYGON_API_KEY = "AQEre657fEzh_Sq_sZZKH5MgIvS83YGQ"

    # Instrument
    TICKER = "QQQ"

    # Timeframes
    ENTRY_TIMEFRAME = "5"      # 5-minute bars for entries
    LEVELS_TIMEFRAME = "60"    # 1-hour bars for levels
    TREND_TIMEFRAME = "5"      # 5-minute trend

    # Strategy Parameters
    RISK_PER_TRADE = 300       # $ risk per trade
    REWARD_MULTIPLE = 3        # 3R target
    LEVEL_TOLERANCE = 0.02     # Within 2 cents of level

    # Test Period
    START_DATE = "2021-03-01"  # 5 years back
    END_DATE = "2026-03-01"

    # Account
    STARTING_CAPITAL = 50000

    # Trade Direction
    TRADE_LONGS = True
    TRADE_SHORTS = True


# ============================================
# RTH FILTER
# ============================================

ET = ZoneInfo("America/New_York")


def filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only bars within Regular Trading Hours (9:30 - 15:30 ET)."""
    # Index must be UTC-aware (set during fetch)
    df_et_idx = df.index.tz_convert(ET)
    mask = (
        (df_et_idx.time >= time(9, 30)) &
        (df_et_idx.time <= time(15, 30))
    )
    return df[mask]


# ============================================
# POLYGON API DATA FETCHER
# ============================================

class PolygonDataFetcher:
    """Fetch historical data from Polygon.io API."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"

    def fetch_aggregates(self,
                         ticker: str,
                         timespan: str,
                         multiplier: int,
                         from_date: str,
                         to_date: str) -> pd.DataFrame:
        """
        Fetch aggregate bars from Polygon.

        Parameters:
        -----------
        ticker : str
            Stock symbol (e.g., 'QQQ')
        timespan : str
            'minute', 'hour', 'day'
        multiplier : int
            Number of timespans (e.g., 5 for 5-minute)
        from_date : str
            Start date 'YYYY-MM-DD'
        to_date : str
            End date 'YYYY-MM-DD'

        Returns:
        --------
        pd.DataFrame with UTC-aware DatetimeIndex and columns:
        open, high, low, close, volume
        """

        url = (
            f"{self.base_url}/v2/aggs/ticker/{ticker}/range"
            f"/{multiplier}/{timespan}/{from_date}/{to_date}"
        )

        params = {
            'apiKey': self.api_key,
            'adjusted': 'true',
            'sort': 'asc',
            'limit': 50000
        }

        all_data = []

        print(f"Fetching {ticker} {multiplier}{timespan} data from {from_date} to {to_date}...")

        while True:
            response = requests.get(url, params=params)

            if response.status_code != 200:
                print(f"Error: {response.status_code}")
                print(response.text)
                break

            data = response.json()

            if 'results' not in data or len(data['results']) == 0:
                break

            all_data.extend(data['results'])

            if 'next_url' not in data:
                break

            url = data['next_url']
            params = {'apiKey': self.api_key}

            time_module.sleep(0.1)  # Rate limiting

        if len(all_data) == 0:
            print("No data received from Polygon API")
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        # Use utc=True so the index is timezone-aware — required for tz_convert later
        df['timestamp'] = pd.to_datetime(df['t'], unit='ms', utc=True)
        df = df.rename(columns={
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume'
        })

        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df = df.set_index('timestamp')
        df = df.sort_index()

        print(f"  Fetched {len(df)} bars (pre-RTH filter)")

        return df


# ============================================
# MTF TRENDS (Converted from Pine Script)
# ============================================

class MTFTrends:
    """
    Multi-timeframe trend detector.
    Converted from Lucas's Pine Script.

    Returns:
        1  = Uptrend
        -1 = Downtrend
        0  = No trend / NaN (during warm-up period)
    """

    def __init__(self,
                 ma_type='EMA',
                 trend_period=20,
                 ma_period=20,
                 channel_rate=1.0,
                 use_linreg=True,
                 linreg_period=5):
        self.ma_type = ma_type
        self.trend_period = trend_period
        self.ma_period = ma_period
        self.channel_rate = channel_rate
        self.use_linreg = use_linreg
        self.linreg_period = linreg_period

    def calculate_ma(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate moving average based on type."""
        if self.ma_type == 'EMA':
            return series.ewm(span=period, adjust=False).mean()
        elif self.ma_type == 'SMA':
            return series.rolling(period).mean()
        elif self.ma_type == 'WMA':
            weights = np.arange(1, period + 1)
            return series.rolling(period).apply(
                lambda x: np.dot(x, weights) / weights.sum(), raw=True
            )
        elif self.ma_type == 'RMA':
            return series.ewm(alpha=1 / period, adjust=False).mean()
        else:
            return series.rolling(period).mean()

    def linear_regression(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate linear regression value (last point of fitted line)."""
        def linreg(y):
            if len(y) < period:
                return np.nan
            x = np.arange(len(y))
            slope, intercept = np.polyfit(x, y, 1)
            return intercept + slope * (len(y) - 1)

        return series.rolling(period).apply(linreg, raw=False)

    def get_trend(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate trend for a dataframe.

        Returns series with values: 1 (up), -1 (down), 0 (neutral/NaN-filled)
        """
        close = df['close']

        # Calculate price range over 280 periods
        pricerange = close.rolling(280).max() - close.rolling(280).min()
        chan = pricerange * (self.channel_rate / 100)

        # Calculate MA then optionally smooth with linear regression
        ma_src = self.calculate_ma(close, self.ma_period)
        ma = self.linear_regression(ma_src, self.linreg_period) if self.use_linreg else ma_src

        # Highest and lowest MA over trend period
        hh = ma.rolling(self.trend_period).max()
        ll = ma.rolling(self.trend_period).min()
        diff = (hh - ll).abs()

        trend = pd.Series(0, index=df.index, dtype=int)
        trend[(diff > chan) & (ma > ll + chan)] = 1
        trend[(diff > chan) & (ma < hh - chan)] = -1

        # Mask warm-up bars where inputs are NaN as 0 (no signal)
        nan_mask = chan.isna() | ma.isna() | hh.isna() | ll.isna()
        trend[nan_mask] = 0

        return trend


# ============================================
# HOURLY LEVELS (Converted from Pine Script)
# ============================================

class HourlyLevels:
    """
    Calculates support/resistance levels from hourly data.
    Converted from Lucas's Pine Script.

    Logic:
    - Level created at current candle's open when candle color changes
      from the previous candle (green→red or red→green).
    - Levels removed when a candle's BODY crosses them (wick-only
      crosses are treated as tests/rejections; level stays active).
    """

    def calculate_levels(self, df_hourly: pd.DataFrame) -> Dict:
        """
        Calculate levels from hourly data.

        Returns:
        --------
        Dict mapping timestamp -> list of active levels at that time
        """
        active_levels: List[float] = []
        levels_by_time: Dict = {}

        prev_is_green: Optional[bool] = None

        for idx, row in df_hourly.iterrows():
            curr_open = row['open']
            curr_close = row['close']

            is_green = curr_close >= curr_open

            # Remove levels whose body crosses an active level.
            # Body crossing = candle open/close bracket the level on opposite sides.
            body_lo = min(curr_open, curr_close)
            body_hi = max(curr_open, curr_close)
            active_levels = [p for p in active_levels if not (body_lo < p < body_hi)]

            # On candle color change, add current open as a new support/resistance level
            if prev_is_green is not None and prev_is_green != is_green:
                active_levels.append(curr_open)

            levels_by_time[idx] = list(active_levels)

            prev_is_green = is_green

        return levels_by_time

    def get_closest_levels(self,
                           levels: List[float],
                           price: float,
                           direction: str,
                           n: int = 10) -> List[float]:
        """
        Get n closest levels above (direction='above') or below (direction='below') price.
        """
        if direction == 'above':
            above = sorted(lv for lv in levels if lv > price)
            return above[:n]
        else:
            below = sorted((lv for lv in levels if lv < price), reverse=True)
            return below[:n]


# ============================================
# STRATEGY LOGIC
# ============================================

class StrategyBacktester:
    """Main backtesting engine for Lucas's strategy."""

    def __init__(self, config: Config):
        self.config = config
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []

    def is_signal_bar(self,
                      current_bar: pd.Series,
                      prev_bar: pd.Series,
                      trend: int) -> bool:
        """
        Check if current_bar is a valid signal bar relative to prev_bar.

        Uptrend  (trend=1):  current wicks below prev low but closes within prev range
        Downtrend (trend=-1): current wicks above prev high but closes within prev range
        """
        if trend == 1:
            breaks_low = current_bar['low'] < prev_bar['low']
            closes_within = prev_bar['low'] <= current_bar['close'] <= prev_bar['high']
            return breaks_low and closes_within

        elif trend == -1:
            breaks_high = current_bar['high'] > prev_bar['high']
            closes_within = prev_bar['low'] <= current_bar['close'] <= prev_bar['high']
            return breaks_high and closes_within

        return False

    def touches_level(self, bar: pd.Series, level: float, tolerance: float) -> bool:
        """Check if bar's range (low–high) touches a level within tolerance."""
        return (bar['low'] - tolerance) <= level <= (bar['high'] + tolerance)

    def run_backtest(self,
                     df_5m: pd.DataFrame,
                     df_1h: pd.DataFrame) -> Dict:
        """
        Run the full backtest.

        Parameters:
        -----------
        df_5m : 5-minute RTH bars (UTC-aware index)
        df_1h : 1-hour RTH bars (UTC-aware index)
        """

        print("\n" + "=" * 60)
        print("RUNNING BACKTEST")
        print("=" * 60)

        # --- 1. Trends ---
        print("\n1. Calculating trends...")
        trend_calc = MTFTrends()
        df_5m = df_5m.copy()
        df_5m['trend'] = trend_calc.get_trend(df_5m)
        print("   Trends calculated")

        # --- 2. Levels ---
        print("\n2. Calculating hourly levels...")
        levels_calc = HourlyLevels()
        levels_by_time = levels_calc.calculate_levels(df_1h)
        print(f"   Levels calculated ({len(levels_by_time)} hourly snapshots)")

        # --- 3. Align levels to 5m bars using merge_asof (O((n+m) log n)) ---
        print("\n3. Aligning levels to 5m bars...")
        levels_series = pd.Series(levels_by_time, name='active_levels')
        levels_df = levels_series.to_frame().sort_index()

        df_5m = pd.merge_asof(
            df_5m.sort_index(),
            levels_df,
            left_index=True,
            right_index=True,
            direction='backward'
        )
        # Drop rows where no prior 1h snapshot exists (very start of data)
        df_5m = df_5m.dropna(subset=['active_levels'])
        print(f"   Aligned. {len(df_5m)} bars after alignment.")

        # --- 4. Main loop ---
        print("\n4. Scanning for setups...")

        capital = self.config.STARTING_CAPITAL
        equity = capital
        peak_equity = capital
        max_drawdown = 0

        bars_with_trend = 0
        bars_with_levels = 0
        bars_touched_level = 0
        bars_signal_bar = 0
        bars_entry_triggered = 0

        in_trade = False
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        trade_direction = 0
        entry_time = None

        df_arr = df_5m.reset_index()  # use positional indexing for speed

        for i in range(2, len(df_arr)):
            current_bar = df_arr.iloc[i]
            prev_bar = df_arr.iloc[i - 1]
            prev_prev_bar = df_arr.iloc[i - 2]

            current_trend = int(current_bar['trend'])
            active_levels = current_bar['active_levels']
            if not isinstance(active_levels, list):
                active_levels = []

            if current_trend != 0:
                bars_with_trend += 1
            if len(active_levels) > 0:
                bars_with_levels += 1

            # --- Manage open trade ---
            if in_trade:
                if trade_direction == 1:  # Long
                    if current_bar['low'] <= stop_loss:
                        pnl = -self.config.RISK_PER_TRADE
                        equity += pnl
                        self.trades.append({
                            'entry_time': entry_time,
                            'exit_time': current_bar['timestamp'],
                            'direction': 'LONG',
                            'entry_price': entry_price,
                            'exit_price': stop_loss,
                            'stop': stop_loss,
                            'target': take_profit,
                            'pnl': pnl,
                            'outcome': 'LOSS'
                        })
                        in_trade = False

                    elif current_bar['high'] >= take_profit:
                        pnl = self.config.RISK_PER_TRADE * self.config.REWARD_MULTIPLE
                        equity += pnl
                        self.trades.append({
                            'entry_time': entry_time,
                            'exit_time': current_bar['timestamp'],
                            'direction': 'LONG',
                            'entry_price': entry_price,
                            'exit_price': take_profit,
                            'stop': stop_loss,
                            'target': take_profit,
                            'pnl': pnl,
                            'outcome': 'WIN'
                        })
                        in_trade = False

                elif trade_direction == -1:  # Short
                    if current_bar['high'] >= stop_loss:
                        pnl = -self.config.RISK_PER_TRADE
                        equity += pnl
                        self.trades.append({
                            'entry_time': entry_time,
                            'exit_time': current_bar['timestamp'],
                            'direction': 'SHORT',
                            'entry_price': entry_price,
                            'exit_price': stop_loss,
                            'stop': stop_loss,
                            'target': take_profit,
                            'pnl': pnl,
                            'outcome': 'LOSS'
                        })
                        in_trade = False

                    elif current_bar['low'] <= take_profit:
                        pnl = self.config.RISK_PER_TRADE * self.config.REWARD_MULTIPLE
                        equity += pnl
                        self.trades.append({
                            'entry_time': entry_time,
                            'exit_time': current_bar['timestamp'],
                            'direction': 'SHORT',
                            'entry_price': entry_price,
                            'exit_price': take_profit,
                            'stop': stop_loss,
                            'target': take_profit,
                            'pnl': pnl,
                            'outcome': 'WIN'
                        })
                        in_trade = False

            # --- Look for new setups ---
            if not in_trade:
                if current_trend == 0 or len(active_levels) == 0:
                    pass
                else:
                    # Check if prev_bar touched any level
                    touched_level = None
                    for level in active_levels:
                        if self.touches_level(prev_bar, level, self.config.LEVEL_TOLERANCE):
                            touched_level = level
                            bars_touched_level += 1
                            break

                    if touched_level is not None:
                        # is_signal_bar(candidate, reference, trend)
                        # prev_bar is the signal candidate; prev_prev_bar is the reference
                        if self.is_signal_bar(prev_bar, prev_prev_bar, current_trend):
                            bars_signal_bar += 1

                            if current_trend == 1 and self.config.TRADE_LONGS:
                                if current_bar['high'] > prev_bar['high']:
                                    bars_entry_triggered += 1
                                    entry_price = prev_bar['high']
                                    stop_loss = prev_bar['low']
                                    risk_points = entry_price - stop_loss

                                    if risk_points <= 0:
                                        pass  # Skip degenerate setup
                                    else:
                                        target_3r = entry_price + risk_points * self.config.REWARD_MULTIPLE
                                        levels_above = levels_calc.get_closest_levels(
                                            active_levels, entry_price, 'above', n=1
                                        )
                                        target_level = levels_above[0] if levels_above else target_3r
                                        take_profit = min(target_3r, target_level)

                                        in_trade = True
                                        trade_direction = 1
                                        entry_time = current_bar['timestamp']

                            elif current_trend == -1 and self.config.TRADE_SHORTS:
                                if current_bar['low'] < prev_bar['low']:
                                    bars_entry_triggered += 1
                                    entry_price = prev_bar['low']
                                    stop_loss = prev_bar['high']
                                    risk_points = stop_loss - entry_price

                                    if risk_points <= 0:
                                        pass  # Skip degenerate setup
                                    else:
                                        target_3r = entry_price - risk_points * self.config.REWARD_MULTIPLE
                                        levels_below = levels_calc.get_closest_levels(
                                            active_levels, entry_price, 'below', n=1
                                        )
                                        target_level = levels_below[0] if levels_below else target_3r
                                        take_profit = max(target_3r, target_level)

                                        in_trade = True
                                        trade_direction = -1
                                        entry_time = current_bar['timestamp']

            # Track equity curve
            peak_equity = max(peak_equity, equity)
            drawdown = peak_equity - equity
            max_drawdown = max(max_drawdown, drawdown)
            self.equity_curve.append({
                'timestamp': current_bar['timestamp'],
                'equity': equity,
                'peak': peak_equity,
                'drawdown': drawdown
            })

        # Diagnostics
        print(f"\n   DIAGNOSTICS:")
        print(f"   Total 5m bars scanned:      {len(df_5m)}")
        print(f"   Bars with trend (!=0):      {bars_with_trend}")
        print(f"   Bars with active levels:    {bars_with_levels}")
        print(f"   Bars that touched level:    {bars_touched_level}")
        print(f"   Valid signal bars:          {bars_signal_bar}")
        print(f"   Entry triggers:             {bars_entry_triggered}")
        print(f"   Actual trades:              {len(self.trades)}")

        return self.calculate_statistics()

    def calculate_statistics(self) -> Dict:
        """Calculate backtest performance statistics."""

        if len(self.trades) == 0:
            return {'total_trades': 0, 'message': 'No trades found'}

        trades_df = pd.DataFrame(self.trades)
        wins = trades_df[trades_df['outcome'] == 'WIN']
        losses = trades_df[trades_df['outcome'] == 'LOSS']

        total_trades = len(trades_df)
        num_wins = len(wins)
        num_losses = len(losses)
        win_rate = (num_wins / total_trades * 100) if total_trades > 0 else 0
        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss = abs(losses['pnl'].mean()) if len(losses) > 0 else 0
        total_pnl = trades_df['pnl'].sum()

        equity_df = pd.DataFrame(self.equity_curve)
        max_dd = equity_df['drawdown'].max() if len(equity_df) > 0 else 0

        gross_loss = losses['pnl'].sum() if len(losses) > 0 else 0
        profit_factor = (
            wins['pnl'].sum() / abs(gross_loss)
            if gross_loss != 0 else 0
        )

        return {
            'total_trades': total_trades,
            'wins': num_wins,
            'losses': num_losses,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_pnl': total_pnl,
            'max_drawdown': max_dd,
            'profit_factor': profit_factor,
            'trades_df': trades_df,
            'equity_df': equity_df,
        }


# ============================================
# MAIN EXECUTION
# ============================================

def main():
    config = Config()
    fetcher = PolygonDataFetcher(config.POLYGON_API_KEY)

    # Fetch data
    print("\nFetching 5-minute data...")
    df_5m_raw = fetcher.fetch_aggregates(
        ticker=config.TICKER,
        timespan='minute',
        multiplier=5,
        from_date=config.START_DATE,
        to_date=config.END_DATE
    )
    if df_5m_raw.empty:
        print("ERROR: No 5-minute data received")
        return

    print("\nFetching 1-hour data...")
    df_1h_raw = fetcher.fetch_aggregates(
        ticker=config.TICKER,
        timespan='hour',
        multiplier=1,
        from_date=config.START_DATE,
        to_date=config.END_DATE
    )
    if df_1h_raw.empty:
        print("ERROR: No 1-hour data received")
        return

    # Filter to RTH (9:30 - 15:30 ET) — removes pre-market/after-hours noise
    print("\nFiltering to RTH (9:30–15:30 ET)...")
    df_5m = filter_rth(df_5m_raw)
    df_1h = filter_rth(df_1h_raw)
    print(f"  5m bars after RTH filter: {len(df_5m)}")
    print(f"  1h bars after RTH filter: {len(df_1h)}")

    # Run backtest
    backtester = StrategyBacktester(config)
    results = backtester.run_backtest(df_5m, df_1h)

    # Print results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    if results['total_trades'] == 0:
        print("\n  NO TRADES FOUND")
        print("\nPossible reasons:")
        print("  1. Levels not being calculated correctly")
        print("  2. Trend filter too strict")
        print("  3. Signal bar criteria not being met")
        print("  4. No pullbacks to levels occurring")
    else:
        print(f"\nTotal Trades:   {results['total_trades']}")
        print(f"Wins:           {results['wins']}")
        print(f"Losses:         {results['losses']}")
        print(f"Win Rate:       {results['win_rate']:.2f}%")
        print(f"Avg Win:        ${results['avg_win']:.2f}")
        print(f"Avg Loss:       ${results['avg_loss']:.2f}")
        print(f"Total P&L:      ${results['total_pnl']:.2f}")
        print(f"Max Drawdown:   ${results['max_drawdown']:.2f}")
        print(f"Profit Factor:  {results['profit_factor']:.2f}")

    # Save output files
    if 'trades_df' in results:
        results['trades_df'].to_csv('backtest_trades.csv', index=False)
        print("\n  Trades saved to: backtest_trades.csv")

    if 'equity_df' in results:
        results['equity_df'].to_csv('equity_curve.csv', index=False)
        print("  Equity curve saved to: equity_curve.csv")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
