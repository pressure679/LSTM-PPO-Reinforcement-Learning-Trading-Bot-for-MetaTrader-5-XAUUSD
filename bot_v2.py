"""
bot_v2.py -- algorithmic XAUUSD trading bot: three purely rule-based
entry strategies, one 1m execution loop, no learned model of any kind.

A second, independent strategy set alongside bot.py's ICT/SMC bot, built
the same way bot.py itself is: it *executes* bar-by-bar on 1-minute
candles (order fills, SL/TP hit detection all happen at 1m granularity)
but *analyzes* higher timeframes -- every 1m row sees the full indicator
+ entry-signal stack recomputed on 5m, 15m, 1h and 4h candles resampled
from that same 1m data (merged back on with a 5m_/15m_/1h_/4h_ prefix,
forward-filled so a bar only ever sees the most recently *closed*
higher-tf candle -- no lookahead), plus a second stack computed directly
on the 1m data itself (prefixed 1m_). This is exactly bot.py's own
multi-timeframe merge (see its add_indicators()), just run across five
timeframes instead of two, and with a much smaller indicator stack per
timeframe. On any bar, at most one of the three strategies below can
produce a signal -- see _select_candidate() for the priority order used
when more than one would fire at once. Whatever it picks is taken
directly, with no learned model deciding whether to -- the whole bot is
deterministic technical rules end to end.

STRATEGY 1 -- stoch/%R zone-breakout (see _add_base_indicators(),
_signal_columns()), 1:4 RR (50-pip SL / 200-pip TP = TP_PIPS):

  Per analyzed timeframe (5m/15m/1h/4h):
    - EMA 7 / 21, each vs. price (distance) and its own slope
    - HalfTrend (bullish/bearish), plus its own distance from price
    - ADX (+DI/-DI) as a trend-strength floor
    - Stochastic oscillator (%K / %K-smooth) and Williams %R
    - The stoch/%R zone-breakout entry signal (below)

  A bar counts as oversold once %K < 20 OR %R < -80, overbought once
  %K > 80 OR %R > -20. Once either state has held for ZONE_BARS (=5)
  consecutive bars *of that timeframe*, that run's high/low forms a
  "zone" -- a breakout above the zone high (confirmed by %K crossing
  back above %K-smooth) is bullish; a breakdown below the zone low (%K
  crossing back below %K-smooth) is bearish. A signal fires whenever
  ANY of the 4 timeframes signals a breakout with that timeframe's own
  ADX clearing ADX_MIN. Direction filter: only becomes a trade if the
  1h and/or 4h HalfTrend agrees (--htf-mode "any"/"both"). This is one
  reasonable reading of "stoch k </> k smooth, and/or %r, wait 5
  candles in ob/os zone, then breakout, executed on 5m/15m/1h/4h, only
  in direction with 1h and/or 4h halftrend" -- adjust the thresholds
  or _signal_columns()'s combination logic if a different reading was
  intended.

STRATEGIES 2 & 3 -- both computed directly on the 1m data itself (see
_add_m1_levels(), _level_signal_columns()), sharing a 1:2 RR (the same
50-pip SL / a 100-pip TP = LEVEL_TP_PIPS), and both gated by 1m
HalfTrend:

  2. PDH/PDL + Asia high/low break, retest and reversal (poi_reversal,
     see _poi_candidate()): a signal fires in whichever direction
     the 1m HalfTrend currently points, on any bar where price is
     within POI_REACH_PIPS (30) of the previous day's high, the
     previous day's low, the current Asia session's high, or its low
     (PDH/PDL/Asia-high/Asia-low distance, ported from bot.py's own
     PDHDistance/PDLDistance/AsiaHighDistance/AsiaLowDistance).
     HalfTrend's live state stands in for whether it's a breakout or a
     reversal at the level (POI_LEVEL_KINDS: HalfTrend continuing
     *through* a resistance level -- PDH/Asia high -- or a support
     level -- PDL/Asia low -- the other way is a breakout; going the
     *other* way right at the level is a reversal), rather than
     classifying breakout/retest/reversal as three separate signals.
     Capped at one trade per (level, breakout-or-reversal) per
     calendar day -- 8 combinations total (PDH breakout, PDH reversal,
     PDL breakout, PDL reversal, same for Asia high/low) -- so e.g. a
     PDH breakout and a PDH reversal can each fire once the same day,
     but a second PDH breakout that day can't, regardless of outcome.

  3. OB mitigation (ob_mitigation): a signal fires when the 1m
     HalfTrend *flips* direction (not just agrees) on a bar that's
     also mitigating -- within the same POI_REACH_PIPS reach, via
     OBMitigation(), ported from bot.py's own BullishOB/BearishOB/
     OBMitigation -- a same-direction order block: a bullish/demand OB
     mitigated with HalfTrend flipping up is a reversal long, a
     bearish/supply OB mitigated with HalfTrend flipping down is a
     reversal short.

Position sizing is a flat fraction of account balance (--risk, default
0.01 = 1%) on every trade -- no confidence-based scaling of any kind,
since there's no model producing a confidence score to scale by.

Weekly stats (trade count, PnL, R-multiple, win rate, mean win/loss,
streaks, Z-score, profit factor, recovery factor, Sharpe, Sortino, and
a trades-by-strategy breakdown) print every simulated week during
--train, same cadence and metrics as bot.py's own weekly report (plus
the strategy breakdown, which bot.py's single-strategy report has no
need for) -- there's no training happening, "week" here is purely the
backtest's own reporting cadence (see TRADING_WEEK_BARS).

Usage mirrors bot.py:
    python bot_v2.py --train
    python bot_v2.py --test --symbol XAUUSD-STDc --risk 0.01
    python bot_v2.py --train --test --risk 0.02 --htf-mode both
"""

import pandas as pd
import numpy as np
import os
import time
import argparse
import multiprocessing
from io import StringIO
from datetime import datetime, timedelta

# MetaTrader5 only ships Windows wheels and needs a running terminal --
# imported lazily/defensively so the indicator pipeline and --train
# backtest stay usable on any platform; test_bot() fails loudly (not at
# import time) if it's missing.
try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None

# ==========================================================================
# ACTIONS -- 0=buy, 1=sell, 2=hold, per spec (note this is NOT the same
# ordering bot.py uses for its own agent).
# ==========================================================================
ACTIONS = ["buy", "sell", "hold"]
BUY, SELL, HOLD = 0, 1, 2

# ==========================================================================
# STRATEGY CONSTANTS
# ==========================================================================
SL_PIPS = 50.0                  # fixed stop-loss
RR_RATIO = 4.0                  # 1:4 risk:reward
TP_PIPS = SL_PIPS * RR_RATIO    # 200-pip target
PIP_VALUE = 0.1                 # $ per pip for XAUUSD, matches bot.py's convention
COMMISSION = 0.6                # pips, subtracted from every closed trade in the backtest

ADX_MIN = 20                    # trend-strength floor a timeframe's breakout must clear to count
ZONE_BARS = 5                   # consecutive OB/OS bars (of whichever tf) required before a breakout can fire
STOCH_OS, STOCH_OB = 20, 80     # %K oversold / overbought thresholds
WR_OS, WR_OB = -80, -20         # Williams %R oversold / overbought thresholds

TRADING_WEEK_BARS = 1440 * 5    # 1m bars in a 5-day trading week -- same definition as bot.py's save_count

# --- PDH/PDL + Asia high/low reversal strategy, and OB-mitigation
# strategy -- see _add_m1_levels()/_level_signal_columns() below. Both
# share the same 50-pip SL (SL_PIPS above) but trade a 1:2 RR, half
# the stoch-breakout strategy's 1:4.
LEVEL_RR_RATIO = 2.0
LEVEL_TP_PIPS = SL_PIPS * LEVEL_RR_RATIO  # 100-pip target
POI_REACH_PIPS = 30             # "within 30 pips of poi" -- shared by both level-based strategies
OB_MULTIPLIER = 1.5             # bot.py's BullishOB/BearishOB impulse-candle size multiplier
OB_LOOKBACK = 72                # bot.py's OBMitigation() lookback, in bars

MAGIC = 234567                  # MT5 order/position tag for this bot -- distinct from bot.py's 123456


# ==========================================================================
# INDICATORS
# ==========================================================================

def EMA(df, period):
    return df["Close"].ewm(span=period, adjust=False).mean().round(2)


def ADX(df, period=14):
    """Returns +DI, -DI and ADX using Wilder's smoothing. Columns
    required: High, Low, Close. (Same implementation as bot.py's ADX.)"""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    up = high.diff()
    dn = -low.diff()

    plus_dm_array = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm_array = np.where((dn > up) & (dn > 0), dn, 0.0)

    plus_dm = pd.Series(plus_dm_array, index=df.index)
    minus_dm = pd.Series(minus_dm_array, index=df.index)

    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    return round(adx, 2), round(plus_di, 2), round(minus_di, 2)


def STOCH(df, period=14, smooth_d=3):
    """Returns %K and %K-smooth. Columns required: High, Low, Close.
    (Same implementation as bot.py's STOCH.)"""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()

    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d = k.rolling(window=smooth_d).mean()

    return round(k, 2), round(d, 2)


def WilliamsR(df, period=14):
    """Williams %R: -100 * (highest_high - close) / (highest_high -
    lowest_low). Ranges -100 (oversold extreme) to 0 (overbought
    extreme). Columns required: High, Low, Close."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()

    r = -100 * (highest_high - close) / (highest_high - lowest_low)

    return round(r, 2)


def HalfTrend(df, amplitude=2):
    """Port of the "HalfTrend" indicator (Alex Orekhov / everget) -- a
    trailing trend band that flips once price and both extreme-EMAs
    confirm a break of the opposite side's recent range. Returns
    (bullish_halftrend, bearish_halftrend, half_trend_line). (Same
    implementation as bot.py's HalfTrend.)"""
    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()
    close = df["Close"].to_numpy()
    n = len(df)

    high_ma = df["High"].ewm(span=amplitude, adjust=False).mean().to_numpy()
    low_ma = df["Low"].ewm(span=amplitude, adjust=False).mean().to_numpy()
    highest_high = df["High"].rolling(amplitude).max().to_numpy()
    lowest_low = df["Low"].rolling(amplitude).min().to_numpy()

    trend = np.zeros(n, dtype=np.int8)
    next_trend = np.zeros(n, dtype=np.int8)
    max_low = np.full(n, np.nan)
    min_high = np.full(n, np.nan)
    half_trend_line = np.full(n, np.nan)

    start = amplitude
    if start >= n:
        empty = np.zeros(n, dtype=np.int8)
        return empty, empty, half_trend_line

    max_low[start] = low[start]
    min_high[start] = high[start]
    half_trend_line[start] = close[start]

    for i in range(start + 1, n):

        if next_trend[i - 1] == 0:
            max_low[i] = max(max_low[i - 1], lowest_low[i])

            if high_ma[i] < max_low[i] and close[i] < low[i - 1]:
                trend[i] = 1
                next_trend[i] = 1
                min_high[i] = highest_high[i]
            else:
                trend[i] = trend[i - 1]
                next_trend[i] = 0
                min_high[i] = min_high[i - 1]

        else:
            min_high[i] = min(min_high[i - 1], highest_high[i])

            if low_ma[i] > min_high[i] and close[i] > high[i - 1]:
                trend[i] = 0
                next_trend[i] = 0
                max_low[i] = lowest_low[i]
            else:
                trend[i] = trend[i - 1]
                next_trend[i] = 1
                max_low[i] = max_low[i - 1]

        if trend[i] == 0:
            half_trend_line[i] = (
                max(max_low[i], half_trend_line[i - 1])
                if trend[i - 1] == 0
                else max_low[i]
            )
        else:
            half_trend_line[i] = (
                min(min_high[i], half_trend_line[i - 1])
                if trend[i - 1] == 1
                else min_high[i]
            )

    bullish_halftrend = (trend == 0).astype(np.int8)
    bearish_halftrend = (trend == 1).astype(np.int8)

    return bullish_halftrend, bearish_halftrend, half_trend_line


def _run_length(mask):
    """Consecutive-True run length of a boolean Series, restarting at 0
    wherever it's False -- e.g. [F,T,T,F,T,T,T] -> [0,1,2,0,1,2,3]."""
    streak_id = (~mask).cumsum()
    run = mask.groupby(streak_id).cumcount() + 1
    return run.where(mask, 0)


def stoch_r_zone_breakout(df, zone_bars=ZONE_BARS):
    """Entry-gate signal: a bar is oversold once %K < STOCH_OS or %R <
    WR_OS, overbought once %K > STOCH_OB or %R > WR_OB. Once either
    state has held zone_bars consecutive bars, a close breaking above
    that run's High (for oversold) or below its Low (for overbought),
    confirmed by %K crossing %K-smooth in the same direction, is a
    breakout signal. Returns (bull_breakout, bear_breakout, os_streak,
    ob_streak), all aligned to df.index. Requires columns k, k_smooth,
    williams_r, High, Low, Close (i.e. run after STOCH/WilliamsR)."""
    k = df["k"]
    k_smooth = df["k_smooth"]
    wr = df["williams_r"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    oversold = (k < STOCH_OS) | (wr < WR_OS)
    overbought = (k > STOCH_OB) | (wr > WR_OB)

    os_streak = _run_length(oversold)
    ob_streak = _run_length(overbought)

    # Rolling zone_bars high/low, shifted so "the zone" only ever refers
    # to bars strictly before the one being evaluated -- no lookahead.
    zone_high = high.rolling(zone_bars).max().shift(1)
    zone_low = low.rolling(zone_bars).min().shift(1)

    zone_formed_os = os_streak.shift(1) >= zone_bars
    zone_formed_ob = ob_streak.shift(1) >= zone_bars

    bull_breakout = (
        zone_formed_os & (close > zone_high) & (k > k_smooth)
    ).fillna(False)

    bear_breakout = (
        zone_formed_ob & (close < zone_low) & (k < k_smooth)
    ).fillna(False)

    return bull_breakout, bear_breakout, os_streak, ob_streak


# ==========================================================================
# LEVEL / ORDER-BLOCK INDICATORS -- ported from bot.py, for the PDH/PDL
# + Asia high/low reversal strategy and the OB-mitigation strategy
# below (both gated by 1m HalfTrend, computed via _add_m1_levels()).
# ==========================================================================

def GetRange(df):
    """Candle range in pips -- bot.py's own convention,
    (High-Low)*10, i.e. already pip units given PIP_VALUE=0.1."""
    return (df["High"] - df["Low"]) * 10


def RangeMA(df, period=14):
    """Rolling mean of df["range"] (GetRange()). Ported from bot.py's
    RangeMA()."""
    return round(df["range"].rolling(period).mean(), 2)


def BullishOB(df, multiplier=OB_MULTIPLIER):
    """Bullish (demand) order block: the last down-candle before an
    up-candle at least `multiplier`x its size, with the down-candle's
    own body bigger than the recent average range. Ported verbatim
    from bot.py's BullishOB(). Requires df["range_ma"] (GetRange() +
    RangeMA()) to already be set."""
    body = (df["Close"] - df["Open"]).abs()
    next_body = body.shift(-1)

    bearish = df["Close"] < df["Open"]
    next_bullish = df["Close"].shift(-1) > df["Open"].shift(-1)

    return (
        bearish
        & next_bullish
        & (body > df["range_ma"])
        & (next_body >= body * multiplier)
    ).astype(int)


def BearishOB(df, multiplier=OB_MULTIPLIER):
    """Bearish (supply) order block -- mirror of BullishOB(). Ported
    verbatim from bot.py's BearishOB()."""
    body = (df["Close"] - df["Open"]).abs()
    next_body = body.shift(-1)

    bullish = df["Close"] > df["Open"]
    next_bearish = df["Close"].shift(-1) < df["Open"].shift(-1)

    return (
        bullish
        & next_bearish
        & (body > df["range_ma"])
        & (next_body >= body * multiplier)
    ).astype(int)


def OBMitigation(df, threshold_pips=POI_REACH_PIPS, lookback=OB_LOOKBACK):
    """True on any bar within `threshold_pips` of a prior order
    block's own candle low/high (not the binary bullish_ob/bearish_ob
    flag itself, which isn't a price level). Same algorithm as bot.py's
    OBMitigation(), adapted to take its threshold in pips -- bot.py's
    own version takes a raw-price threshold (its default of 30 is
    literally $30, i.e. 300 pips at PIP_VALUE=0.1); this version's
    default of 30 is genuinely 30 pips, per spec ("same 30 pip
    reach")."""
    threshold = threshold_pips * PIP_VALUE

    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()

    bullish_ob = df["bullish_ob"].to_numpy().astype(bool)
    bearish_ob = df["bearish_ob"].to_numpy().astype(bool)

    bull = np.zeros(len(df), dtype=np.bool_)
    bear = np.zeros(len(df), dtype=np.bool_)

    n = len(df)

    for i in range(n):
        start = max(0, i - lookback)

        for j in range(i - 1, start - 1, -1):
            if bullish_ob[j] and abs(low[i] - low[j]) <= threshold:
                bull[i] = True
                break

        for j in range(i - 1, start - 1, -1):
            if bearish_ob[j] and abs(high[i] - high[j]) <= threshold:
                bear[i] = True
                break

    return bull, bear


def PDHDistance(df):
    """Previous day's high minus current close, in raw price. Ported
    verbatim from bot.py's PDHDistance(). Requires a DatetimeIndex."""
    day = df.index.date

    daily_high = df["High"].groupby(day).transform("max")

    pdh = (
        daily_high
        .groupby(day)
        .first()
        .shift(1)
        .reindex(day)
        .to_numpy()
    )

    return pdh - df["Close"]


def PDLDistance(df):
    """Current close minus previous day's low, in raw price. Ported
    verbatim from bot.py's PDLDistance()."""
    day = df.index.date

    daily_low = df["Low"].groupby(day).transform("min")

    pdl = (
        daily_low
        .groupby(day)
        .first()
        .shift(1)
        .reindex(day)
        .to_numpy()
    )

    return df["Close"] - pdl


def AsiaHighDistance(df):
    """Current Asia session's high minus current close, in raw price.
    Adapted from bot.py's AsiaHighDistance(): its session mask,
    `(hour >= 1) | (hour <= 9)`, is a bug (that OR covers nearly the
    entire day, not a session) -- this uses AND, matching that
    function's own "01:00-08:59" docstring and bot.py's GetKillzone()
    asia-session convention."""
    asia = (df.index.hour >= 1) & (df.index.hour < 9)

    trade_day = (df.index - pd.Timedelta(hours=24)).date

    asia_high = (
        df["High"]
        .where(asia)
        .groupby(trade_day)
        .transform("max")
        .ffill()
    )

    return asia_high - df["Close"]


def AsiaLowDistance(df):
    """Current close minus current Asia session's low, in raw price.
    Same OR->AND session-mask fix as AsiaHighDistance() above."""
    asia = (df.index.hour >= 1) & (df.index.hour < 9)

    trade_day = (df.index - pd.Timedelta(hours=24)).date

    asia_low = (
        df["Low"]
        .where(asia)
        .groupby(trade_day)
        .transform("min")
        .ffill()
    )

    return df["Close"] - asia_low


# ==========================================================================
# FEATURE PIPELINE
# ==========================================================================

# The indicator + entry-signal stack computed independently on each
# analyzed timeframe (see _add_base_indicators()) -- never on the raw
# 1m data itself, which only supplies OHLC for execution.
BASE_INDICATOR_FEATURES = [
    "k", "k_smooth", "williams_r", "adx", "+di", "-di",
    "EMA7_dist", "EMA7_slope", "EMA21_dist", "EMA21_slope",
    "bullish_halftrend", "bearish_halftrend", "halftrend_dist",
    "os_streak", "ob_streak", "bull_breakout", "bear_breakout",
]

# (column prefix, pandas resample frequency) -- the four timeframes
# analyzed off of the raw 1m execution data. 1h/4h double as the
# direction filter (see _htf_ok_columns()); all four feed the
# stoch/%R zone-breakout entry signal (see bull_signal/bear_signal in
# train_bot()/test_bot()).
ANALYZED_TIMEFRAMES = (
    ("5m", "5min"),
    ("15m", "15min"),
    ("1h", "1h"),
    ("4h", "4h"),
)

# The indicator stack computed directly on the raw 1m execution data
# itself (never resampled) -- see _add_m1_levels() -- feeding the
# PDH/PDL + Asia high/low reversal strategy and the OB-mitigation
# strategy, both gated by this same 1m HalfTrend.
M1_LEVEL_FEATURES = [
    "bullish_halftrend", "bearish_halftrend",
    "bullish_ob", "bearish_ob",
    "bullish_ob_mitigation", "bearish_ob_mitigation",
    "pdh_dist", "pdl_dist", "asia_high_dist", "asia_low_dist",
]


def _add_base_indicators(df):
    """Adds the shared indicator + zone-breakout stack
    (BASE_INDICATOR_FEATURES) to an OHLC dataframe at whatever
    timeframe it's given. Called once per analyzed timeframe by
    add_indicators() below -- never on the raw 1m data directly."""
    df = df.copy()

    df["adx"], df["+di"], df["-di"] = ADX(df)
    df["k"], df["k_smooth"] = STOCH(df)
    df["williams_r"] = WilliamsR(df)

    df["EMA7"] = EMA(df, 7)
    df["EMA7_slope"] = df["EMA7"].diff()
    df["EMA7_dist"] = df["Close"] - df["EMA7"]

    df["EMA21"] = EMA(df, 21)
    df["EMA21_slope"] = df["EMA21"].diff()
    df["EMA21_dist"] = df["Close"] - df["EMA21"]

    df["bullish_halftrend"], df["bearish_halftrend"], halftrend_line = HalfTrend(df)
    df["halftrend_dist"] = df["Close"] - halftrend_line

    df["bull_breakout"], df["bear_breakout"], df["os_streak"], df["ob_streak"] = (
        stoch_r_zone_breakout(df)
    )

    return df[BASE_INDICATOR_FEATURES]


def _add_m1_levels(df):
    """Adds M1_LEVEL_FEATURES -- HalfTrend plus the order-block and
    PDH/PDL/Asia-session levels the two level-based strategies trade
    off of -- directly on the raw 1m execution data. Unlike
    _add_base_indicators() (run on a resampled higher-tf copy and
    merged back with a lag), this runs on the same 1m frame the bot
    executes on, so every value is already "live" at the bar it's
    computed on -- no forward-fill needed."""
    df = df.copy()

    df["bullish_halftrend"], df["bearish_halftrend"], _ = HalfTrend(df)

    df["range"] = GetRange(df)
    df["range_ma"] = RangeMA(df)
    df["bullish_ob"] = BullishOB(df)
    df["bearish_ob"] = BearishOB(df)
    df["bullish_ob_mitigation"], df["bearish_ob_mitigation"] = OBMitigation(df)

    df["pdh_dist"] = PDHDistance(df) / PIP_VALUE
    df["pdl_dist"] = PDLDistance(df) / PIP_VALUE
    df["asia_high_dist"] = AsiaHighDistance(df) / PIP_VALUE
    df["asia_low_dist"] = AsiaLowDistance(df) / PIP_VALUE

    return df[M1_LEVEL_FEATURES]


def add_indicators(df):
    """df must be raw 1-minute OHLC candles (Open/High/Low/Close). The
    bot executes at this same 1m granularity (bar-by-bar SL/TP
    tracking, order fills) -- same execution model as bot.py.

    Two kinds of indicators get merged onto every 1m row:
      - M1_LEVEL_FEATURES (_add_m1_levels): computed directly on the
        1m data itself -- HalfTrend, order blocks/mitigation, PDH/PDL
        and Asia high/low distance -- prefixed 1m_.
      - BASE_INDICATOR_FEATURES (_add_base_indicators): computed on
        each of ANALYZED_TIMEFRAMES (5m/15m/1h/4h) resampled from this
        same 1m data (right-labeled/left-closed, so a bin is only
        visible once it has actually closed), then merged back
        forward-filled so a bar only ever sees the most recently
        *closed* higher-tf candle -- prefixed 5m_/15m_/1h_/4h_. Same
        no-lookahead multi-timeframe merge bot.py's own
        add_indicators() uses, just run across four timeframes instead
        of two.
    """

    result = df[["Open", "High", "Low", "Close"]].copy()

    m1_levels = _add_m1_levels(df).add_prefix("1m_")
    result = pd.concat([result, m1_levels], axis=1)

    for prefix, freq in ANALYZED_TIMEFRAMES:

        df_tf = df.resample(
            freq, label="right", closed="left"
        ).agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
        }).dropna()

        df_tf = _add_base_indicators(df_tf).add_prefix(f"{prefix}_")

        result = pd.concat(
            [result, df_tf.reindex(result.index, method="ffill")],
            axis=1
        )

    result.dropna(inplace=True)
    return result


def load_last_mb_xauusd(file_path=None, mb=20, delimiter=",", col_names=None):
    """Same loader as bot.py's -- reads only the last `mb` megabytes of
    the CSV rather than the whole (potentially huge) file."""
    if file_path is None:
        file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "download", "XAUUSD.csv"
        )

    print(f"Loading: {file_path}")

    file_size = os.path.getsize(file_path)
    offset = max(file_size - mb * 1024 * 1024, 0)

    with open(file_path, "rb") as f:
        f.seek(offset)
        data = f.read().decode(errors="ignore")

        if offset > 0:
            data = data.split("\n", 1)[-1]

    df = pd.read_csv(StringIO(data), delimiter=delimiter, header=0)

    df.columns = ["Date", "Timestamp", "Open", "High", "Low", "Close", "Volume"]

    df["Date"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Timestamp"],
        format="%Y%m%d %H:%M:%S",
        utc=True
    )
    df.set_index("Date", inplace=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

    print(f"Loaded: {file_path}")

    return df.dropna()


# ==========================================================================
# STATS HELPERS -- identical to bot.py's.
# ==========================================================================

def sharpe_ratio(returns, risk_free_rate=0.0):
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    if std_ret == 0:
        return 0
    return (mean_ret - risk_free_rate) / std_ret


def sortino_ratio(returns, risk_free_rate=0.0):
    mean_ret = np.mean(returns)
    downside_diff = [(r - risk_free_rate) ** 2 for r in returns if r < risk_free_rate]

    if len(downside_diff) == 0:
        return 0

    downside_std = np.sqrt(np.mean(downside_diff))

    if downside_std == 0:
        return 0

    return (mean_ret - risk_free_rate) / downside_std


def max_drawdown(returns):
    if len(returns) == 0:
        return 0

    equity = np.cumsum(returns)
    peak = equity[0]
    max_dd = 0

    for value in equity:
        peak = max(peak, value)
        dd = peak - value
        max_dd = max(max_dd, dd)

    return max_dd


def streak_stats(returns):
    win_streaks = []
    loss_streaks = []

    current_len = 0
    current_sign = 0

    for r in returns:
        sign = 1 if r > 0 else (-1 if r < 0 else 0)

        if sign != 0 and sign == current_sign:
            current_len += 1
        else:
            if current_sign == 1 and current_len > 0:
                win_streaks.append(current_len)
            elif current_sign == -1 and current_len > 0:
                loss_streaks.append(current_len)
            current_len = 1 if sign != 0 else 0
            current_sign = sign

    if current_sign == 1 and current_len > 0:
        win_streaks.append(current_len)
    elif current_sign == -1 and current_len > 0:
        loss_streaks.append(current_len)

    avg_win_streak = sum(win_streaks) / len(win_streaks) if win_streaks else 0
    avg_loss_streak = sum(loss_streaks) / len(loss_streaks) if loss_streaks else 0

    return avg_win_streak, avg_loss_streak


# ==========================================================================
# MT5 I/O
# ==========================================================================

def open_long(symbol, lot_size, sl_pips, tp_pips, magic=MAGIC):
    tick = mt5.symbol_info_tick(symbol)
    entry = tick.ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": mt5.ORDER_TYPE_BUY,
        "price": entry,
        "sl": entry - sl_pips / 10,
        "tp": entry + tp_pips / 10,
        "deviation": 20,
        "magic": magic,
        "comment": "stoch-halftrend",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC
    }

    result = mt5.order_send(request)

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order failed: {result.retcode}")
        return None


def open_short(symbol, lot_size, sl_pips, tp_pips, magic=MAGIC):
    tick = mt5.symbol_info_tick(symbol)
    entry = tick.bid

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": mt5.ORDER_TYPE_SELL,
        "price": entry,
        "sl": entry + sl_pips / 10,
        "tp": entry - tp_pips / 10,
        "deviation": 20,
        "magic": magic,
        "comment": "stoch-halftrend",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC
    }

    result = mt5.order_send(request)

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order failed: {result.retcode}")
        return None


def open_positions(symbol, magic=MAGIC):
    positions = mt5.positions_get(symbol=symbol)
    positions = [p for p in positions if p.magic == magic]
    return len(positions)


def get_today_profit(symbol, day, now=None, magic=MAGIC):
    day_start = datetime.combine(day, datetime.min.time())

    if now is None:
        now = datetime.now()

    deals = mt5.history_deals_get(day_start, now + timedelta(days=1))

    if deals is None:
        return 0.0

    return sum(
        d.profit + d.swap + d.commission
        for d in deals
        if d.magic == magic
        and d.symbol == symbol
        and d.entry == mt5.DEAL_ENTRY_OUT
    )


def close_trades(magic=MAGIC):
    positions = mt5.positions_get()

    if positions is None:
        print("Failed to get positions:", mt5.last_error())
        return

    for pos in positions:
        if pos.magic != magic:
            continue

        tick = mt5.symbol_info_tick(pos.symbol)
        if tick is None:
            continue

        if pos.type == mt5.POSITION_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": order_type,
            "position": pos.ticket,
            "price": price,
            "deviation": 20,
            "magic": magic,
            "comment": "End of Day",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Failed to close {pos.ticket}: {result.retcode}")


# ==========================================================================
# CANDIDATE SELECTION + BACKTEST
# ==========================================================================

def _htf_ok_columns(df, htf_mode):
    if htf_mode == "any":
        long_ok = (df["1h_bullish_halftrend"] == 1) | (df["4h_bullish_halftrend"] == 1)
        short_ok = (df["1h_bearish_halftrend"] == 1) | (df["4h_bearish_halftrend"] == 1)
    else:  # "both"
        long_ok = (df["1h_bullish_halftrend"] == 1) & (df["4h_bullish_halftrend"] == 1)
        short_ok = (df["1h_bearish_halftrend"] == 1) & (df["4h_bearish_halftrend"] == 1)
    return long_ok, short_ok


def _signal_columns(df):
    """bull_signal/bear_signal: True on any bar where ANY analyzed
    timeframe (5m/15m/1h/4h) has a live zone-breakout AND that same
    timeframe's own ADX clears ADX_MIN -- i.e. the stoch/%R zone-
    breakout strategy is "executed on" all four timeframes at once,
    any one of them firing is enough to produce a candidate."""
    bull_signal = pd.Series(False, index=df.index)
    bear_signal = pd.Series(False, index=df.index)

    for prefix, _ in ANALYZED_TIMEFRAMES:
        tf_adx_ok = df[f"{prefix}_adx"] >= ADX_MIN
        bull_signal |= df[f"{prefix}_bull_breakout"].astype(bool) & tf_adx_ok
        bear_signal |= df[f"{prefix}_bear_breakout"].astype(bool) & tf_adx_ok

    return bull_signal, bear_signal


# PDH/PDL are resistance/support the same way every day; Asia high/low
# are resistance/support for that day's session. HalfTrend continuing
# *through* a resistance level (bullish) or a support level (bearish)
# is a breakout; HalfTrend going the *other* way right at the level
# (bearish at resistance, bullish at support) is a rejection/reversal.
# Order here is also _poi_candidate()'s fixed priority when more than
# one level is in reach on the same bar.
POI_LEVEL_KINDS = (
    ("pdh", "resistance"),
    ("pdl", "support"),
    ("asia_high", "resistance"),
    ("asia_low", "support"),
)


def _level_signal_columns(df):
    """ob_bull/ob_bear ("OB mitigation"): the 1m HalfTrend *flipping*
    direction this bar (not just agreeing, per spec's "halftrend
    reversal") on a bar that's also mitigating (within POI_REACH_PIPS
    of, via OBMitigation()'s own threshold) a same-direction order
    block -- a bullish/demand OB mitigated with HalfTrend flipping up
    is a reversal long, a bearish/supply OB mitigated with HalfTrend
    flipping down is a reversal short.

    (The PDH/PDL/Asia high-low strategy's candidate selection lives in
    _poi_candidate() instead of here, since -- unlike this one -- it
    needs per-bar state (at most one trade per level per
    breakout-or-reversal event per day) that a pure vectorized column
    can't express.)
    """
    bull_ht = df["1m_bullish_halftrend"].astype(bool)
    bear_ht = df["1m_bearish_halftrend"].astype(bool)

    flip_bull = bull_ht & ~bull_ht.shift(1).fillna(False)
    flip_bear = bear_ht & ~bear_ht.shift(1).fillna(False)

    ob_bull = flip_bull & df["1m_bullish_ob_mitigation"].astype(bool)
    ob_bear = flip_bear & df["1m_bearish_ob_mitigation"].astype(bool)

    return ob_bull, ob_bear


def _poi_candidate(bull_ht, bear_ht, near_pdh, near_pdl, near_asia_high, near_asia_low, used_today):
    """Picks a PDH/PDL/Asia high/low candidate for the current bar --
    "PDH/PDL + Asia high/low break, retest and reversal", per spec, at
    most once per (level, breakout-or-reversal) per calendar day. HalfTrend's
    live state stands in for whichever of breakout/retest/reversal is
    actually happening at the level (see POI_LEVEL_KINDS) rather than
    classifying the three separately, same as before -- what's new is
    that each (level, event) combination -- e.g. "pdh breakout", "pdh
    reversal", same for pdl/asia_high/asia_low, 8 combinations total --
    can only produce a candidate once per day; `used_today` is a set of
    "{level}_{event}" keys already spent today, and it's the caller's
    job to add the key this returns to it once a trade off it actually
    opens (not just whenever the candidate fires -- a bar that's
    already in_position, so nothing gets opened off it, doesn't spend
    the day's slot). Levels are checked in POI_LEVEL_KINDS order;
    ties (more than one level in reach at once) go to the first.
    Returns (action, key) or (HOLD, None)."""
    near_by_level = {
        "pdh": near_pdh, "pdl": near_pdl,
        "asia_high": near_asia_high, "asia_low": near_asia_low,
    }

    for level, kind in POI_LEVEL_KINDS:
        if not near_by_level[level]:
            continue

        if kind == "resistance":
            if bull_ht:
                action, event = BUY, "breakout"
            elif bear_ht:
                action, event = SELL, "reversal"
            else:
                continue
        else:  # support
            if bear_ht:
                action, event = SELL, "breakout"
            elif bull_ht:
                action, event = BUY, "reversal"
            else:
                continue

        key = f"{level}_{event}"
        if key in used_today:
            continue

        return action, key

    return HOLD, None


def _select_candidate(
    bull_stoch, bear_stoch, long_htf_ok, short_htf_ok,
    bull_ht, bear_ht, near_pdh, near_pdl, near_asia_high, near_asia_low, poi_used_today,
    ob_bull, ob_bear,
):
    """Picks a candidate (action, tp_pips, strategy_name, poi_key) for
    the current bar across all three entry strategies, in a fixed
    priority order: stoch/%R zone-breakout first, then the
    PDH/PDL/Asia-level reversal (itself further gated by
    _poi_candidate()'s once-per-level-per-event-per-day cap), then OB
    mitigation. More than one firing on the same bar (in the same or
    opposite directions) is rare given how differently each triggers,
    and resolved by this order rather than reconciled -- whatever this
    returns is taken directly (see train_bot()/test_bot()), there's no
    further "should we take this" decision downstream. poi_key is only
    meaningful when strategy_name == "poi_reversal";
    the caller adds it to poi_used_today once (and only once) a trade
    off it actually opens. Returns (HOLD, TP_PIPS, None, None) if
    nothing fires."""
    if bull_stoch and long_htf_ok:
        return BUY, TP_PIPS, "stoch_breakout", None
    if bear_stoch and short_htf_ok:
        return SELL, TP_PIPS, "stoch_breakout", None

    poi_action, poi_key = _poi_candidate(
        bull_ht, bear_ht, near_pdh, near_pdl, near_asia_high, near_asia_low, poi_used_today
    )
    if poi_action != HOLD:
        return poi_action, LEVEL_TP_PIPS, "poi_reversal", poi_key

    if ob_bull:
        return BUY, LEVEL_TP_PIPS, "ob_mitigation", None
    if ob_bear:
        return SELL, LEVEL_TP_PIPS, "ob_mitigation", None

    return HOLD, TP_PIPS, None, None


def _print_period_stats(symbol, label, trade_returns, strategy_counts):
    """Prints one stats block (see train_bot()'s WEEKLY STATS comment
    for why every value is guarded against an empty trade_returns).
    Shared by the weekly cadence and the final partial period at the
    end of the backtest."""
    wins = [r for r in trade_returns if r > 0]
    losses = [r for r in trade_returns if r < 0]

    weekly_pnl = np.sum(trade_returns)
    winrate = len(wins) / len(trade_returns) if trade_returns else 0.0
    mean_win = np.mean(wins) if wins else 0.0
    mean_loss = np.mean(losses) if losses else 0.0

    sharpe = sharpe_ratio(trade_returns) if trade_returns else 0.0
    sortino = sortino_ratio(trade_returns) if trade_returns else 0.0

    std_ret = np.std(trade_returns) if trade_returns else 0.0
    zscore = np.mean(trade_returns) / std_ret if std_ret > 0 else 0.0

    avg_win_streak, avg_loss_streak = streak_stats(trade_returns)

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    max_dd = max_drawdown(trade_returns)
    R_pnl = weekly_pnl / SL_PIPS
    rf = R_pnl / (max_dd / SL_PIPS) if max_dd > 0 else 0.0

    print()
    print("================================================")
    print(f"[{symbol}] {label}")
    print("================================================")
    print(f"Trades:          {len(trade_returns)}")
    print(f"PnL:             {weekly_pnl:.0f} pips")
    print(f"R PnL:           {R_pnl:.2f}R")
    print(f"Max DD:          {max_dd/SL_PIPS:.2f}R")
    print(f"Winrate:         {winrate*100:.2f}%")
    print(f"Mean Win:        {mean_win:.0f} pips")
    print(f"Mean Loss:       {mean_loss:.0f} pips")
    print(f"Avg Win Streak:  {avg_win_streak:.2f}")
    print(f"Avg Loss Streak: {avg_loss_streak:.2f}")
    print(f"Z-score:         {zscore:.2f}")
    print(f"PF:              {profit_factor:.2f}")
    print(f"RF:              {rf:.2f}")
    print(f"Sharpe:          {sharpe:.2f}")
    print(f"Sortino:         {sortino:.2f}")
    print(
        f"By strategy:     "
        f"stoch={strategy_counts['stoch_breakout']} "
        f"poi={strategy_counts['poi_reversal']} "
        f"ob={strategy_counts['ob_mitigation']}"
    )
    print("================================================")
    print()


def train_bot(symbol="XAUUSD", risk=0.01, htf_mode="any"):
    """Backtests the three strategies bar-by-bar, once, over the
    historical 1m data -- there's no learned model, so unlike bot.py's
    train_bot() there's nothing to iterate passes over; a single pass
    is the whole backtest. Prints the same weekly stats bot.py's own
    train_bot() does, every TRADING_WEEK_BARS, plus a final block for
    whatever's left in the last partial week. --train is kept as the
    flag name (matching bot.py's convention) even though nothing here
    trains anything."""

    print("Backtesting (stoch/%R breakout + PDH/PDL/Asia reversal + OB mitigation)")
    start = time.perf_counter()

    TRAIN_HISTORY_MB = 20
    df_m1 = load_last_mb_xauusd(mb=TRAIN_HISTORY_MB)
    print(f"Computing 1m/5m/15m/1h/4h indicators over 1m execution data... ({time.strftime('%H:%M')})")
    df = add_indicators(df_m1)
    elapsed = int((time.perf_counter() - start) // 60)
    print(f"Loaded indicators on {len(df)} 1m bars (Elapsed: {elapsed}m)")

    long_htf_ok, short_htf_ok = _htf_ok_columns(df, htf_mode)
    df["long_htf_ok"] = long_htf_ok
    df["short_htf_ok"] = short_htf_ok

    bull_signal, bear_signal = _signal_columns(df)
    df["bull_signal"] = bull_signal
    df["bear_signal"] = bear_signal

    ob_bull, ob_bear = _level_signal_columns(df)
    df["ob_bull"] = ob_bull
    df["ob_bear"] = ob_bear

    df["near_pdh"] = df["1m_pdh_dist"].abs() <= POI_REACH_PIPS
    df["near_pdl"] = df["1m_pdl_dist"].abs() <= POI_REACH_PIPS
    df["near_asia_high"] = df["1m_asia_high_dist"].abs() <= POI_REACH_PIPS
    df["near_asia_low"] = df["1m_asia_low_dist"].abs() <= POI_REACH_PIPS

    # Precompute everything the loop needs as plain arrays once, up
    # front -- much cheaper than repeated .iloc[i] Series construction
    # over what can be a few hundred thousand 1m bars.
    close_arr = df["Close"].to_numpy()
    high_arr = df["High"].to_numpy()
    low_arr = df["Low"].to_numpy()
    bull_signal_arr = df["bull_signal"].to_numpy()
    bear_signal_arr = df["bear_signal"].to_numpy()
    long_htf_ok_arr = df["long_htf_ok"].to_numpy()
    short_htf_ok_arr = df["short_htf_ok"].to_numpy()
    bull_ht_arr = df["1m_bullish_halftrend"].to_numpy().astype(bool)
    bear_ht_arr = df["1m_bearish_halftrend"].to_numpy().astype(bool)
    near_pdh_arr = df["near_pdh"].to_numpy()
    near_pdl_arr = df["near_pdl"].to_numpy()
    near_asia_high_arr = df["near_asia_high"].to_numpy()
    near_asia_low_arr = df["near_asia_low"].to_numpy()
    ob_bull_arr = df["ob_bull"].to_numpy()
    ob_bear_arr = df["ob_bear"].to_numpy()
    date_arr = df.index.date

    save_counter = 0
    in_position = False
    position_type = None
    entry_price = sl_price = tp_price = 0.0

    # poi_reversal's once-per-(level, breakout-or-reversal)-per-day cap
    # -- see _poi_candidate(). Reset whenever the calendar date changes.
    poi_used_today = set()
    current_poi_day = None

    trade_returns = []
    strategy_counts = {"stoch_breakout": 0, "poi_reversal": 0, "ob_mitigation": 0}

    run_start = time.time()
    period_start = time.time()

    print(f"[{symbol}] [INFO] Backtesting {len(df)} bars...")

    for i in range(len(df)):

        current_price = close_arr[i]
        high = high_arr[i]
        low = low_arr[i]

        if date_arr[i] != current_poi_day:
            current_poi_day = date_arr[i]
            poi_used_today = set()

        # Candidate direction (and which of the three strategies it
        # comes from, and that strategy's own TP) comes entirely from
        # the deterministic technical stack -- whatever it returns is
        # taken directly, no learned model in the loop deciding
        # whether to.
        action, tp_pips, strategy, poi_key = _select_candidate(
            bull_signal_arr[i], bear_signal_arr[i],
            long_htf_ok_arr[i], short_htf_ok_arr[i],
            bull_ht_arr[i], bear_ht_arr[i],
            near_pdh_arr[i], near_pdl_arr[i], near_asia_high_arr[i], near_asia_low_arr[i],
            poi_used_today,
            ob_bull_arr[i], ob_bear_arr[i],
        )

        pnl = 0.0

        if action == BUY and not in_position:
            in_position = True
            position_type = "long"
            entry_price = current_price

            sl_price = entry_price - SL_PIPS * PIP_VALUE
            tp_price = entry_price + tp_pips * PIP_VALUE

            strategy_counts[strategy] += 1
            if strategy == "poi_reversal":
                poi_used_today.add(poi_key)

        elif action == SELL and not in_position:
            in_position = True
            position_type = "short"
            entry_price = current_price

            sl_price = entry_price + SL_PIPS * PIP_VALUE
            tp_price = entry_price - tp_pips * PIP_VALUE

            strategy_counts[strategy] += 1
            if strategy == "poi_reversal":
                poi_used_today.add(poi_key)

        if in_position:
            trade_closed = False

            if position_type == "long":
                if high >= tp_price:
                    pnl = (tp_price - entry_price) / PIP_VALUE - COMMISSION
                    trade_closed = True
                if low <= sl_price:
                    pnl = (sl_price - entry_price) / PIP_VALUE - COMMISSION
                    trade_closed = True

            elif position_type == "short":
                if low <= tp_price:
                    pnl = (entry_price - tp_price) / PIP_VALUE - COMMISSION
                    trade_closed = True
                if high >= sl_price:
                    pnl = (entry_price - sl_price) / PIP_VALUE - COMMISSION
                    trade_closed = True

            if trade_closed:
                in_position = False
                trade_returns.append(pnl)

        save_counter += 1

        # ==================================================================
        # WEEKLY STATS
        # ==================================================================
        # Printed every TRADING_WEEK_BARS regardless of how many trades
        # that stretch had -- including zero -- so a quiet week is
        # still visible rather than silently skipped. Every stat in
        # _print_period_stats() is guarded against an empty
        # trade_returns (np.mean/np.std/winrate would otherwise divide
        # by zero or warn); streak_stats() and max_drawdown() already
        # handle empty input on their own.
        if save_counter % TRADING_WEEK_BARS == 0:

            _print_period_stats(symbol, "WEEKLY STATS", trade_returns, strategy_counts)

            trade_returns = []
            strategy_counts = {"stoch_breakout": 0, "poi_reversal": 0, "ob_mitigation": 0}

            completed = int(save_counter / TRADING_WEEK_BARS)
            total = max(int(round(len(df) / TRADING_WEEK_BARS, 0)), 1)

            elapsed_run = time.time() - run_start
            avg_time = elapsed_run / max(completed, 1)
            remaining = max(total - completed, 0)
            eta = remaining * avg_time

            print(
                f"[{symbol}] [INFO] "
                f"{completed}/{total} ({completed/total*100:.1f}%) | "
                f"Elapsed: {timedelta(seconds=int(elapsed_run))} | "
                f"ETA: {timedelta(seconds=int(eta))} | "
                f"Period: {timedelta(seconds=int(time.time() - period_start))}"
            )

            period_start = time.time()

    # Whatever's left in the final, partial week.
    if trade_returns or any(strategy_counts.values()):
        _print_period_stats(symbol, "FINAL PARTIAL PERIOD", trade_returns, strategy_counts)

    print(
        f"[{symbol}] [INFO] Finished backtest "
        f"(Elapsed: {timedelta(seconds=int(time.time() - run_start))})"
    )


# ==========================================================================
# LIVE TRADING
# ==========================================================================

def _rename_mt5_rates(d):
    d = d.rename(columns={
        "open": "Open", "high": "High", "low": "Low", "close": "Close", "time": "Date"
    })
    d["Date"] = pd.to_datetime(d["Date"], unit="s", utc=True)
    d.set_index("Date", inplace=True)
    return d[["Open", "High", "Low", "Close"]]


def test_bot(symbol="XAUUSD", risk=0.01, htf_mode="any"):

    if mt5 is None:
        raise RuntimeError(
            "test_bot() needs the MetaTrader5 package and a running MT5 "
            "terminal (Windows-only) -- see README."
        )

    mt5.initialize()

    # ==========================================================
    # INITIAL LOAD
    # ==========================================================
    # 1-minute bars -- generous margin over the largest lookback (the
    # 4h branch's EMA21/ADX/STOCH need ~21 four-hour candles = ~5040
    # 1m bars) plus HalfTrend/zone-breakout warmup on every analyzed
    # timeframe, so add_indicators()'s dropna() never wipes out the
    # tail we actually need. Same role bot.py's LIVE_HISTORY_BARS
    # plays for its own (smaller, 15m-max) multi-timeframe merge.
    LIVE_HISTORY_BARS = 12000

    rates_m1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, LIVE_HISTORY_BARS)
    raw_df = _rename_mt5_rates(pd.DataFrame(rates_m1))

    df = add_indicators(raw_df.copy())

    last_m1_epoch = int(raw_df.index[-1].timestamp())

    # poi_reversal's once-per-(level, breakout-or-reversal)-per-day cap
    # -- see _poi_candidate(). Reset whenever the calendar date changes.
    poi_used_today = set()
    current_poi_day = df.index[-1].date()

    # ==========================================================
    # MAIN LOOP
    # ==========================================================

    while True:

        now = datetime.now()
        seconds_until_next_minute = 60 - now.second - now.microsecond / 1_000_000
        if seconds_until_next_minute <= 0:
            seconds_until_next_minute += 0.25
        time.sleep(seconds_until_next_minute)

        # ======================================================
        # CHECK FOR NEW M1 CANDLE
        # ======================================================
        new_m1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 1)
        current_m1_epoch = int(new_m1[0]["time"])

        while current_m1_epoch == last_m1_epoch:
            new_m1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 1)
            current_m1_epoch = int(new_m1[0]["time"])

        last_m1_epoch = current_m1_epoch

        # ======================================================
        # APPEND NEW CANDLE, RECOMPUTE INDICATORS
        # ======================================================
        new_row = _rename_mt5_rates(pd.DataFrame(new_m1))

        if new_row.index[-1] != raw_df.index[-1]:
            raw_df = pd.concat([raw_df, new_row])
            raw_df = raw_df.tail(LIVE_HISTORY_BARS)
            df = add_indicators(raw_df.copy())

        # ======================================================
        # DECISION
        # ======================================================
        open_pos = open_positions(symbol)

        current = df.iloc[-1]
        prev = df.iloc[-2]

        if htf_mode == "any":
            long_ok = bool(current["1h_bullish_halftrend"]) or bool(current["4h_bullish_halftrend"])
            short_ok = bool(current["1h_bearish_halftrend"]) or bool(current["4h_bearish_halftrend"])
        else:
            long_ok = bool(current["1h_bullish_halftrend"]) and bool(current["4h_bullish_halftrend"])
            short_ok = bool(current["1h_bearish_halftrend"]) and bool(current["4h_bearish_halftrend"])

        # Stoch/%R breakout candidate: ANY analyzed timeframe
        # (5m/15m/1h/4h) signalling a zone-breakout with its own ADX
        # clearing ADX_MIN is enough -- same _signal_columns() logic
        # as train_bot(), evaluated on just the latest bar here.
        bull_signal = any(
            bool(current[f"{prefix}_bull_breakout"]) and current[f"{prefix}_adx"] >= ADX_MIN
            for prefix, _ in ANALYZED_TIMEFRAMES
        )
        bear_signal = any(
            bool(current[f"{prefix}_bear_breakout"]) and current[f"{prefix}_adx"] >= ADX_MIN
            for prefix, _ in ANALYZED_TIMEFRAMES
        )

        # PDH/PDL + Asia high/low reversal candidate: 1m HalfTrend's
        # current direction within POI_REACH_PIPS of one of the four
        # levels, at most once per (level, breakout-or-reversal) per
        # day -- see _poi_candidate(), same logic as train_bot()'s.
        current_date = current.name.date()
        if current_date != current_poi_day:
            current_poi_day = current_date
            poi_used_today = set()

        bull_ht = bool(current["1m_bullish_halftrend"])
        bear_ht = bool(current["1m_bearish_halftrend"])

        near_pdh = abs(current["1m_pdh_dist"]) <= POI_REACH_PIPS
        near_pdl = abs(current["1m_pdl_dist"]) <= POI_REACH_PIPS
        near_asia_high = abs(current["1m_asia_high_dist"]) <= POI_REACH_PIPS
        near_asia_low = abs(current["1m_asia_low_dist"]) <= POI_REACH_PIPS

        # OB-mitigation candidate: 1m HalfTrend flipping direction this
        # bar (not just agreeing) while also mitigating a
        # same-direction order block -- same logic as
        # _level_signal_columns()'s ob_bull/ob_bear.
        flip_bull = bull_ht and not bool(prev["1m_bullish_halftrend"])
        flip_bear = bear_ht and not bool(prev["1m_bearish_halftrend"])
        ob_bull = flip_bull and bool(current["1m_bullish_ob_mitigation"])
        ob_bear = flip_bear and bool(current["1m_bearish_ob_mitigation"])

        # Candidate direction (and which of the three strategies it
        # comes from, and that strategy's own TP) comes entirely from
        # the deterministic technical stack -- taken directly below,
        # no learned model deciding whether to.
        action, candidate_tp_pips, candidate_strategy, candidate_poi_key = _select_candidate(
            bull_signal, bear_signal, long_ok, short_ok,
            bull_ht, bear_ht, near_pdh, near_pdl, near_asia_high, near_asia_low, poi_used_today,
            ob_bull, ob_bear,
        )

        current_time = df.index[-1]

        # Flatten and force HOLD heading into the daily close, same
        # convention as bot.py.
        if current_time.hour == 23 and current_time.minute >= 55:
            if open_pos != 0:
                close_trades()
            action = HOLD

        # ======================================================
        # OPEN NEW TRADE
        # ======================================================
        if open_pos == 0 and action in (BUY, SELL):

            account = mt5.account_info()
            balance = account.balance

            # Lot size: hitting the initial SL costs risk% of balance.
            lot = min(max((balance * risk) / (SL_PIPS * 10), 0.01), 100.0)
            lot = round(lot, 2)

            print(f"[{symbol}] Opening {ACTIONS[action]} via {candidate_strategy} "
                  f"(TP={candidate_tp_pips:.0f} pips, lot={lot})")

            if candidate_strategy == "poi_reversal":
                poi_used_today.add(candidate_poi_key)

            if action == BUY:
                open_long(symbol, lot, SL_PIPS, candidate_tp_pips)
            else:
                open_short(symbol, lot, SL_PIPS, candidate_tp_pips)


# ==========================================================================
# ENTRY POINT
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Algorithmic XAUUSD bot: stochastic/%R zone-breakout, PDH/PDL + "
            "Asia high/low reversal, and OB-mitigation entries, gated by "
            "1h/4h and 1m HalfTrend direction -- purely rule-based, no "
            "learned model."
        )
    )

    parser.add_argument("--train", action="store_true", help="Backtest against historical data.")
    parser.add_argument("--test", action="store_true", help="Trade live via MT5.")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument(
        "--risk", type=float, default=0.01,
        help="Flat fraction of account balance risked per trade (default 0.01 = 1%%)."
    )
    parser.add_argument(
        "--htf-mode", choices=["any", "both"], default="any",
        help=(
            "'any' (default): a breakout only needs 1h OR 4h HalfTrend "
            "to agree with its direction. 'both': needs both."
        )
    )

    args = parser.parse_args()

    # Separate OS processes, not threads -- see bot.py's main() for why
    # (both train_bot/test_bot are long-running CPU loops that would
    # starve each other under one GIL as threads).
    procs = []

    if args.train:
        p = multiprocessing.Process(
            target=train_bot,
            kwargs=dict(symbol=args.symbol, risk=args.risk, htf_mode=args.htf_mode),
            daemon=True
        )
        p.start()
        procs.append(p)

    if args.test:
        p = multiprocessing.Process(
            target=test_bot,
            kwargs=dict(symbol=args.symbol, risk=args.risk, htf_mode=args.htf_mode),
            daemon=True
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
