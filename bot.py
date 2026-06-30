import pandas as pd
import numpy as np
import os
import pickle
from io import StringIO
import random
from collections import deque
from datetime import datetime, timedelta
import subprocess
import time
import argparse
import threading
import math
import glob
import MetaTrader5 as mt5
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
"""
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    FeedForwardConfig,
)
"""
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
)

ACTIONS = ['hold', 'long', 'short']

def find_latest_dukascopy_csv():
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "download")

    pattern = os.path.join(folder, "xauusd-m5-bid-*.csv")

    # print(pattern)
    # print(glob.glob(pattern))
    # print(os.listdir(folder))

    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(f"No CSV found matching {pattern}")

    return max(files, key=os.path.getmtime)

def load_last_mb_xauusd(file_path=None, mb=3, delimiter=',', col_names=None):
    if file_path is None:
        file_path = find_latest_dukascopy_csv()

    print(f"Loading: {file_path}")

    file_size = os.path.getsize(file_path)
    offset = max(file_size - mb * 1024 * 1024, 0)  # start position
    
    with open(file_path, 'rb') as f:
        # Seek to approximately 20 MB before EOF
        f.seek(offset)
        
        # Read to the end of file from that offset
        data = f.read().decode(errors='ignore')
        
        # If not at start of file, discard partial first line (incomplete)
        if offset > 0:
            data = data.split('\n', 1)[-1]
    
    """
    df = pd.read_csv(StringIO(data), delimiter=delimiter, header=None, engine='python')
    
    #if col_names:
    print(df.head())
    print(df.columns)
    print(df.shape)

    df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    """

    df = pd.read_csv(
        StringIO(data),
        delimiter=delimiter,
        header=0,
        engine="python"
    )

    """
    df.rename(columns={
        "timestamp": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    }, inplace=True)
    """

    df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]

    # Convert columns if needed, e.g.:
    # df["Date"] = pd.to_datetime(df["Date"], format="%Y.%m.%d %H:%M", errors='coerce')
    df["Date"] = pd.to_datetime(
        df["Date"],
        unit="ms",
        utc=True
    )
    # print(df.columns.tolist())
    # print(df.head())
    # for col in ["Open", "High", "Low", "Close", "Volume"]:
    #     df[col] = pd.to_numeric(df[col], errors='coerce')
    # df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

    # df = df.resample('15min').agg({
    #     'Open': 'first',
    #     'High': 'max',
    #     'Low': 'min',
    #     'Close': 'last'
    # }).dropna()

    print(f"Loaded: {file_path}")
    
    df = df.dropna()
    
    return df

def ADX(df, period=14):
    """
    Returns +DI, -DI and ADX using Wilder's smoothing.
    Columns required: High, Low, Close
    """
    high  = df['High']
    low   = df['Low']
    close = df['Close']

    # --- directional movement -----------------------------------------
    # plus_dm  = (high.diff()  > low.diff())  * (high.diff()).clip(lower=0)
    # minus_dm = (low.diff()   > high.diff()) * (low.diff().abs()).clip(lower=0)̈́
    up  =  high.diff()
    dn  = -low.diff()

    plus_dm_array  = np.where((up  >  dn) & (up  > 0),  up,  0.0)
    minus_dm_array = np.where((dn  >  up) & (dn  > 0),  dn,  0.0)

    plus_dm = pd.Series(plus_dm_array, index=df.index) # ← wrap
    minus_dm = pd.Series(minus_dm_array, index=df.index) # ← wrap

    # --- true range ----------------------------------------------------
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)

    # --- Wilder smoothing ---------------------------------------------
    atr       = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di   = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di  = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)

    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    adx = round(adx , 2)
    plus_di = round(plus_di, 2)
    minus_di = round(minus_di, 2)

    return adx, plus_di, minus_di

def STOCH(df, period=14, smooth_d=3):
    """
    Returns %K and %D stochastic oscillator.

    Columns required:
    High, Low, Close
    """

    high  = df['High']
    low   = df['Low']
    close = df['Close']

    # --- highest high / lowest low ------------------------------------
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()

    # --- %K ------------------------------------------------------------
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))

    # --- %D (smoothed %K) ---------------------------------------------
    d = k.rolling(window=smooth_d).mean()

    k = round(k, 2)
    d = round(d, 2)

    return k, d

def EMA(df, period):
    return df['Close'].ewm(span=period, adjust=False).mean().round(2)

def EQH_EQL(df, threshold=10, lookahead=36, min_distance=3):

    open_ = df["Open"].to_numpy()
    close = df["Close"].to_numpy()
    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()

    body_high = np.maximum(open_, close)
    body_low = np.minimum(open_, close)

    eqh = np.zeros(len(df), dtype=np.bool_)
    eql = np.zeros(len(df), dtype=np.bool_)

    n = len(df)

    for i in range(n):

        end = min(i + lookahead + 1, n)

        for j in range(i + min_distance, end):

            # Equal High
            if (
                abs(high[i] - high[j]) <= threshold or
                abs(high[i] - body_high[j]) <= threshold or
                abs(body_high[i] - high[j]) <= threshold or
                abs(body_high[i] - body_high[j]) <= threshold
            ):
                eqh[i] = True

            # Equal Low
            if (
                abs(low[i] - low[j]) <= threshold or
                abs(low[i] - body_low[j]) <= threshold or
                abs(body_low[i] - low[j]) <= threshold or
                abs(body_low[i] - body_low[j]) <= threshold
            ):
                eql[i] = True

            if eqh[i] and eql[i]:
                break

    return eqh, eql

def Indecision(df, threshold=0.2):
    body = (df["Close"] - df["Open"]).abs()
    candle_range = (df["High"] - df["Low"]).replace(0, 1e-9)

    return (body / candle_range < threshold).astype(int)

def RejectionBlocks(df, wick_ratio=2.0):
    body = (df["Close"] - df["Open"]).abs()

    upper = df["High"] - df[["Open", "Close"]].max(axis=1)
    lower = df[["Open", "Close"]].min(axis=1) - df["Low"]

    bullish_rb = (
        (lower > body * wick_ratio) &
        (lower > upper)
    ).astype(int)

    bearish_rb = (
        (upper > body * wick_ratio) &
        (upper > lower)
    ).astype(int)

    return bullish_rb, bearish_rb

def BullishOB(df, multiplier=1.5):
    body = (df["Close"] - df["Open"]).abs()
    next_body = body.shift(-1)

    bearish = df["Close"] < df["Open"]
    next_bullish = df["Close"].shift(-1) > df["Open"].shift(-1)

    return (bearish &
            next_bullish &
            (next_body >= body * multiplier)).astype(int)

def BearishOB(df, multiplier=1.5):
    body = (df["Close"] - df["Open"]).abs()
    next_body = body.shift(-1)

    bullish = df["Close"] > df["Open"]
    next_bearish = df["Close"].shift(-1) < df["Open"].shift(-1)

    return (bullish &
            next_bearish &
            (next_body >= body * multiplier)).astype(int)

def BullishFVG(df):
    return (df["Low"].shift(-1) > df["High"].shift(1)).astype(int)

def BearishFVG(df):
    return (df["High"].shift(-1) < df["Low"].shift(1)).astype(int)

def BullishIFVG(df):
    return (
        (df["bearish_fvg"].shift(1) == 1) &
        (df["Close"] > df["Low"].shift(2))
    ).astype(int)

def BearishIFVG(df):
    return (
        (df["bullish_fvg"].shift(1) == 1) &
        (df["Close"] < df["High"].shift(2))
    ).astype(int)

def BullishMB(df, multiplier=1.0):
    body = (df["Close"] - df["Open"]).abs()

    bearish = df["Close"] < df["Open"]
    next_bullish = df["Close"].shift(-1) > df["Open"].shift(-1)
    displacement = body.shift(-1) >= body * multiplier

    ob = bearish & next_bullish & displacement

    ob_high = df["High"].where(ob).ffill()
    ob_low = df["Low"].where(ob).ffill()

    return ((df["Low"] <= ob_high) &
            (df["High"] >= ob_low)).astype(int)

def BearishMB(df, multiplier=1.0):
    body = (df["Close"] - df["Open"]).abs()

    bullish = df["Close"] > df["Open"]
    next_bearish = df["Close"].shift(-1) < df["Open"].shift(-1)
    displacement = body.shift(-1) >= body * multiplier

    ob = bullish & next_bearish & displacement

    ob_high = df["High"].where(ob).ffill()
    ob_low = df["Low"].where(ob).ffill()

    return ((df["Low"] <= ob_high) &
            (df["High"] >= ob_low)).astype(int)

def Swings(df, left=2, right=2):

    body_high = np.maximum(df["Open"].to_numpy(), df["Close"].to_numpy())
    body_low = np.minimum(df["Open"].to_numpy(), df["Close"].to_numpy())

    swing_high = [0] * len(df)
    swing_low = [0] * len(df)

    for i in range(left, len(df) - right):

        # Swing High (body only)
        if (
            body_high[i] > body_high[i-1] and
            body_high[i] > body_high[i-2] and
            body_high[i] > body_high[i+1] and
            body_high[i] > body_high[i+2]
        ):
            swing_high[i] = 1

        # Swing Low (body only)
        if (
            body_low[i] < body_low[i-1] and
            body_low[i] < body_low[i-2] and
            body_low[i] < body_low[i+1] and
            body_low[i] < body_low[i+2]
        ):
            swing_low[i] = 1

    return swing_high, swing_low

def MSS(df):

    bullish = pd.Series(0, index=df.index)
    bearish = pd.Series(0, index=df.index)

    last_swing_high = None
    last_swing_low = None

    for i in range(len(df)):

        if df["swing_high"].iloc[i]:
            last_swing_high = df["High"].iloc[i]

        if df["swing_low"].iloc[i]:
            last_swing_low = df["Low"].iloc[i]

        if (
            last_swing_high is not None
            and df["Close"].iloc[i] > last_swing_high
        ):
            bullish.iloc[i] = 1
            last_swing_high = None

        if (
            last_swing_low is not None
            and df["Close"].iloc[i] < last_swing_low
        ):
            bearish.iloc[i] = 1
            last_swing_low = None

    return bullish, bearish

def BullishBB(df, lookahead=36):

    bb = [0] * len(df)

    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()
    close = df["Close"].to_numpy()
    bearish_ob = df["bearish_ob"].to_numpy()

    n = len(df)

    for i in range(n):

        if bearish_ob[i] != 1:
            continue

        ob_high = high[i]
        ob_low = low[i]

        broken = False

        for j in range(i + 1, min(i + lookahead + 1, n)):

            # Break above the bearish OB
            if not broken:

                if close[j] > ob_high:
                    broken = True

            else:

                # Retest the breaker
                if (
                    low[j] <= ob_high and
                    high[j] >= ob_low
                ):
                    bb[j] = 1
                    break

    return bb

def BearishBB(df, lookahead=36):

    bb = [0] * len(df)

    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()
    close = df["Close"].to_numpy()
    bullish_ob = df["bullish_ob"].to_numpy()

    n = len(df)

    for i in range(n):

        if bullish_ob[i] != 1:
            continue

        ob_high = high[i]
        ob_low = low[i]

        broken = False

        for j in range(i + 1, min(i + lookahead + 1, n)):

            if not broken:

                if close[j] < ob_low:
                    broken = True

            else:

                if (
                    low[j] <= ob_high and
                    high[j] >= ob_low
                ):
                    bb[j] = 1
                    break

    return bb

def ExhaustionCandle(df):
    open_ = df["Open"].to_numpy()
    close = df["Close"].to_numpy()

    body = np.abs(close - open_)

    exhaustion = np.zeros(len(df), dtype=np.bool_)

    exhaustion[1:] = body[1:] <= body[:-1] * 0.10

    return exhaustion

def OBMitigation(df, threshold=30, lookahead=36):

    low = df["Low"].to_numpy()
    high = df["High"].to_numpy()

    bullish_ob = df["bullish_ob"].to_numpy()
    bearish_ob = df["bearish_ob"].to_numpy()

    bull = np.zeros(len(df), dtype=np.bool_)
    bear = np.zeros(len(df), dtype=np.bool_)

    n = len(df)

    for i in range(n):

        if not np.isnan(bullish_ob[i]):

            ob = bullish_ob[i]

            for j in range(i + 1, min(i + lookahead + 1, n)):

                if abs(low[j] - ob) <= threshold:
                    bull[j] = True
                    break

        if not np.isnan(bearish_ob[i]):

            ob = bearish_ob[i]

            for j in range(i + 1, min(i + lookahead + 1, n)):

                if abs(high[j] - ob) <= threshold:
                    bear[j] = True
                    break

    return bull, bear

def FVGRetracement(df):

    open_ = df["Open"].to_numpy()
    close = df["Close"].to_numpy()

    body_high = np.maximum(open_, close)
    body_low = np.minimum(open_, close)

    bull_fvg = df["bullish_fvg"].to_numpy()
    bear_fvg = df["bearish_fvg"].to_numpy()

    bull_ret = np.zeros(len(df), dtype=np.bool_)
    bear_ret = np.zeros(len(df), dtype=np.bool_)

    n = len(df)

    # Bullish FVG
    for i in range(1, n - 1):

        if bull_fvg[i] == 0:
            continue

        lower = df["High"].iat[i - 1]
        upper = df["Low"].iat[i + 1]

        # Start from the upper FVG candle (i+1)
        for j in range(i + 2, n):

            # Stop when bullish candle appears
            if close[j] >= open_[j]:
                break

            # Body overlaps FVG
            if (
                body_high[j] >= lower and
                body_low[j] <= upper
            ):
                bull_ret[j] = True
                break

    # Bearish FVG
    for i in range(1, n - 1):

        if bear_fvg[i] == 0:
            continue

        upper = df["Low"].iat[i - 1]
        lower = df["High"].iat[i + 1]

        for j in range(i + 2, n):

            # Stop when bearish candle appears
            if close[j] <= open_[j]:
                break

            if (
                body_high[j] >= lower and
                body_low[j] <= upper
            ):
                bear_ret[j] = True
                break

    return bull_ret, bear_ret

def EQHEQLRetest(df, threshold=20, lookahead=12):

    open_ = df["Open"].to_numpy()
    close = df["Close"].to_numpy()
    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()

    body_high = np.maximum(open_, close)
    body_low = np.minimum(open_, close)

    eqh = df["eqh"].to_numpy()
    eql = df["eql"].to_numpy()

    eqh_retest = np.zeros(len(df), dtype=np.bool_)
    eql_retest = np.zeros(len(df), dtype=np.bool_)

    n = len(df)

    for i in range(n):

        if eqh[i]:

            level = max(high[i], body_high[i])

            end = min(i + lookahead + 1, n)

            for j in range(i + 1, end):

                if (
                    abs(high[j] - level) <= threshold or
                    abs(body_high[j] - level) <= threshold and
                    df["bullish"].iloc[j]
                ):
                    eqh_retest[j] = True
                    break

        if eql[i]:

            level = min(low[i], body_low[i])

            end = min(i + lookahead + 1, n)

            for j in range(i + 1, end):

                if (
                    abs(low[j] - level) <= threshold or
                    abs(body_low[j] - level) <= threshold and
                    df["bearish"].iloc[j]
                ):
                    eql_retest[j] = True
                    break

    return eqh_retest, eql_retest

def TrendLines(df):

    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()
    close = df["Close"].to_numpy()

    swing_high = df["swing_high"].to_numpy()
    swing_low = df["swing_low"].to_numpy()

    above_support = np.zeros(len(df), dtype=np.bool_)
    below_resistance = np.zeros(len(df), dtype=np.bool_)

    n = len(df)

    # =====================================================
    # Support
    # =====================================================

    lows = np.where(swing_low)[0]

    for k in range(1, len(lows)):

        i1 = lows[k - 1]
        i2 = lows[k]

        # Only higher lows
        if low[i2] <= low[i1]:
            continue

        slope = (low[i2] - low[i1]) / (i2 - i1)

        end = lows[k + 1] if k + 1 < len(lows) else n

        for j in range(i2 + 1, end):

            trend_price = low[i2] + slope * (j - i2)

            if close[j] >= trend_price:
                above_support[j] = True

    # =====================================================
    # Resistance
    # =====================================================

    highs = np.where(swing_high)[0]

    for k in range(1, len(highs)):

        i1 = highs[k - 1]
        i2 = highs[k]

        # Only lower highs
        if high[i2] >= high[i1]:
            continue

        slope = (high[i2] - high[i1]) / (i2 - i1)

        end = highs[k + 1] if k + 1 < len(highs) else n

        for j in range(i2 + 1, end):

            trend_price = high[i2] + slope * (j - i2)

            if close[j] <= trend_price:
                below_resistance[j] = True

    return above_support, below_resistance

def Fibonacci(df, min_swing=7.5, threshold=2):

    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()
    open_ = df["Open"].to_numpy()
    close = df["Close"].to_numpy()

    body_high = np.maximum(open_, close)
    body_low = np.minimum(open_, close)

    swing_high = df["swing_high"].to_numpy()
    swing_low = df["swing_low"].to_numpy()

    bullish_fib382 = np.zeros(len(df), dtype=np.bool_)
    bullish_fib500 = np.zeros(len(df), dtype=np.bool_)
    bullish_fib618 = np.zeros(len(df), dtype=np.bool_)
    bullish_fib786 = np.zeros(len(df), dtype=np.bool_)

    bearish_fib382 = np.zeros(len(df), dtype=np.bool_)
    bearish_fib500 = np.zeros(len(df), dtype=np.bool_)
    bearish_fib618 = np.zeros(len(df), dtype=np.bool_)
    bearish_fib786 = np.zeros(len(df), dtype=np.bool_)

    last_type = None
    last_index = None

    n = len(df)

    for i in range(n):

        # ==========================================================
        # Bullish impulse (Swing Low -> Swing High)
        # ==========================================================

        if swing_high[i] and last_type == "low":

            swing = high[i] - low[last_index]

            if swing >= min_swing:

                fib382 = high[i] - swing * 0.382
                fib500 = high[i] - swing * 0.500
                fib618 = high[i] - swing * 0.618
                fib786 = high[i] - swing * 0.786
                threshold = swing * 0.1

                j = i

                while (
                    j < n and
                    not swing_high[j] and
                    not swing_low[j]
                ):

                    bullish_fib382[j] = (
                        body_low[j] <= fib382 + threshold and
                        body_high[j] >= fib382 - threshold
                    )

                    bullish_fib500[j] = (
                        body_low[j] <= fib500 + threshold and
                        body_high[j] >= fib500 - threshold
                    )

                    bullish_fib618[j] = (
                        body_low[j] <= fib618 + threshold and
                        body_high[j] >= fib618 - threshold
                    )

                    bullish_fib786[j] = (
                        body_low[j] <= fib786 + threshold and
                        body_high[j] >= fib786 - threshold
                    )

                    j += 1

            last_type = "high"
            last_index = i

        # ==========================================================
        # Bearish impulse (Swing High -> Swing Low)
        # ==========================================================

        elif swing_low[i] and last_type == "high":

            swing = high[last_index] - low[i]

            if swing >= min_swing:

                fib382 = low[i] + swing * 0.382
                fib500 = low[i] + swing * 0.500
                fib618 = low[i] + swing * 0.618
                fib786 = low[i] + swing * 0.786

                j = i

                while (
                    j < n and
                    not swing_high[j] and
                    not swing_low[j]
                ):

                    bearish_fib382[j] = (
                        body_low[j] <= fib382 + threshold and
                        body_high[j] >= fib382 - threshold
                    )

                    bearish_fib500[j] = (
                        body_low[j] <= fib500 + threshold and
                        body_high[j] >= fib500 - threshold
                    )

                    bearish_fib618[j] = (
                        body_low[j] <= fib618 + threshold and
                        body_high[j] >= fib618 - threshold
                    )

                    bearish_fib786[j] = (
                        body_low[j] <= fib786 + threshold and
                        body_high[j] >= fib786 - threshold
                    )

                    j += 1

            last_type = "low"
            last_index = i

        elif swing_low[i]:
            last_type = "low"
            last_index = i

        elif swing_high[i]:
            last_type = "high"
            last_index = i

    return (
        bullish_fib382,
        bullish_fib500,
        bullish_fib618,
        bullish_fib786,
        bearish_fib382,
        bearish_fib500,
        bearish_fib618,
        bearish_fib786,
    )

def AsiaHighDistance(df):
    # Asia session: 01:00-08:59 (your timezone)
    asia = (df.index.hour >= 1) | (df.index.hour <= 9)

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
    asia = (df.index.hour >= 1) | (df.index.hour <= 9)

    trade_day = (df.index - pd.Timedelta(hours=24)).date

    asia_low = (
        df["Low"]
        .where(asia)
        .groupby(trade_day)
        .transform("min")
        .ffill()
    )

    return df["Close"] - asia_low

def PDHDistance(df):
    day = df.index.date

    daily_high = (
        df["High"]
        .groupby(day)
        .transform("max")
    )

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
    day = df.index.date

    daily_low = (
        df["Low"]
        .groupby(day)
        .transform("min")
    )

    pdl = (
        daily_low
        .groupby(day)
        .first()
        .shift(1)
        .reindex(day)
        .to_numpy()
    )

    return df["Close"] - pdl

def PDPOCDistance(df, bins=50):
    day = df.index.date
    poc = np.full(len(df), np.nan)

    unique_days = np.unique(day)

    prev_poc = np.nan

    for d in unique_days:
        # Assign previous day's POC to today's candles
        mask = day == d
        poc[mask] = prev_poc

        # Compute today's POC for use tomorrow
        today = df.loc[mask]

        if len(today) > 1:
            prices = ((today["High"] + today["Low"] + today["Close"]) / 3).values
            volumes = today["Volume"].values

            hist, edges = np.histogram(
                prices,
                bins=bins,
                weights=volumes
            )

            idx = np.argmax(hist)
            prev_poc = (edges[idx] + edges[idx + 1]) / 2

    return df["Close"] - poc

def VWAP(df):

    # print(type(df.index))
    # print(df.index.dtype)
    # print(df.index[:5])

    # --------------------------------------------------
    # Select volume column
    # --------------------------------------------------
    if "Volume" in df.columns:
        volume = df["Volume"]
    elif "tick_volume" in df.columns:
        volume = df["tick_volume"]
    elif "real_volume" in df.columns:
        volume = df["real_volume"]
    else:
        raise ValueError("No volume column found.")

    # --------------------------------------------------
    # VWAP
    # --------------------------------------------------
    typical_price = (
        df["High"] +
        df["Low"] +
        df["Close"]
    ) / 3

    # print(df.columns)
    # df["Date"] = (
    #     pd.to_datetime(df["Date"], unit="ms", utc=True)
    #     .dt.tz_convert("Europe/Denmark")
    #     .dt.tz_localize(None)
    # )
    # index = df.index.tz_convert("Europe/Copenhagen")
    # print(f"vwap time: {df.index[-1]}")

    session = df.index.normalize()
    # session = (df.index - pd.Timedelta(hours=1)).normalize()

    cum_tpv = (typical_price * volume).groupby(session).cumsum()
    cum_volume = volume.groupby(session).cumsum()

    vwap = round(cum_tpv / cum_volume, 2)

    # --------------------------------------------------
    # Session VWAP Standard Deviation
    # --------------------------------------------------

    # Squared distance from VWAP
    sq_diff = ((typical_price - vwap) ** 2) * volume

    # Cumulative weighted variance
    cum_sq_diff = sq_diff.groupby(session).cumsum()

    variance = cum_sq_diff / cum_volume
    stddev = variance.pow(0.5)

    upper = round(vwap + stddev, 2)
    lower = round(vwap - stddev, 2)

    # --------------------------------------------------
    # Derived features
    # --------------------------------------------------
    # dist = df["Close"] - vwap

    above = (df["Close"] > vwap).astype(int)
    below = (df["Close"] < vwap).astype(int)

    above_upper = (df["Close"] > upper).astype(int)
    below_lower = (df["Close"] < lower).astype(int)

    slope = vwap.diff()

    return (
        vwap,
        upper,
        lower,
        # dist,
        above,
        below,
        above_upper,
        below_lower,
        slope
    )

def VolumeMA(df, period=14):

    # -----------------------------------------
    # Select volume column
    # -----------------------------------------
    if "Volume" in df.columns:
        volume = df["Volume"]
    elif "tick_volume" in df.columns:
        volume = df["tick_volume"]
    elif "volume" in df.columns:
        volume = df["volume"]
    else:
        raise ValueError("No volume column found.")

    # print(type(volume))
    # print(volume.shape)
    # print(df.columns.tolist())

    return round(volume.rolling(period).mean(), 2)

def BullishBearish(df):
    """
    Returns:
        bullish (np.ndarray[bool])
        bearish (np.ndarray[bool])
    """

    open = df["Open"].to_numpy()
    close = df["Close"].to_numpy()

    bullish = close > open
    bearish = close < open

    return bullish, bearish

def add_irl_erl(df, swing_window=5):
    """
    Adds:
        range_high
        range_low
        erl_high
        erl_low
        irl

    Assumes df contains:
        High
        Low
        Close
    """

    # df = df.copy()

    # ----------------------------
    # Current dealing range
    # ----------------------------
    df["range_high"] = np.where(df["swing_high"], df["High"], np.nan)
    df["range_low"] = np.where(df["swing_low"], df["Low"], np.nan)

    df["range_high"] = df["range_high"].ffill()
    df["range_low"] = df["range_low"].ffill()

    # ----------------------------
    # External liquidity
    # ----------------------------
    df["erl_high"] = df["High"] > df["range_high"]
    df["erl_low"] = df["Low"] < df["range_low"]

    # ----------------------------
    # Internal liquidity
    # ----------------------------
    df["irl"] = (
        (df["High"] <= df["range_high"])
        & (df["Low"] >= df["range_low"])
    )

    return df

def BuyScore(df):

    return (
        (df["EMA7"] > df["EMA21"]).astype(int) * 1 +
        (df["EMA721_DIFF"] > 0).astype(int) * 1 +
        (df["EMA7_Slope"] > 0).astype(int) * 1 +
        (df["+di"] > df["-di"]).astype(int) * 1 +
        (df["adx"] > 20).astype(int) * 1 +
        (df["k"] > df["k_smooth"]).astype(int) * 1 +
        (df["k"] < 20).astype(int) * 1 +

        df["bullish_ob"] * 1 +
        df["bullish_ob_mitigation"] * 1 +

        df["bullish_mb"] * 1 +

        df["bullish_fvg"] * 1 +
        df["bullish_ifvg"] * 1 -
        df["bearish_fvg"] * 1 -
        df["bearish_ifvg"] * 1 -

        df["eqh"] * 1 -
        df["eql_retest"] * 1 +

        df["bullish_rb"] * 1 +
        df["bullish_mss"] * 1 +
        df["bullish_bb"] * 1 +

        # df["bullish_fib382"] * 1 +
        # df["bullish_fib500"] * 1 +
        # df["bullish_fib618"] * 1 +
        df["bullish_fib786"] * 1 -

        # df["bearish_fib382"] * 1 -
        # df["bearish_fib500"] * 1 -
        # df["bearish_fib618"] * 1 -
        df["bearish_fib786"] * 1 -

        (df["above_support"] == 0).astype(int) +
        (df["below_resistance"] == 0).astype(int) * 1 +

        df["above_vwap"] * 1 +
        df["vwap_above_upper"] * 1 -

        df["below_vwap"] * 1 -
        df["vwap_below_lower"] * 1 -
        df["bearish_rb"] * 1 -
        df["bearish_ob"] * 1 -
        df["bearish_ob_mitigation"] * 1 -
        df["bearish_fvg"] * 1 +
        df["eql"] * 1 +
        df["eqh_retest"] * 1 -
        df["bearish_mss"] * 1 -
        df["bearish_bb"] * 1 -
        (df["k"] >= 80) * 1
    )

def SellScore(df):

    return (
        (df["EMA7"] < df["EMA21"]).astype(int) * 1 +
        (df["EMA721_DIFF"] < 0).astype(int) * 1 +
        (df["EMA7_Slope"] < 0).astype(int) * 1 +
        (df["-di"] > df["+di"]).astype(int) * 1 +
        (df["adx"] > 20).astype(int) * 1 +
        (df["k"] < df["k_smooth"]).astype(int) * 1 +
        (df["k"] > 80).astype(int) * 1 +

        df["bearish_ob"] * 1 +
        df["bearish_ob_mitigation"] * 1 +

        df["bearish_mb"] * 1 +

        df["bearish_fvg"] * 1 +
        df["bearish_ifvg"] * 1 -
        df["bullish_fvg"] * 1 -
        df["bullish_ifvg"] * 1 -

        df["eql"] * 1 -
        df["eqh_retest"] * 1 +

        df["bearish_mss"] * 1 +
        df["bearish_bb"] * 1 +

        # df["bearish_fib382"] * 1 +
        # df["bearish_fib500"] * 1 +
        # df["bearish_fib618"] * 1 +
        df["bearish_fib786"] * 1 -

        # df["bullish_fib382"] * 1 -
        # df["bullish_fib500"] * 1 -
        # df["bullish_fib618"] * 1 -
        df["bullish_fib786"] * 1 +

        (df["above_support"] == 0).astype(int) -
        (df["below_resistance"] == 0).astype(int) * 1 +

        df["below_vwap"] * 1 +
        df["vwap_below_lower"] * 1 -

        df["above_vwap"] * 1 -
        df["vwap_above_upper"] * 1 -
        df["bearish_rb"] * 1 -
        df["bullish_rb"] * 1 -
        df["bullish_ob"] * 1 -
        df["bullish_ob_mitigation"] * 1 -
        df["bullish_fvg"] * 1 +
        df["eqh"] * 1 +
        df["eql_retest"] * 1 -
        df["bullish_mss"] * 1 -
        df["bullish_bb"] * 1 -
        (df["k"] <= 20) * 1
    )

def add_indicators(df):
    # print("Loading indicators")
    df['adx'], df['+di'], df['-di'] = ADX(df)

    df['k'], df['k_smooth'] = STOCH(df)

    df['EMA7'] = EMA(df, 7)
    # df['EMA1'] = EMA(df, 1)
    # df['EMA1_Slope'] = df['EMA1'].diff()
    df['EMA7_Slope'] = df['EMA7'].diff()
    df['EMA21'] = EMA(df, 21)
    df['EMA21_Slope'] = df['EMA21'].diff()
    df['EMA721_DIFF'] = df['EMA7'] - df['EMA21']
    # df['EMA1'] = EMA(df, 1)
    # df['EMA71_DIFF'] = df['EMA1'] - df['EMA7']

    df["indecision"] = Indecision(df)

    df["bullish_ob"] = BullishOB(df)
    df["bearish_ob"] = BearishOB(df)
    df["bullish_ob_mitigation"], df["bearish_ob_mitigation"] = OBMitigation(df)

    df["bullish_fvg"] = BullishFVG(df)
    df["bearish_fvg"] = BearishFVG(df)
    df["bullish_fvg_retracement"], df["bearish_fvg_retracement"] = FVGRetracement(df)
    df["bullish_ifvg"] = BullishIFVG(df)
    df["bearish_ifvg"] = BearishIFVG(df)

    df["bullish"], df["bearish"] = BullishBearish(df)

    df["eqh"], df["eql"] = EQH_EQL(df)
    df["eqh_retest"], df["eql_retest"] = EQHEQLRetest(df)

    df["bearish_mb"] = BearishMB(df)
    df["bullish_mb"] = BullishMB(df)

    df["bullish_rb"], df["bearish_rb"] = RejectionBlocks(df)

    df["swing_high"], df["swing_low"] = Swings(df)
    df["bullish_mss"], df["bearish_mss"] = MSS(df)

    df["bullish_bb"] = BullishBB(df)
    df["bearish_bb"] = BearishBB(df)

    df["exhaustion"] = ExhaustionCandle(df)

    df["above_support"], df["below_resistance"] = TrendLines(df)

    df["bullish_fib382"], df["bullish_fib500"], df["bullish_fib618"], df["bullish_fib786"], df["bearish_fib382"], df["bearish_fib500"], df["bearish_fib618"], df["bearish_fib786"] = Fibonacci(df)

    df["vwap"], df["vwap_upper"], df["vwap_lower"], df["above_vwap"], df["below_vwap"], df["vwap_above_upper"], df["vwap_below_lower"], df["vwap_slope"] = VWAP(df)

    df = add_irl_erl(df)

    df["sell_score"] = SellScore(df)
    df["buy_score"] = BuyScore(df)

    """
    if "Volume" in df.columns:
        df.rename(columns={
            'Volume': 'volume',
        }, inplace=True)
    elif "tick_volume" in df.columns:
        df.rename(columns={
            'tick_volume': 'volume',
        }, inplace=True)
    """
    df["volume_ma"] = VolumeMA(df)

    df["asia_high_dist"] = AsiaHighDistance(df)
    df["asia_low_dist"] = AsiaLowDistance(df)

    df["pdh_dist"] = PDHDistance(df)
    df["pdl_dist"] = PDLDistance(df)

    df["pd_poc_dist"] = PDPOCDistance(df)

    df = df[["Open", "High", "Low", "Close",
            "k", "k_smooth", "adx", "+di", "-di", "EMA7", "EMA21", "EMA721_DIFF", "EMA7_Slope", "EMA21_Slope",
            "indecision", "bullish_ob", "bearish_ob", "bullish_fvg", "bearish_fvg", "bullish_ifvg", "bearish_ifvg", "eqh", "eql", "bearish_mb", "bullish_mb","bullish_rb", "bearish_rb", "bullish_bb", "bearish_bb",
            "bullish_ob_mitigation", "bearish_ob_mitigation", "bullish_fvg_retracement", "bearish_fvg_retracement", "eqh_retest", "eql_retest",
            "swing_high", "swing_low", "bullish_mss", "bearish_mss",
            "range_high", "range_low", "erl_high", "erl_low", "irl",
            "exhaustion",
            "above_support", "below_resistance",
            "bullish_fib382", "bullish_fib500", "bullish_fib618", "bullish_fib786", "bearish_fib382", "bearish_fib500", "bearish_fib618", "bearish_fib786",
            "vwap", "vwap_upper", "vwap_lower", "above_vwap", "below_vwap", "vwap_above_upper", "vwap_below_lower", "vwap_slope",
            "volume_ma", "Volume",
            "bullish", "bearish",
            "asia_high_dist", "asia_low_dist",
            "pdh_dist", "pdl_dist",
            "pd_poc_dist",
            "sell_score", "buy_score",
            ]].copy()

    # print("Loaded indicators")

    df.dropna(inplace=True)
    return df

class PPOxLSTMNetwork(nn.Module):
    def __init__(
        self,
        state_size,
        hidden_size=128,
        action_size=3,
        seq_len=32,
    ):
        super().__init__()

        self.embedding = nn.Linear(state_size, hidden_size)

        cfg = xLSTMBlockStackConfig(
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="vanilla",
                    num_heads=4,
                )
            ),

            slstm_at=[0, 1],

            context_length=seq_len,
            num_blocks=2,
            embedding_dim=hidden_size,
        )

        self.backbone = xLSTMBlockStack(cfg)

        # Buy / Sell / Hold
        self.policy = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

        # RR multiplier
        self.rr_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # SL multiplier
        self.sl_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # PPO critic
        self.value = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):

        # (batch, seq, features)
        x = self.embedding(x)

        x = self.backbone(x)

        # Last timestep
        h = x[:, -1]

        # Action logits
        logits = self.policy(h)

        # RR between 1 and 10
        rr_mult = 1.0 + torch.sigmoid(
            self.rr_head(h)
        ) * 9.0

        # SL multiplier between 0.75x and 2.0x
        sl_mult = 0.75 + torch.sigmoid(
            self.sl_head(h)
        ) * 1.25

        # PPO critic
        value = self.value(h).squeeze(-1)

        return logits, rr_mult, sl_mult, value

class xLSTMPPOAgent:
    def __init__(
        self,
        state_size,
        hidden_size,
        action_size,
        lr=3e-4,
        gamma=0.95,
        clip_ratio=0.2,
        gae_lambda=0.95
    ):
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.action_size = action_size

        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.gae_lambda = gae_lambda

        self.train_epochs = 10
        self.batch_size = 64
        self.entropy_coef = 0.01
        self.value_coef = 0.5

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = PPOxLSTMNetwork(
            state_size,
            hidden_size,
            action_size
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr
        )

        self.trajectory = []

    def _state_tensor(self, state_seq):
        return torch.tensor(
            state_seq,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

    def select_action(self, state_seq, in_position=False, training=False):

        state = self._state_tensor(state_seq)

        with torch.no_grad():
            logits, rr_mult, sl_mult, value = self.model(state)

        logits = logits.squeeze(0)
        rr_mult = rr_mult.squeeze()
        sl_mult = sl_mult.squeeze()

        valid_actions = [0] if in_position else [0,1,2]

        masked_logits = logits.clone()

        for i in range(self.action_size):
            if i not in valid_actions:
                masked_logits[i] = -1e9

        probs = torch.softmax(masked_logits, dim=-1)
        dist = Categorical(probs)

        action = dist.sample() if training else torch.argmax(probs)

        logprob = dist.log_prob(action)

        return (
            int(action.item()),
            float(logprob.item()),
            float(value.item()),
            float(rr_mult.item()),
            float(sl_mult.item())
        )

    def store_transition(
        self,
        state_seq,
        action,
        logprob,
        value,
        rr_mult,
        sl_mult,
        reward,
        done
    ):
        self.trajectory.append(
            (
                np.array(state_seq, dtype=np.float32),
                action,
                logprob,
                value,
                rr_mult,
                sl_mult,
                reward,
                done
            )
        )

    def compute_gae(self, rewards, values, dones):

        advantages = []
        gae = 0

        values = np.append(values, 0.0)

        for t in reversed(range(len(rewards))):

            delta = (
                rewards[t]
                + self.gamma * values[t + 1] * (1 - dones[t])
                - values[t]
            )

            gae = (
                delta
                + self.gamma
                * self.gae_lambda
                * (1 - dones[t])
                * gae
            )

            advantages.insert(0, gae)

        return np.array(advantages, dtype=np.float32)

    def train(self):

        if len(self.trajectory) < 32:
            return

        (
            states,
            actions,
            old_logprobs,
            values,
            rr_mults,
            sl_mults,
            rewards,
            dones
        ) = zip(*self.trajectory)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        old_logprobs = torch.tensor(old_logprobs, dtype=torch.float32, device=self.device)
        values = np.array(values, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        advantages = self.compute_gae(rewards, values, dones)
        returns = advantages + values

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)

        n = len(states)

        for _ in range(self.train_epochs):

            idx = torch.randperm(n, device=self.device)

            for start in range(0, n, self.batch_size):

                batch = idx[start:start+self.batch_size]

                logits, rr_pred, sl_pred, value_pred = self.model(states[batch])

                dist = Categorical(logits=logits)

                new_logprob = dist.log_prob(actions[batch])

                ratio = torch.exp(new_logprob - old_logprobs[batch])

                surr1 = ratio * advantages[batch]
                surr2 = torch.clamp(
                    ratio,
                    1-self.clip_ratio,
                    1+self.clip_ratio
                ) * advantages[batch]

                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(
                    value_pred,
                    returns[batch]
                )

                entropy = dist.entropy().mean()

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

        self.trajectory.clear()

    def savecheckpoint(self, symbol):
        os.makedirs("LSTM-PPO-saves", exist_ok=True)

        filename = (
            f"LSTM-PPO-saves/"
            f"{datetime.now().strftime('%Y-%m-%d')}-"
            f"{symbol}.checkpoint.pt"
        )

        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict()
            },
            filename
        )

    def loadcheckpoint(self, symbol):

        if not os.path.exists("LSTM-PPO-saves"):
            return

        files = [
            os.path.join("LSTM-PPO-saves", f)
            for f in os.listdir("LSTM-PPO-saves")
            if f.endswith(".checkpoint.pt")
            and symbol in f
        ]

        if not files:
            return

        latest = max(files, key=os.path.getmtime)

        checkpoint = torch.load(
            latest,
            map_location=self.device
        )

        self.model.load_state_dict(checkpoint["model"])

        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(
                checkpoint["optimizer"]
            )

        print(f"Loaded checkpoint: {latest}")

class PPOLSTMNetwork(nn.Module):
    def __init__(self, state_size=12, hidden_size=64, action_size=3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=state_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.policy = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

        self.value = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # self.debug = False

    def forward(self, x):
        """
        if self.debug:
            print("x shape before lstm:", x.shape)

        if x.shape[1] == 0:
            print("ERROR: zero sequence length")
            print("x shape:", x.shape)
            raise ValueError("Zero sequence length")
        """

        out, _ = self.lstm(x)
        h = out[:, -1, :]

        logits = self.policy(h)
        value = self.value(h).squeeze(-1)

        return logits, value

class LSTMPPOAgent:
    def __init__(
        self,
        state_size,
        hidden_size,
        action_size,
        lr=3e-4,
        gamma=0.95,
        clip_ratio=0.2,
        gae_lambda=0.95
    ):
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.action_size = action_size

        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.gae_lambda = gae_lambda

        self.train_epochs = 10
        self.batch_size = 64
        self.entropy_coef = 0.01
        self.value_coef = 0.5

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = PPOLSTMNetwork(
            state_size,
            hidden_size,
            action_size
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr
        )

        self.trajectory = []

    def _state_tensor(self, state_seq):
        return torch.tensor(
            state_seq,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

    def select_action(self, state_seq, in_position=False, training=False):

        state = self._state_tensor(state_seq)

        # if training is False:
        #     print("state type:", type(state))
        #     print("state shape:", state.shape if hasattr(state, "shape") else "no shape")

        with torch.no_grad():
            logits, value = self.model(state)

        logits = logits.squeeze(0)

        if in_position:
            valid_actions = [0]
        else:
            valid_actions = [0, 1, 2]

        masked_logits = logits.clone()

        for i in range(self.action_size):
            if i not in valid_actions:
                masked_logits[i] = -1e9

        probs = torch.softmax(masked_logits, dim=-1)

        dist = Categorical(probs)

        if training:
            action = dist.sample()
        else:
            action = torch.argmax(probs)

        logprob = dist.log_prob(action)

        """
        print(
            f"H={probs[0]:.2f} "
            f"B={probs[1]:.2f} "
            f"S={probs[2]:.2f}"
        )
        """

        return (
            int(action.item()),
            float(logprob.item()),
            float(value.item())
        )

    def store_transition(
        self,
        state_seq,
        action,
        logprob,
        value,
        reward,
        done
    ):
        self.trajectory.append(
            (
                np.array(state_seq, dtype=np.float32),
                action,
                logprob,
                value,
                reward,
                done
            )
        )

    def compute_gae(self, rewards, values, dones):

        advantages = []
        gae = 0

        values = np.append(values, 0.0)

        for t in reversed(range(len(rewards))):

            delta = (
                rewards[t]
                + self.gamma * values[t + 1] * (1 - dones[t])
                - values[t]
            )

            gae = (
                delta
                + self.gamma
                * self.gae_lambda
                * (1 - dones[t])
                * gae
            )

            advantages.insert(0, gae)

        return np.array(advantages, dtype=np.float32)

    def train(self):

        if len(self.trajectory) < 32:
            return

        states, actions, old_logprobs, values, rewards, dones = zip(
            *self.trajectory
        )

        states = np.array(states, dtype=np.float32)
        actions = np.array(actions)
        old_logprobs = np.array(old_logprobs, dtype=np.float32)
        values = np.array(values, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        advantages = self.compute_gae(
            rewards,
            values,
            dones
        )

        returns = advantages + values

        advantages = (
            advantages - advantages.mean()
        ) / (advantages.std() + 1e-8)

        states = torch.tensor(
            states,
            dtype=torch.float32,
            device=self.device
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long,
            device=self.device
        )

        old_logprobs = torch.tensor(
            old_logprobs,
            dtype=torch.float32,
            device=self.device
        )

        returns = torch.tensor(
            returns,
            dtype=torch.float32,
            device=self.device
        )

        advantages = torch.tensor(
            advantages,
            dtype=torch.float32,
            device=self.device
        )

        n = len(states)

        for _ in range(self.train_epochs):

            idx = torch.randperm(n, device=self.device)

            for start in range(0, n, self.batch_size):

                batch_idx = idx[start:start+self.batch_size]

                b_states = states[batch_idx]
                b_actions = actions[batch_idx]
                b_old_logprobs = old_logprobs[batch_idx]
                b_returns = returns[batch_idx]
                b_advantages = advantages[batch_idx]

                logits, values_pred = self.model(b_states)

                dist = Categorical(logits=logits)

                new_logprobs = dist.log_prob(
                    b_actions
                )

                entropy = dist.entropy().mean()

                ratio = torch.exp(
                    new_logprobs - b_old_logprobs
                )

                surr1 = ratio * b_advantages

                surr2 = torch.clamp(
                    ratio,
                    1 - self.clip_ratio,
                    1 + self.clip_ratio
                ) * b_advantages

                policy_loss = -torch.min(
                    surr1,
                    surr2
                ).mean()

                value_loss = F.mse_loss(
                    values_pred,
                    b_returns
                )

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    1.0
                )

                self.optimizer.step()

        self.trajectory.clear()

    def savecheckpoint(self, symbol):

        os.makedirs(
            "LSTM-PPO-saves",
            exist_ok=True
        )

        filename = (
            f"LSTM-PPO-saves/"
            f"{datetime.now().strftime('%Y-%m-%d')}-"
            f"{symbol}.checkpoint.pt"
        )

        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict()
            },
            filename
        )

    def loadcheckpoint(self, symbol):

        if not os.path.exists("LSTM-PPO-saves"):
            return

        files = [
            os.path.join("LSTM-PPO-saves", f)
            for f in os.listdir("LSTM-PPO-saves")
            if f.endswith(".checkpoint.pt")
            and symbol in f
        ]

        if not files:
            return

        latest = max(files, key=os.path.getmtime)

        checkpoint = torch.load(
            latest,
            map_location=self.device
        )

        self.model.load_state_dict(
            checkpoint["model"]
        )

        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(
                checkpoint["optimizer"]
            )

        print(f"Loaded checkpoint: {latest}")

class WinRateKNN:
    def __init__(self, symbol, k=10):
        self.k = k
        self.symbol = symbol
        self.states = []
        self.labels = []  # 1 = win, 0 = loss
        self.model = None

    def add(self, state, is_win):
        try:
            state = np.array(state, dtype=np.float32).flatten()  # Force all elements to float
        except Exception as e:
            # print("❌ Could not convert state to float:", state, "| Error:", e)
            return

        if not np.all(np.isfinite(state)):
            # print("⚠️ Skipping state with NaN or Inf:", state)
            return

        self.states.append(state)
        self.labels.append(1 if is_win else 0)

        if len(self.states) >= 100:
            self._remove_redundant_neighbor()
            # self.states.pop(0)
            # self.labels.pop(0)

        if len(self.states) >= self.k:
            self._fit()

    def _remove_redundant_neighbor(self):
        if len(self.states) < 2:
            return  # Nothing to remove

        X = np.array(self.states)

        # Compute pairwise similarity (cosine, or use euclidean if you prefer)
        sim_matrix = cosine_similarity(X)

        # Zero out diagonal (self-similarity)
        np.fill_diagonal(sim_matrix, 0)

        # Compute average similarity for each row (how redundant each entry is)
        redundancy_scores = sim_matrix.mean(axis=1)

        # Remove the most redundant (highest avg similarity)
        idx_to_remove = np.argmax(redundancy_scores)

        del self.states[idx_to_remove]
        del self.labels[idx_to_remove]
    def _fit(self):
        """
        Fit the KNN model with stored data.
        """
        if len(self.states) < 1:
            # print("⚠️ Not enough data to fit KNN.")
            return

        # Safety check
        k_neighbors = max(1, min(self.k, len(self.states)))

        self.model = NearestNeighbors(n_neighbors=k_neighbors, algorithm="kd_tree")
        self.model.fit(self.states)

    def predict_win_rate(self, state_seq, k_near=5, k_far=5):
        """
        Return the win rate based on k nearest neighbors of the input state.
        """
        # if not self.model or len(self.states) < self.k:
        if len(self.states) < 1000:
            # return True  # Not enough data
            return 1  # Not enough data

        # Find the 100 nearest neighbors
        distances, indices = self.model.kneighbors(state.reshape(1, -1), n_neighbors=50)
        distances = distances[0]
        indices = indices[0]

        # Split into nearest and farthest groups
        nearest_idx = indices[:k_near]
        nearest_dist = distances[:k_near]

        farthest_idx = indices[-k_far:]
        farthest_dist = distances[-k_far:]

        # Combine indices and distances
        combined_idx = np.concatenate([nearest_idx, farthest_idx])
        combined_dist = np.concatenate([nearest_dist, farthest_dist])

        # Get win/loss labels for selected neighbors
        selected_labels = np.array([self.labels[i] for i in combined_idx])

        # Calculate weights (closer gets higher weight)
        weights = 1 / (combined_dist + 1e-6)  # Add epsilon to avoid div-by-zero

        # Normalize weights
        weights /= weights.sum()

        # Compute weighted win rate
        win_rate = np.dot(selected_labels, weights)

        return win_rate
    def save(self):
        """
        Save the KNN model to disk.
        """
        path = f"LSTM-PPO-saves/{datetime.now().strftime('%Y-%m-%d')}-{self.symbol}.win_rate_knn.pkl"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "states": self.states,
                "labels": self.labels,
                "model": self.model
            }, f)

    def load(self):
        """
        Load the KNN model from disk.
        """
        # path = f"LSTM-PPO-saves/win_rate_knn-{self.symbol}.pkl"
        files = sorted(os.listdir("LSTM-PPO-saves"))
        files = [f for f in files if f.endswith(".win_rate_knn.pkl") and self.symbol in f]
        if not files:
            print(f"[!] No checkpoint found for {self.symbol}")
            return

        latest = os.path.join("LSTM-PPO-saves", files[-1])

        try:
            with open(latest, "rb") as f:
                data = pickle.load(f)
                self.states = data["states"]
                self.labels = data["labels"]
                self.model = data["model"]
                # print(f"✅ Loaded WinRateKNN from {latest}")
        except FileNotFoundError:
            print(f"⚠️ No saved KNN found at {path}. Starting fresh.")
    
def sharpe_ratio(returns, risk_free_rate=0.0):
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    if std_ret == 0:
        return 0
    return (mean_ret - risk_free_rate) / std_ret

def sortino_ratio(returns, risk_free_rate=0.0):
    mean_ret = np.mean(returns)
    # Downside deviation: only consider returns below risk-free rate, and their square differences
    downside_diff = [(r - risk_free_rate)**2 for r in returns if r < risk_free_rate]
    
    if len(downside_diff) == 0:
        return 0  # Or float('inf') if you'd rather signal perfect performance
    
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

def train_bot(symbol="XAUUSD"):

    print("Training bot")

    df = load_last_mb_xauusd()
    print("Loading indicators...")
    df = add_indicators(df)
    print("Loaded indicators")

    SEQ_LEN = 12 * 3

    save_count = 1440

    FEATURES = [
        # "Open",
        # "High",
        # "Low",
        # "Close",
        "k",
        "k_smooth",
        "adx",
        "+di",
        "-di",
        # "EMA7",
        # "EMA21",
        "EMA721_DIFF",
        "EMA7_Slope",
        "EMA21_Slope",
        # "EMA71_DIFF",
        "indecision",
        "bullish_ob",
        "bearish_ob",
        "bullish_fvg",
        "bearish_fvg",
        "bullish_ifvg",
        "bearish_ifvg",
        "eqh",
        "eql",
        "bearish_mb",
        "bullish_mb",
        "bullish_rb",
        "bearish_rb",
        "bullish_bb",
        "bearish_bb",
        "bullish_mss",
        "bearish_mss",
        "bullish_ob_mitigation",
        "bearish_ob_mitigation",
        "bullish_fvg_retracement",
        "bearish_fvg_retracement",
        "eqh_retest",
        "eql_retest",
        "exhaustion",
        "above_support",
        "below_resistance",
        "bullish_fib382",
        "bullish_fib500",
        "bullish_fib618",
        "bullish_fib786",
        "bearish_fib382",
        "bearish_fib500",
        "bearish_fib618",
        "bearish_fib786",
        "range_high",
        "range_low",
        "erl_high",
        "erl_low",
        "irl",
        # "vwap",
        # "vwap_upper",
        # "vwap_lower",
        "above_vwap",
        "below_vwap",
        "vwap_above_upper",
        "vwap_below_lower",
        "vwap_slope",
        "volume_ma",
        "Volume",
        "bullish",
        "bearish",
        "asia_high_dist",
        "asia_low_dist",
        "pdh_dist",
        "pdl_dist",
        "pd_poc_dist",
        "sell_score",
        "buy_score",
    ]

    agent = xLSTMPPOAgent(
        state_size=len(FEATURES),
        hidden_size=64,
        action_size=3
    )

    # knn = WinRateKNN(symbol)

    """
    try:
        agent.loadcheckpoint(symbol)
        # knn.load()
        print(f"[{symbol}] Loaded checkpoint")
    except:
        print(f"[{symbol}] Starting fresh")
    """

    save_counter = 0

    in_position = False
    position_type = None

    entry_price = 0
    sl_price = 0
    tp1_price = 0.0      # half TP
    tp2_price = 0.0      # full TP

    trade1_open = False
    trade2_open = False

    be_moved = False

    trade_returns = []

    # STANDARD_SL_PIPS = 100
    rr_list = []
    SPREAD_AND_COMMISSION = 6

    sl_pips = 0
    sl_list = []

    PIP_VALUE = 0.1

    state_buffer = deque(maxlen=SEQ_LEN)

    training_start_2 = time.time()
    training_start_3 = time.time()

    # preload sequence
    # for i in range(SEQ_LEN):
    #     row = df.iloc[i][FEATURES].values.astype(np.float32)
    #     state_buffer.append(row)

    df["range"] = (df["High"] - df["Low"]) / PIP_VALUE

    # for i in range(SEQ_LEN):
    #     row = df.iloc[i][FEATURES].values.astype(np.float32)
    #     state_buffer.append(row)

    for i in range(SEQ_LEN, len(df)):

        current = df.iloc[i]

        # base_sl = min(100, current["volume_ma"] / 10)
        # print(f"current volume ma: {current['volume_ma'] / 10}")
        # print(f"current volume: {current['Volume']/10}")
        current_price = current["Close"]
        high = current["High"]
        low = current["Low"]
        base_sl = min(50, df["range"].rolling(20).mean().iloc[i])
        # print(f"base sl: {base_sl:.2f}")
        # TP1_PIPS = round(SL_PIPS * RR_RATIO, 0)
        # relative_volume = (current["Volume"] / current["volume_ma"])
        # base_sl = max(10, current["volume_ma"] / 100)

        state = current[FEATURES].values.astype(np.float32)

        state_buffer.append(state)

        if len(state_buffer) < SEQ_LEN:
            continue

        state_seq = np.array(state_buffer)

        # === Select action ============================================
        result = agent.select_action(state_seq, in_position, training=True)

        if result is None:
            continue

        action, logprob, value, rr_mult, sl_mult = result
        # action, logprob, value = result

        # sl_mult = 1

        # rr_mult = 4

        rr_list.append(rr_mult)

        # print(f"value: {value}")

        # if df["adx"].iloc[i] < 20:
        #     action = 0

        pnl = 0.0
        done = False
        # position_multiplier = 1

        # ==============================================================
        # OPEN LONG
        # ==============================================================

        # if action == 1 and not in_position and df["+di"].iloc[i] > df["-di"].iloc[i] and df["EMA_DIFF"].iloc[i] > 0 and df["k"].iloc[i] < 80:
        # if action == 1 and not in_position and df["EMA7"].iloc[i] > df["EMA21"].iloc[i] and df["k"].iloc[i] < 80:
        # if action == 1 and not in_position and df["EMA7_Slope"].iloc[i] > 0 and df["k"].iloc[i] < 80:
        # if action == 1 and not in_position and df["k"].iloc[i] > df["k_smooth"].iloc[i]:
        if action == 1:
            """
            if df["above_vwap"].iloc[i] == 1:
                position_multiplier = 2
            if df["vwap_above_upper"].iloc[i] == 1:
                position_multiplier = 3
            """

            # print("Entered long")
            in_position = True
            trade1_open = True
            trade2_open = True
            position_type = "long"

            entry_price = current_price

            sl_pips = np.clip(base_sl * sl_mult, 20, 50)
            # sl_pips = base_sl

            sl_list.append(sl_pips)

            sl_price = entry_price - sl_pips * PIP_VALUE

            full_tp_distance = sl_pips * rr_mult * PIP_VALUE

            tp1_price = entry_price + full_tp_distance * 0.5
            tp2_price = entry_price + full_tp_distance

            # print(f"Entered long, tp1: {tp1_price:.2f}, tp2: {tp2_price:.2f}, entry: {entry_price:.2f}, sl: {sl_price:.2f}, sl pips: {sl_pips:.2f}")

            pnl = 0.0
            done = False

            agent.store_transition(
                state_seq,
                action,
                logprob,
                value,
                rr_mult,
                sl_mult,
                pnl,
                done
            )

            tp_hit = False

        # ==============================================================
        # OPEN SHORT
        # ==============================================================

        # elif action == 2 and not in_position and df["-di"].iloc[i] > df["+di"].iloc[i] and df["EMA_DIFF"].iloc[i] < 0 and df["k"].iloc[i] > 20:
        # elif action == 2 and not in_position and df["buy_score"].iloc[i] < df["sell_score"].iloc[i]:
        # elif action == 2 and not in_position and df["EMA7"].iloc[i] < df["EMA21"].iloc[i] and df["k"].iloc[i] > 20:
        # elif action == 2 and not in_position and df["EMA7_Slope"].iloc[i] < 0 and df["k"].iloc[i] > 20:
        # elif action == 2 and not in_position and df["k"].iloc[i] < df["k_smooth"].iloc[i]:
        elif action == 2:
            """
            if df["below_vwap"].iloc[i] == 1:
                position_multiplier = 2
            if df["vwap_below_lower"].iloc[i] == 1:
                position_multiplier = 3
            """

            in_position = True
            trade1_open = True
            trade2_open = True
            position_type = "short"

            entry_price = current_price

            # sl_pips = np.clip(base_sl * sl_mult, 20, 50)
            sl_pips = base_sl

            sl_list.append(sl_pips)

            sl_price = entry_price + sl_pips * PIP_VALUE

            full_tp_distance = sl_pips * rr_mult * PIP_VALUE

            tp1_price = entry_price - full_tp_distance * 0.5
            tp2_price = entry_price - full_tp_distance

            entry_price = current_price

            # print(f"Entered short, tp1: {tp1_price:.2f}, tp2: {tp2_price:.2f}, entry: {entry_price:.2f}, sl: {sl_price:.2f}, sl pips: {sl_pips:.2f}")

            reward = 0.0
            pnl = 0.0
            done = False

            agent.store_transition(
                state_seq,
                action,
                logprob,
                value,
                rr_mult,
                sl_mult,
                pnl,
                done
            )

            tp_hit = False

        # ==============================================================
        # MANAGE POSITION
        # ==============================================================
        if not in_position:
            done = False
            pnl = 0

        if in_position:

            trade_closed = False

            # ==================================================
            # LONG
            # ==================================================

            if position_type == "long":

                # ----------------------------
                # First position (half TP)
                # ----------------------------
                if trade1_open and high >= tp1_price:

                    pnl += sl_pips * rr_mult * 0.5 - SPREAD_AND_COMMISSION

                    trade1_open = False
                    # print("long tp1 hit")

                    if not be_moved:
                        sl_price = entry_price
                        be_moved = True

                # ----------------------------
                # Runner
                # ----------------------------
                if trade2_open:

                    if high >= tp2_price:
                        # print("long tp2 hit")

                        pnl += sl_pips * rr_mult - SPREAD_AND_COMMISSION

                        trade2_open = False
                        trade_closed = True
                if not trade_closed:
                    if low <= sl_price:
                        if sl_price != entry_price:
                            pnl -= sl_pips * 2 - SPREAD_AND_COMMISSION
                            # print(f"sl hit, -{sl_pips * 2:.2f}")
                        trade2_open = False
                        trade1_open = False
                        trade_closed = True

            # ==================================================
            # SHORT
            # ==================================================

            elif position_type == "short":

                # ----------------------------
                # First position (half TP)
                # ----------------------------
                if trade1_open and low <= tp1_price:
                    # print("short tp1 hit")

                    pnl += (sl_pips * rr_mult * 0.5) - SPREAD_AND_COMMISSION

                    trade1_open = False

                    if not be_moved:
                        sl_price = entry_price
                        be_moved = True

                # ----------------------------
                # Runner
                # ----------------------------
                if trade2_open:

                    if low <= tp2_price:
                        # print("short tp2 hit")

                        pnl += sl_pips * rr_mult - SPREAD_AND_COMMISSION

                        trade2_open = False
                        trade_closed = True

                if not trade_closed:
                    if high <= sl_price:
                        if sl_price != entry_price:
                            pnl -= sl_pips * 2 - SPREAD_AND_COMMISSION
                            # print(f"sl hit, -{sl_pips * 2:.2f}")

                        trade1_open = False
                        trade2_open = False
                        trade_closed = True

            if trade_closed:

                in_position = False
                done = True
                trade_returns.append(pnl)
                pnl = 0
                agent.store_transition(
                    state_seq,
                    action,
                    logprob,
                    value,
                    rr_mult,
                    sl_mult,
                    pnl,
                    done
                )

        save_counter += 1
        # print(save_counter)

        # ==============================================================
        # WEEKLY TRAINING
        # ==============================================================

        if save_counter % save_count == 0:

            # ==========================================================
            # WEEKLY STATS
            # ==========================================================

            if len(trade_returns) > 5:

                wins = [r for r in trade_returns if r > 0]
                losses = [r for r in trade_returns if r < 0]

                weekly_pnl = np.sum(trade_returns)

                winrate = (
                    len(wins) / len(trade_returns)
                    if len(trade_returns) > 0 else 0
                )

                mean_win = (
                    np.mean(wins)
                    if len(wins) > 0 else 0
                )

                mean_loss = (
                    np.mean(losses)
                    if len(losses) > 0 else 0
                )

                sharpe = sharpe_ratio(trade_returns)
                sortino = sortino_ratio(trade_returns)

                gross_profit = sum(wins)
                gross_loss = abs(sum(losses))
                profit_factor = (
                    gross_profit / gross_loss
                    if gross_loss > 0
                    else float("inf")
                )
                avg_rr = sum(rr_list) / len(rr_list) if rr_list else 0
                avg_sl = sum(sl_list) / len(sl_list) if sl_list else 0
                max_dd = max_drawdown(trade_returns)
                R_pnl = weekly_pnl / avg_sl
                # rf = (
                    # (len(trade_returns) * winrate * RR_RATIO)
                    # - (len(trade_returns) * (1 - winrate))
                # ) / (max_dd/SL_PIPS)
                #     R_pnl/(max_dd/avg_sl)
                # )
                rf = 0.0
                if max_dd > 0 and avg_sl > 0:
                    rf = R_pnl / (max_dd / (avg_sl / 10))
                else:
                    rf = 0

                print()
                print("================================================")
                print(f"[{symbol}] WEEKLY PPO TRAINING")
                print("================================================")
                print(f"Trades:          {len(trade_returns)}")
                print(f"Weekly PnL:      {weekly_pnl:.0f} pips")
                print(f"Winrate:         {winrate*100:.2f}%")
                print(f"Mean Win:        {mean_win:.0f} pips")
                print(f"Mean Loss:       {mean_loss:.0f} pips")
                print(f"Max DD:          {max_dd/(avg_sl):.2f}R")
                print(f"PF:              {profit_factor:.2f}")
                print(f"RF:              {rf:.2f}")
                # print(f"RF:              {round(rf, 0):.0f}")
                print(f"Weekly R PnL:    {R_pnl:.2f}R")
                print(f"Sharpe:          {sharpe:.2f}")
                print(f"Sortino:         {sortino:.2f}")
                print("================================================")
                print()

                print(
                    f"[{symbol}] "
                    f"[INFO]"
                    f" Trained on data (Elapsed: {timedelta(seconds=int(time.time() - training_start_3))})"
                )

                trade_returns = []

                # if len(agent.trajectory) >= 512:
                training_start = time.time()
                print(f"[{symbol}] [INFO] Training PPO...")
                agent.train()

                print(
                    f"[{symbol}] "
                    f"[INFO] Finished training PPO "
                    f"(Elapsed: {timedelta(seconds=int(time.time() - training_start))})"
                )

                # if save_counter % (1440 * 4) == 0:
                agent.savecheckpoint(symbol)

                completed = int((save_counter / save_count))
                total = int(round(len(df) / 1440, 0))

                elapsed = time.time() - training_start_2
                avg_time = elapsed / max(completed, 1)

                remaining = max(total - completed, 0)
                eta = remaining * avg_time

                print(
                    f"[{symbol}] [INFO] "
                    f"{completed}/{total} "
                    f"({completed/total*100:.1f}%) | "
                    f"Elapsed: {timedelta(seconds=int(elapsed))} | "
                    f"ETA: {timedelta(seconds=int(eta))}"
                )

                # training_start_2 = time.time()
                training_start_3 = time.time()
                sl_list = []
                rr_list = []

    # ==============================================================
    # FINAL TRAINING
    # ==============================================================

    agent.train()
    agent.savecheckpoint(symbol)

    print(
        f"[{symbol}] "
        f"[INFO] Finished training "
    )

    return agent

def open_long(symbol, lot_size, sl_pips, rr_mult):

    tick = mt5.symbol_info_tick(symbol)

    entry = tick.ask

    sl = entry - sl_pips / 10

    tp1 = entry + sl_pips / 10 * rr_mult

    tps = [tp1]

    for tp in tps:

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": mt5.ORDER_TYPE_BUY,
            "price": entry,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 123456,
            "comment": "Project Helios",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC
        }

        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Order failed: {result.retcode}")
            return None

        # return result.position

        # print(result)

def open_short(symbol, lot_size, sl_pips, rr_mult):

    tick = mt5.symbol_info_tick(symbol)

    entry = tick.bid

    sl = entry + sl_pips / 10

    tp1 = entry - sl_pips / 10 * rr_mult,

    tps = [tp1]

    for tp in tps:

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": mt5.ORDER_TYPE_SELL,
            "price": entry,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 123456,
            "comment": "Project Helios",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC
        }

        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Order failed: {result.retcode}")
            return None

        # return result.position

        # print(result)

def open_positions(symbol):
    positions = mt5.positions_get(symbol=symbol)
    positions = [
        p
        for p in positions
        if p.magic == 123456
    ]
    return len(positions)

def close_trades(magic=123456):
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
        else:
            print(f"Closed {pos.ticket}")

def modify_sl():

    positions = mt5.positions_get()

    if positions is None:
        return False

    # Only our EA's positions
    positions = [p for p in positions if p.magic == 123456]

    # Only modify if exactly one position remains
    if len(positions) != 1:
        return False

    position = positions[0]

    # Already at breakeven?
    if abs(position.sl - position.price_open) < 0.01:
        return True

    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": position.ticket,
        "symbol": position.symbol,
        "sl": position.price_open,
        "tp": position.tp,
    }

    result = mt5.order_send(request)

    return result.retcode == mt5.TRADE_RETCODE_DONE

def test_bot(symbol="XAUUSD"):
    SEQ_LEN = 12 * 3
    
    mt5.initialize()
    account = mt5.account_info()
    balance = account.balance
    RISK = 0.0005

    FEATURES = [
        # "Open",
        # "High",
        # "Low",
        # "Close",
        "k",
        "k_smooth",
        "adx",
        "+di",
        "-di",
        # "EMA7",
        # "EMA21",
        "EMA721_DIFF",
        "EMA7_Slope",
        "EMA21_Slope",
        # "EMA71_DIFF",
        "indecision",
        "bullish_ob",
        "bearish_ob",
        "bullish_fvg",
        "bearish_fvg",
        "bullish_ifvg",
        "bearish_ifvg",
        "eqh",
        "eql",
        "bearish_mb",
        "bullish_mb",
        "bullish_rb",
        "bearish_rb",
        "bullish_bb",
        "bearish_bb",
        "bullish_mss",
        "bearish_mss",
        "bullish_ob_mitigation",
        "bearish_ob_mitigation",
        "bullish_fvg_retracement",
        "bearish_fvg_retracement",
        "eqh_retest",
        "eql_retest",
        "exhaustion",
        "above_support",
        "below_resistance",
        "bullish_fib382",
        "bullish_fib500",
        "bullish_fib618",
        "bullish_fib786",
        "bearish_fib382",
        "bearish_fib500",
        "bearish_fib618",
        "bearish_fib786",
        "range_high",
        "range_low",
        "erl_high",
          "erl_low",
        "irl",
        # "vwap",
        # "vwap_upper",
        # "vwap_lower",
        "above_vwap",
        "below_vwap",
        "vwap_above_upper",
        "vwap_below_lower",
        "vwap_slope",
        "volume_ma",
        "Volume",
        "bullish",
        "bearish",
        "asia_high_dist",
        "asia_low_dist",
        "pdh_dist",
        "pdl_dist",
        "pd_poc_dist",
        "sell_score",
        "buy_score",
    ]

    # last_m15 = None
    last_m5 = None

    agent = xLSTMPPOAgent(
        state_size=len(FEATURES),
        hidden_size=64,
        action_size=3
    )

    # agent.model.debug = True
    # try:
    agent.loadcheckpoint("XAUUSD")
    # except:
    #     print("No file for prior training, cancelling test.")
    #     return

    # ==========================================================
    # INITIAL LOAD
    # ==========================================================

    rates_m5 = mt5.copy_rates_from_pos(
        symbol,
        mt5.TIMEFRAME_M5,
        0,
        600
    )

    df = pd.DataFrame(rates_m5)

    df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'time': 'Date',
        "tick_volume": "Volume"
    }, inplace=True)
    # print(f"columns: {df.columns}")

    # df["Date"] = pd.to_datetime(df["Date"], unit="s")
    df["Date"] = (
    pd.to_datetime(df["Date"], unit="s", utc=True)
      # .dt.tz_convert("Europe/Helsinki")
    )
    df.set_index("Date", inplace=True)

    raw_df = df

    df = add_indicators(df)

    last_m5 = df.index[-1]

    # trade = []

    # ==========================================================
    # MAIN LOOP
    # ==========================================================

    # print("entering main loop in test function")
    while True:

        # print("in main loop")
        now = datetime.now()
        position_multiplier = 1

        seconds_until_next_5m = (
            (5 - now.minute % 5) * 60
            - now.second
            - now.microsecond / 1_000_000
        )
        # print(f"sleeping {seconds_until_next_5m:.0f} seconds, current time: {datetime.now()}")
        if seconds_until_next_5m <= 0:
            seconds_until_next_5m += 30

        time.sleep(seconds_until_next_5m)
        # print(f"slept {seconds_until_next_5m:.0f} seconds, current time: {datetime.now()}")

        tick = mt5.symbol_info_tick(symbol)
        # SL_PIPS = round(tick.bid * 0.00125 * 10, 0)
        # SL_PIPS = 40

        # ======================================================
        # CHECK FOR NEW M15 CANDLE
        # ======================================================

        new_m5 = mt5.copy_rates_from_pos(
            symbol,
            mt5.TIMEFRAME_M5,
            0,
            1
        )

        current_m5 = new_m5[0]["time"]

        while True:
            if current_m5 != last_m5:
                break
            else:
                time.sleep(0.25)
                new_m5 = mt5.copy_rates_from_pos(
                    symbol,
                    mt5.TIMEFRAME_M5,
                    0,
                    1
                )
                current_m5 = new_m5[0]["time"]

        # current_m5 = new_m5[0]["time"]

        if current_m5 != last_m5:

            last_m5 = current_m5

            # ==================================================
            # APPEND NEW CANDLE
            # ==================================================

            new_row = pd.DataFrame(new_m5)

            new_row.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'time': 'Date',
                "tick_volume": "Volume"
            }, inplace=True)

            new_row["Date"] = (
                pd.to_datetime(new_row["Date"], unit="s", utc=True)
                # .dt.tz_convert("Europe/Helsinki")
            )
            new_row.set_index("Date", inplace=True)

            if new_row.index[-1] != df.index[-1]:

                raw_df = pd.concat(
                    [raw_df, new_row]
                )

                raw_df = raw_df.tail(600)

                df = add_indicators(raw_df.copy())

            # ==================================================
            # BUILD STATE SEQUENCE
            # ==================================================

            state_seq = (
                df[FEATURES]
                .tail(SEQ_LEN)
                .values
                .astype(np.float32)
            )

            # ==================================================
            # POSITION CHECK
            # ==================================================

            open_pos = open_positions(symbol)

            # ==================================================
            # PPO DECISION
            # ==================================================

            if state_seq.shape[0] != SEQ_LEN:
                print(f"Bad state shape: {state_seq.shape}")
                continue

            result = agent.select_action(state_seq, open_pos > 0, training=True)

            if result is None:
                continue

            action, logprob, value, rr_mult, sl_mult = result

            # if value < 0:
            #     value = value * -1

            current_time = df.index[-1]  # or however you store timestamps
            # print(f"time: {current_time.hour}h{current_time.minute}m")

            base_sl = max(10, df["Volume"].iloc[-1] / 100)

            if current_time.hour == 23 and current_time.minute >= 54:
                # Force close any open position
                if open_pos != 0:
                    close_trades()

                # Force HOLD
                action = 0

            # print(f"adx: {df['adx'].iloc[-1]}")
            # print(f"+di: {df['+di'].iloc[-1]}")
            # print(f"-di: {df['-di'].iloc[-1]}")
            # print(f"stoch k: {df['k'].iloc[-1]}")
            # print(f"stock d: {df['k_smooth'].iloc[-1]}")

            # if df["adx"].iloc[-1] < 20:
            #     action = 0

            # print(f"Test action: {action}")

            # ==================================================
            # OPEN NEW TRADE
            # ==================================================

            if open_pos == 0:

                # trade = None

                account = mt5.account_info()

                balance = account.balance

                # risk_per_position = max(
                #     balance * RISK / 500 / 4,
                #     0.01
                # )

                # if action == 1 and df["adx"].iloc[-1] > 20 and df["+di"].iloc[-1] > df["-di"].iloc[-1] and df["EMA_DIFF"].iloc[-1] > 0 and df["k"].iloc[-1] < 80:
                # if action == 1 and df["buy_score"].iloc[-1] > df["sell_score"].iloc[-1]:
                # if action == 1 and df["EMA7"].iloc[-1] > df["EMA21"].iloc[-1] and df["k"].iloc[-1] < 80:
                if action == 1:
                    """
                    if df["above_vwap"].iloc[-1] == 1:
                        position_multiplier = 2
                    if df["vwap_above_upper"].iloc[-1] == 1:
                        position_multiplier = 3
                    """
                    # print(
                    #     f"[{symbol}] PPO BUY"
                    # )

                    sl_pips = np.clip(
                        base_sl * sl_mult,
                        20,
                        50
                    )

                    risk_per_position = min(
                        max((balance * RISK) / (sl_pips* 10), 0.01),
                        100.0
                    )
                    risk_per_position = round(risk_per_position, 2)

                    # sl_price = entry_price - (sl_pips * PIP_VALUE)
                    # if value < 0:
                    #     value = value * -1
                    # tp_price = entry_price + (
                    #     base_sl * rr_mult * PIP_VALUE
                    # # 2
                    # )

                    half_lot = max(round(risk_per_position / 2, 2), 0.01)

                    open_long(
                        symbol,
                        half_lot,
                        sl_pips,
                        rr_mult * 0.5
                    )

                    open_long(
                        symbol,
                        half_lot,
                        sl_pips,
                        rr_mult
                    )

                    # trade.append({
                    #     "tp1_ticket": ticket1,
                    #     "tp2_ticket": ticket2,
                    #     "entry": df["Close"].iloc[-1],
                    #     "sl": df["Close"].iloc[-1] - sl_pips / 10,
                    #     "breakeven": False,
                    # })

                # elif action == 2 and df["adx"].iloc[-1] > 20 and df["-di"].iloc[-1] > df["+di"].iloc[-1] and df["EMA_DIFF"].iloc[-1] < 0 and df["k"].iloc[-1] > 20:
                # elif action == 2 and df["buy_score"].iloc[-1] < df["sell_score"].iloc[-1]:
                # elif action == 2 and df["EMA7"].iloc[-1] < df["EMA21"].iloc[-1] and df["k"].iloc[-1] > 20:
                elif action == 2:
                    """
                    if df["below_vwap"].iloc[-1] == 1:
                        position_multiplier = 2
                    if df["vwap_below_lower"].iloc[-1] == 1:
                        position_multiplier = 3
                    """
                    # print(
                    #     f"[{symbol}] PPO SELL"
                    # )

                    sl_pips = np.clip(
                        base_sl * sl_mult,
                        20,
                        50
                    )

                    risk_per_position = min(
                        max((balance * RISK) / (sl_pips* 10), 0.01),
                        100.0
                    )
                    risk_per_position = round(risk_per_position, 2)

                    # sl_price = entry_price + (sl_pips * PIP_VALUE)
                    # if value < 0:
                    #     value = value * -1
                    # tp_price = entry_price - (
                    #     base_sl * rr_mult * PIP_VALUE
                    #     # 2
                    # )

                    half_lot = max(round(risk_per_position / 2, 2), 0.01)

                    open_short(
                        symbol,
                        half_lot,
                        sl_pips,
                        rr_mult * 0.5
                    )

                    open_short(
                        symbol,
                        half_lot,
                        sl_pips,
                        rr_mult
                    )

                    # trade.append({
                    #     "tp1_ticket": ticket1,
                    #     "tp2_ticket": ticket2,
                    #     "entry": df["Close"].iloc[-1],
                    #     "sl": df["Close"].iloc[-1] + sl_pips / 10,
                    #     "breakeven": False,
                    # })

                # else:

                #     print(
                #         f"[{symbol}] PPO HOLD"
                #     )
            else:
                positions = mt5.positions_get()

                if positions is not None:

                    # Only this EA's positions
                    positions = [p for p in positions if p.magic == 123456]

                    # TP1 has closed, only TP2 remains
                    if len(positions) == 1:
                        modify_sl()

def get_last_date():

    if not os.path.exists(CSV_FILE):
        return None

    df = pd.read_csv(
        CSV_FILE,
        sep=";"
    )

    if df.empty:
        return None

    return pd.to_datetime(
        df["Date"].iloc[-1]
    )

def download_xauusd_data():

    last_date = get_last_date()

    if (
        last_date is not None
        and (
            datetime.now().date()
            - last_date.date()
        ).days <= 90
    ):

        print(
            "Data already up to date."
        )

        return None

    if last_date is None:

        start_date = (
            datetime.now()
            - timedelta(days=365 * 3)
        ).strftime(
            "%Y-%m-%d"
        )

    else:

        start_date = (
            last_date
            - timedelta(days=1)
        ).strftime(
            "%Y-%m-%d"
        )

    end_date = (
        datetime.now()
        - timedelta(days=1)
    ).strftime(
        "%Y-%m-%d"
    )

    print(
        f"Downloading "
        f"{start_date} -> {end_date}"
    )

    subprocess.run(
        [
            # "npx",
            "dukascopy-node",
            "-i",
            "xauusd",
            "-from",
            start_date,
            "-to",
            end_date,
            "-t",
            "m5",
            "-f",
            "csv"
        ],
        check=True
    )

    files = [
        f
        for f in os.listdir(".")
        if f.startswith("xauusd")
        and f.endswith(".csv")
    ]

    if not files:

        raise FileNotFoundError(
            "No Dukascopy CSV was downloaded."
        )

    return max(
        files,
        key=os.path.getmtime
    )

def append_xauusd_data(downloaded_file):

    if downloaded_file is None:
        return

    new_df = pd.read_csv(
        downloaded_file
    )

    new_df.rename(
        columns={
            "timestamp": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume"
        },
        inplace=True
    )

    if os.path.exists(CSV_FILE):

        old_df = pd.read_csv(
            CSV_FILE,
            sep=";"
        )

        df = pd.concat(
            [
                old_df,
                new_df
            ],
            ignore_index=True
        )

    else:

        df = new_df

    df.drop_duplicates(
        subset=["Date"],
        keep="last",
        inplace=True
    )

    df.sort_values(
        "Date",
        inplace=True
    )

    df.to_csv(
        CSV_FILE,
        sep=";",
        index=False
    )

    os.remove(
        downloaded_file
    )

    print(
        f"Saved "
        f"{len(df)} candles "
        f"to {CSV_FILE}"
    )

def update_xauusd_data():

    downloaded_file = (
        download_xauusd_data()
    )

    append_xauusd_data(
        downloaded_file
    )

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--symbol", default="XAUUSD-VIP")

    args = parser.parse_args()

    threads = []

    if args.train:
        t = threading.Thread(
            target=train_bot,
            # args=(args.symbol),
            daemon=True
        )
        t.start()
        threads.append(t)

    if args.test:
        t = threading.Thread(
            target=test_bot,
            args=(args.symbol,),
            daemon=True
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

main()
