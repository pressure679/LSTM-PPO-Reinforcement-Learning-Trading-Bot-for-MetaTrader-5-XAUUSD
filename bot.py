import pandas as pd
import numpy as np
import os
import pickle
from io import StringIO, BytesIO
import random
from collections import deque
from datetime import datetime, timedelta
import subprocess
import time
import argparse
import multiprocessing
import math
import glob
import json
import urllib.request
import MetaTrader5 as mt5
from xgboost import XGBClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

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

def load_last_mb_xauusd(file_path=None, mb=20, delimiter=',', col_names=None):
    if file_path is None:
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "download", "XAUUSD.csv")

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

    df = pd.read_csv(
        StringIO(data),
        delimiter=delimiter,
        header=0
    )

    # XAUUSD.csv columns: Date (YYYYMMDD), Timestamp (HH:MM:SS), Open,
    # High, Low, Close, Volume — date and time of day are separate
    # fields, unlike the old single-column "%Y.%m.%d %H:%M" export.
    df.columns = ["Date", "Timestamp", "Open", "High", "Low", "Close", "Volume"]

    df["Date"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Timestamp"],
        format="%Y%m%d %H:%M:%S",
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

def GetRange(df):
    """
    Returns normalized candle range as a percentage of price.
    """
    return (df["High"] - df["Low"]) * 10

def RangeMA(df, period=14):

    # -----------------------------------------
    # Select volume column
    # -----------------------------------------
    # if "volume" in df.columns:
    #     volume = df["Volume"]
    # elif "tick_volume" in df.columns:
    #     volume = df["tick_volume"]
    # elif "Volume" in df.columns:
    #     volume = df["Volume"]
    # else:
    #     raise ValueError("No volume column found.")

    volume = df["range"]

    # print(type(volume))
    # print(volume.shape)
    # print(df.columns.tolist())

    return round(volume.rolling(period).mean(), 2)

def BullishBearish(df):
    # Returns:
    #     bullish (np.ndarray[bool])
    #     bearish (np.ndarray[bool])
    

    open = df["Open"].to_numpy()
    close = df["Close"].to_numpy()

    bullish = close > open
    bearish = close < open

    return bullish, bearish

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

    swing_high = df["swing_high"].to_numpy()
    swing_low = df["swing_low"].to_numpy()
    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()
    close = df["Close"].to_numpy()

    bullish = np.zeros(len(df), dtype=np.int8)
    bearish = np.zeros(len(df), dtype=np.int8)

    last_swing_high = None
    last_swing_low = None

    for i in range(len(df)):

        if swing_high[i]:
            last_swing_high = high[i]

        if swing_low[i]:
            last_swing_low = low[i]

        if (
            last_swing_high is not None
            and close[i] > last_swing_high
        ):
            bullish[i] = 1
            last_swing_high = None

        if (
            last_swing_low is not None
            and close[i] < last_swing_low
        ):
            bearish[i] = 1
            last_swing_low = None

    return bullish, bearish

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
    # Support — ascending line through the latest swing low
    # and the most recent earlier swing low that's actually
    # lower than it, searching back past any non-conforming
    # swing lows in between rather than only ever comparing
    # immediate neighbors.
    # =====================================================

    lows = np.where(swing_low)[0]

    for k in range(1, len(lows)):

        i2 = lows[k]

        i1 = None
        for m in range(k - 1, -1, -1):
            if low[i2] > low[lows[m]]:
                i1 = lows[m]
                break

        if i1 is None:
            continue

        slope = (low[i2] - low[i1]) / (i2 - i1)

        end = lows[k + 1] if k + 1 < len(lows) else n

        for j in range(i2 + 1, end):

            trend_price = low[i2] + slope * (j - i2)

            if close[j] >= trend_price:
                above_support[j] = True

    # =====================================================
    # Resistance — descending line through the latest swing
    # high and the most recent earlier swing high that's
    # actually higher than it, same back-searching approach.
    # =====================================================

    highs = np.where(swing_high)[0]

    for k in range(1, len(highs)):

        i2 = highs[k]

        i1 = None
        for m in range(k - 1, -1, -1):
            if high[i2] < high[highs[m]]:
                i1 = highs[m]
                break

        if i1 is None:
            continue

        slope = (high[i2] - high[i1]) / (i2 - i1)

        end = highs[k + 1] if k + 1 < len(highs) else n

        for j in range(i2 + 1, end):

            trend_price = high[i2] + slope * (j - i2)

            if close[j] <= trend_price:
                below_resistance[j] = True

    return above_support, below_resistance

def BullishOB(df, multiplier=1.5):
    body = (df["Close"] - df["Open"]).abs()
    next_body = body.shift(-1)

    bearish = df["Close"] < df["Open"]
    next_bullish = df["Close"].shift(-1) > df["Open"].shift(-1)

    return (
        bearish &
        next_bullish &
        (body > df["range_ma"]) &
        (next_body >= body * multiplier)
    ).astype(int)

def BearishOB(df, multiplier=1.5):
    body = (df["Close"] - df["Open"]).abs()
    next_body = body.shift(-1)

    bullish = df["Close"] > df["Open"]
    next_bearish = df["Close"].shift(-1) < df["Open"].shift(-1)

    return (
        bullish &
        next_bearish &
        (body > df["range_ma"]) &
        (next_body >= body * multiplier)
    ).astype(int)

def OBMitigation(df, threshold=30, lookback=72):
    """
    Tracks each order block's own candle low/high as its level, same
    pattern as ReversalBlockMitigation()/BullishBB()/BearishBB() — not
    the binary bullish_ob/bearish_ob flag itself, which isn't a price
    level and would make the check permanently false.
    """
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

def Fibonacci(df, min_swing=7.5, tolerance=0.05):
    """
    Recomputes `threshold = swing * tolerance` in both the bullish- and
    bearish-impulse branches — computing it only in the bullish branch
    would leave the bearish branch reusing a stale value (or crashing
    with a NameError if no bullish impulse has happened yet).
    """

    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()
    open_ = df["Open"].to_numpy()
    close = df["Close"].to_numpy()

    body_high = np.maximum(open_, close)
    body_low = np.minimum(open_, close)

    swing_high = df["swing_high"].to_numpy()
    swing_low = df["swing_low"].to_numpy()

    bullish_fib236 = np.zeros(len(df), dtype=np.bool_)
    bullish_fib382 = np.zeros(len(df), dtype=np.bool_)
    bullish_fib500 = np.zeros(len(df), dtype=np.bool_)
    bullish_fib618 = np.zeros(len(df), dtype=np.bool_)
    bullish_fib786 = np.zeros(len(df), dtype=np.bool_)

    bearish_fib236 = np.zeros(len(df), dtype=np.bool_)
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

                fib236 = high[i] - swing * 0.236
                fib382 = high[i] - swing * 0.382
                fib500 = high[i] - swing * 0.500
                fib618 = high[i] - swing * 0.618
                fib786 = high[i] - swing * 0.786
                threshold = swing * tolerance

                j = i

                while (
                    j < n and
                    not swing_high[j] and
                    not swing_low[j]
                ):

                    bullish_fib236[j] = (
                        body_low[j] <= fib236 + threshold and
                        body_high[j] >= fib236 - threshold
                    )

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

                fib236 = low[i] + swing * 0.236
                fib382 = low[i] + swing * 0.382
                fib500 = low[i] + swing * 0.500
                fib618 = low[i] + swing * 0.618
                fib786 = low[i] + swing * 0.786
                threshold = swing * tolerance

                j = i

                while (
                    j < n and
                    not swing_high[j] and
                    not swing_low[j]
                ):

                    bearish_fib236[j] = (
                        body_low[j] <= fib236 + threshold and
                        body_high[j] >= fib236 - threshold
                    )

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
        bullish_fib236,
        bullish_fib382,
        bullish_fib500,
        bullish_fib618,
        bullish_fib786,
        bearish_fib236,
        bearish_fib382,
        bearish_fib500,
        bearish_fib618,
        bearish_fib786,
    )

def FVGRetracement(df, tolerance=0.05):

    open_ = df["Open"].to_numpy()
    close = df["Close"].to_numpy()

    body_high = np.maximum(open_, close)
    body_low = np.minimum(open_, close)
    body_mid = (body_high + body_low) / 2

    bull_fvg = df["bullish_fvg"].to_numpy()
    bear_fvg = df["bearish_fvg"].to_numpy()

    bull_ret = np.zeros(len(df), dtype=np.bool_)
    bear_ret = np.zeros(len(df), dtype=np.bool_)

    n = len(df)

    # -------------------------
    # Bullish FVG
    # -------------------------
    for i in range(1, n - 1):

        if not bull_fvg[i]:
            continue

        # Fib: 0 = first candle low, 1 = third candle high
        swing_low = df["Low"].iat[i - 1]
        swing_high = df["High"].iat[i + 1]

        height = swing_high - swing_low
        if height <= 0:
            continue

        fib382 = swing_high - 0.382 * height
        fib500 = swing_high - 0.500 * height

        tol = height * tolerance

        for j in range(i + 2, n):

            # Stop when bullish continuation resumes
            if close[j] >= open_[j]:
                break

            if (
                abs(body_mid[j] - fib382) <= tol or
                abs(body_mid[j] - fib500) <= tol
            ):
                bull_ret[j] = True
                break

    # -------------------------
    # Bearish FVG
    # -------------------------
    for i in range(1, n - 1):

        if not bear_fvg[i]:
            continue

        # Fib: 0 = first candle high, 1 = third candle low
        swing_high = df["High"].iat[i - 1]
        swing_low = df["Low"].iat[i + 1]

        height = swing_high - swing_low
        if height <= 0:
            continue

        fib382 = swing_low + 0.382 * height
        fib500 = swing_low + 0.500 * height

        tol = height * tolerance

        for j in range(i + 2, n):

            # Stop when bearish continuation resumes
            if close[j] <= open_[j]:
                break

            if (
                abs(body_mid[j] - fib382) <= tol or
                abs(body_mid[j] - fib500) <= tol
            ):
                bear_ret[j] = True
                break

    return bull_ret, bear_ret

def MSSFibRetracement(df, lookback=72, tolerance=0.05):
    # The 0.786 retracement of the impulse from the swing low that
    # anchored the broken swing high (bullish) / swing high that anchored
    # the broken swing low (bearish), up to the MSS breakout candle's own
    # high/low, flagged while price closes near that level within
    # `lookback` bars of the MSS event (and before an opposing MSS
    # invalidates the setup).

    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()
    close = df["Close"].to_numpy()

    swing_high = df["swing_high"].to_numpy()
    swing_low = df["swing_low"].to_numpy()

    bullish_mss = df["bullish_mss"].to_numpy()
    bearish_mss = df["bearish_mss"].to_numpy()

    bullish_mss_retracement = np.zeros(len(df), dtype=np.bool_)
    bearish_mss_retracement = np.zeros(len(df), dtype=np.bool_)

    last_swing_low_val = None
    last_swing_high_val = None

    # The swing low/high in place at the moment each swing high/low formed
    # — the anchor an MSS event later uses when it breaks that swing.
    swing_low_for_last_high = None
    swing_high_for_last_low = None

    n = len(df)

    for i in range(n):

        if swing_low[i]:
            last_swing_low_val = low[i]
            swing_high_for_last_low = last_swing_high_val

        if swing_high[i]:
            last_swing_high_val = high[i]
            swing_low_for_last_high = last_swing_low_val

        if bullish_mss[i] and swing_low_for_last_high is not None:

            mss_high = high[i]
            impulse = mss_high - swing_low_for_last_high

            if impulse > 0:

                level = mss_high - 0.786 * impulse
                threshold = impulse * tolerance
                end = min(i + 1 + lookback, n)

                for j in range(i + 1, end):
                    if bearish_mss[j]:
                        break
                    if abs(close[j] - level) <= threshold:
                        bullish_mss_retracement[j] = True
                        break

        if bearish_mss[i] and swing_high_for_last_low is not None:

            mss_low = low[i]
            impulse = swing_high_for_last_low - mss_low

            if impulse > 0:

                level = mss_low + 0.786 * impulse
                threshold = impulse * tolerance
                end = min(i + 1 + lookback, n)

                for j in range(i + 1, end):
                    if bullish_mss[j]:
                        break
                    if abs(close[j] - level) <= threshold:
                        bearish_mss_retracement[j] = True
                        break

    return bullish_mss_retracement, bearish_mss_retracement

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

def EQH_EQL(df, threshold=10, lookback=72, min_distance=3):

    open_ = df["Open"].to_numpy()
    close = df["Close"].to_numpy()
    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()

    swing_high = df["swing_high"].to_numpy()
    swing_low = df["swing_low"].to_numpy()

    body_high = np.maximum(open_, close)
    body_low = np.minimum(open_, close)

    eqh = np.zeros(len(df), dtype=np.bool_)
    eql = np.zeros(len(df), dtype=np.bool_)

    n = len(df)

    for i in range(n):

        start = max(0, i - lookback)

        for j in range(i - min_distance, start - 1, -1):

            # Equal High — only against a prior swing high, not any candle
            if swing_high[j] and (
                abs(high[i] - high[j]) <= threshold or
                abs(high[i] - body_high[j]) <= threshold or
                abs(body_high[i] - high[j]) <= threshold or
                abs(body_high[i] - body_high[j]) <= threshold
            ):
                eqh[i] = True

            # Equal Low — only against a prior swing low, not any candle
            if swing_low[j] and (
                abs(low[i] - low[j]) <= threshold or
                abs(low[i] - body_low[j]) <= threshold or
                abs(body_low[i] - low[j]) <= threshold or
                abs(body_low[i] - body_low[j]) <= threshold
            ):
                eql[i] = True

            if eqh[i] and eql[i]:
                break

    return eqh, eql

def EQHEQLRetest(df, threshold=20, lookback=12):

    open_ = df["Open"].to_numpy()
    close = df["Close"].to_numpy()
    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()

    body_high = np.maximum(open_, close)
    body_low = np.minimum(open_, close)

    eqh = df["eqh"].to_numpy()
    eql = df["eql"].to_numpy()
    bearish = df["bearish"].to_numpy()

    eqh_retest = np.zeros(len(df), dtype=np.bool_)
    eql_retest = np.zeros(len(df), dtype=np.bool_)

    n = len(df)

    for i in range(n):

        start = max(0, i - lookback)

        # Previous EQH
        for j in range(i - 1, start - 1, -1):

            if not eqh[j]:
                continue

            level = max(high[j], body_high[j])

            if (
                abs(high[i] - level) <= threshold or
                abs(body_high[i] - level) <= threshold
                # and df["bullish"].iloc[i]
            ):
                eqh_retest[i] = True
                break

        # Previous EQL
        for j in range(i - 1, start - 1, -1):

            if not eql[j]:
                continue

            level = min(low[j], body_low[j])

            if (
                (
                    abs(low[i] - level) <= threshold or
                    abs(body_low[i] - level) <= threshold
                )
                and bearish[i]
            ):
                eql_retest[i] = True
                break

    return eqh_retest, eql_retest

def MSS_Retest(df, lookback=72, threshold=3):

    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()

    bullish_mss = df["bullish_mss"].to_numpy()
    bearish_mss = df["bearish_mss"].to_numpy()

    bullish_retest = np.zeros(len(df), dtype=np.int8)
    bearish_retest = np.zeros(len(df), dtype=np.int8)

    last_bull_level = None
    last_bear_level = None
    last_bull_index = -1
    last_bear_index = -1

    for i in range(len(df)):

        # Remember bullish MSS level
        if bullish_mss[i]:
            last_bull_level = high[i]
            last_bull_index = i

        # Remember bearish MSS level
        if bearish_mss[i]:
            last_bear_level = low[i]
            last_bear_index = i

        # Bullish retest
        if (
            last_bull_level is not None
            and i - last_bull_index <= lookback
            and low[i] <= last_bull_level + threshold
            and high[i] >= last_bull_level - threshold
        ):
            bullish_retest[i] = 1
            last_bull_level = None

        # Bearish retest
        if (
            last_bear_level is not None
            and i - last_bear_index <= lookback
            and high[i] >= last_bear_level - threshold
            and low[i] <= last_bear_level + threshold
        ):
            bearish_retest[i] = 1
            last_bear_level = None

    return bullish_retest, bearish_retest

def BoS(df):

    swing_high = df["swing_high"].to_numpy()
    swing_low = df["swing_low"].to_numpy()
    high_arr = df["High"].to_numpy()
    low_arr = df["Low"].to_numpy()

    bullish_bos = np.zeros(len(df), dtype=np.int8)
    bearish_bos = np.zeros(len(df), dtype=np.int8)

    last_high = None
    last_low = None

    trend = None

    for i in range(len(df)):

        # New swing high
        if swing_high[i]:

            high = high_arr[i]

            if last_high is not None:

                if high > last_high:
                    # Higher High
                    if trend == "bull":
                        bullish_bos[i] = 1
                    trend = "bull"

                elif high < last_high:
                    # Lower High
                    trend = "bear"

            last_high = high

        # New swing low
        if swing_low[i]:

            low = low_arr[i]

            if last_low is not None:

                if low < last_low:
                    # Lower Low
                    if trend == "bear":
                        bearish_bos[i] = 1
                    trend = "bear"

                elif low > last_low:
                    # Higher Low
                    trend = "bull"

            last_low = low

    return bullish_bos, bearish_bos

def Trend(df):

    bullish_bos = df["bullish_bos"].to_numpy()
    bearish_bos = df["bearish_bos"].to_numpy()

    bullish_trend = np.zeros(len(df), dtype=np.int8)
    bearish_trend = np.zeros(len(df), dtype=np.int8)

    bull_count = 0
    bear_count = 0

    trend = 0
    # 0 = none
    # 1 = bullish
    # -1 = bearish

    for i in range(len(df)):

        if bullish_bos[i]:
            bull_count += 1
            bear_count = 0

            if bull_count >= 2:
                trend = 1

        elif bearish_bos[i]:
            bear_count += 1
            bull_count = 0

            if bear_count >= 2:
                trend = -1

        if trend == 1:
            bullish_trend[i] = 1
        elif trend == -1:
            bearish_trend[i] = 1

    return bullish_trend, bearish_trend

def CHoCH(df, lookback=72):

    bullish_trend = df["bullish_trend"].to_numpy()
    bearish_trend = df["bearish_trend"].to_numpy()
    swing_low = df["swing_low"].to_numpy()
    swing_high = df["swing_high"].to_numpy()
    low = df["Low"].to_numpy()
    high = df["High"].to_numpy()
    close = df["Close"].to_numpy()

    bullish_choch = np.zeros(len(df), dtype=np.int8)
    bearish_choch = np.zeros(len(df), dtype=np.int8)

    for i in range(lookback, len(df)):

        start = max(0, i - lookback)

        # ---------- Bullish trend -> look for bearish CHoCH ----------
        if bullish_trend[i]:

            last_hl = None

            for j in range(i - 1, start - 1, -1):
                if swing_low[j]:
                    last_hl = low[j]
                    break

            if last_hl is not None and close[i] < last_hl:
                bearish_choch[i] = 1

        # ---------- Bearish trend -> look for bullish CHoCH ----------
        elif bearish_trend[i]:

            last_lh = None

            for j in range(i - 1, start - 1, -1):
                if swing_high[j]:
                    last_lh = high[j]
                    break

            if last_lh is not None and close[i] > last_lh:
                bullish_choch[i] = 1

    return bullish_choch, bearish_choch

def BullishBB(df, lookback=72):

    bb = np.zeros(len(df), dtype=np.int8)

    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()
    bearish_ob = df["bearish_ob"].to_numpy()

    n = len(df)

    for i in range(n):

        start = max(0, i - lookback)

        for j in range(i - 1, start - 1, -1):

            if bearish_ob[j] != 1:
                continue

            ob_high = high[j]
            ob_low = low[j]

            # Highest high and lowest low since the OB formed
            highest = np.max(high[j+1:i+1])
            lowest = np.min(low[j+1:i+1])

            if highest >= ob_high and lowest <= ob_low:
                bb[i] = 1
                break

    return bb

def BearishBB(df, lookback=72):

    bb = np.zeros(len(df), dtype=np.int8)

    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()
    bullish_ob = df["bullish_ob"].to_numpy()

    n = len(df)

    for i in range(n):

        start = max(0, i - lookback)

        for j in range(i - 1, start - 1, -1):

            if bullish_ob[j] != 1:
                continue

            ob_high = high[j]
            ob_low = low[j]

            # Highest high and lowest low since the OB formed
            highest = np.max(high[j + 1:i + 1])
            lowest = np.min(low[j + 1:i + 1])

            # Order block has been fully mitigated
            if highest >= ob_high and lowest <= ob_low:
                bb[i] = 1
                break

    return bb

def ExhaustionCandle(df):
    open_ = df["Open"].to_numpy()
    close = df["Close"].to_numpy()

    body = np.abs(close - open_)

    exhaustion = np.zeros(len(df), dtype=np.bool_)

    exhaustion[1:] = body[1:] <= body[:-1] * 0.10

    return exhaustion

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

    df["range_high"] = np.where(df["swing_high"], df["High"], np.nan)
    df["range_low"] = np.where(df["swing_low"], df["Low"], np.nan)

    df["range_high"] = df["range_high"].ffill()
    df["range_low"] = df["range_low"].ffill()

    df["erl_high"] = df["High"] > df["range_high"]
    df["erl_low"] = df["Low"] < df["range_low"]

    df["irl"] = (
        (df["High"] <= df["range_high"])
        & (df["Low"] >= df["range_low"])
    )

    return df

def GetKillzone(df, asia=(1, 4), london=(10, 13), newyork=(15, 18)):
    """
    Returns:
        0 = None
        1 = Asia
        2 = London
        3 = New York

    Expects df.index to be a DatetimeIndex.
    """

    hours = df.index.hour

    session = np.zeros(len(df), dtype=np.int8)

    session[(hours >= asia[0]) & (hours < asia[1])] = 1
    session[(hours >= london[0]) & (hours < london[1])] = 2
    session[(hours >= newyork[0]) & (hours < newyork[1])] = 3

    return session

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

def BuyScore(df):
    """Weighted sum of bullish confluences — trend alignment, OB
    mitigation, breaker blocks, MSS/CHoCH/MSS-retest/MSS-retracement,
    FVG+MB/RB confluence, liquidity distances, equal-highs retest, fib
    retracement in-trend, trendline position — gated to killzones and
    adx>=20."""
    score = (
        # Trend filter
        (df["EMA7"] > df["EMA21"]).astype(int) +
        (df["EMA721_DIFF"] > 0).astype(int) +
        (df["EMA7_Slope"] > 0).astype(int) +
        # (df["EMA2150_DIFF"] > 0).astype(int) +
        (df["EMA21_Slope"] > 0).astype(int) +
        # (df["EMA50"] > df["EMA200"]).astype(int) +
        # (df["EMA50200_DIFF"] > 0).astype(int) +
        # (df["EMA50_Slope"] > 0).astype(int) +
        # (df["EMA200_Slope"] > 0).astype(int) +
        (df["+di"] > df["-di"]).astype(int) +
        (df["adx"] > 20).astype(int) +
        (df["k"] > df["k_smooth"]).astype(int) +

        # OB mitigation
        df["bullish_ob_mitigation"] +

        # Breaker Block
        df["bullish_bb"] +

        df["bullish_rb"] +

        df["bullish_mb"] +

        df["bullish_ifvg"] +

        (df["bearish"].shift(1) &
        df["exhaustion"]) +

        df["bullish_ob"] +

        # MSS
        # (df["bullish_mss"] &
        # (df["bullish_trend"])
        # ).astype(int) +

        df["bullish_choch"] +

        # MSS Retest
        df["bullish_mss_retest"] +

        # MSS 0.786 retracement
        df["bullish_mss_retracement"].astype(int) +

        # Reversal block mitigating the last swing low
        # df["bullish_rb_mitigation"] +

        # FVG + MB/RB confluence
        (
            df["bullish_fvg_retracement"].shift(1) &
            (df["bullish_mb"] | df["bullish_rb"])
        ).astype(int) +

        # FVG retracement (standalone)
        # df["bullish_fvg_retracement"].astype(int) +

        # Liquidity
        ((df["asia_high_dist"] >= 0) & (df["asia_high_dist"] <= 3)).astype(int) +
        ((df["asia_low_dist"] >= -3) & (df["asia_low_dist"] <= 0)).astype(int) +
        ((df["pdh_dist"] >= 0) & (df["pdh_dist"] <= 3)).astype(int) +
        ((df["pdl_dist"] >= -3) & (df["pdl_dist"] <= 0)).astype(int) +

        # Equal lows
        df["eql"] +

        # Equal highs retest
        df["eqh_retest"].astype(int) +

        # Fib retracement from the swing high: 0.236 (5% tolerance), else
        # 0.786 — only while in a bullish trend.
        # (
        #     (
        #         df["bullish_fib236"].astype(bool)
        #         | df["bullish_fib786"].astype(bool)
        #     )
        #     & df["bullish_trend"].astype(bool)
        # ).astype(int) +

        # Not capped by a descending resistance line
        (~df["below_resistance"].astype(bool)).astype(int) +
        # df["above_vwap"] +
        # df["vwap_above_upper"] +
        (df["vwap_slope"] > 0) +
        (df["killzone"].isin([1, 2, 3])).astype(int)
    )

    return score

def SellScore(df):
    """See BuyScore() — same confluence terms, mirrored bearish side."""
    score = (
        # Trend filter
        (df["EMA7"] < df["EMA21"]).astype(int) +
        (df["EMA721_DIFF"] < 0).astype(int) +
        (df["EMA7_Slope"] < 0).astype(int) +
        # (df["EMA21"] < df["EMA50"]).astype(int) +
        # (df["EMA2150_DIFF"] < 0).astype(int) +
        (df["EMA21_Slope"] < 0).astype(int) +
        # (df["EMA50"] < df["EMA200"]).astype(int) +
        # (df["EMA50200_DIFF"] < 0).astype(int) +
        # (df["EMA50_Slope"] < 0).astype(int) +
        # (df["EMA200_Slope"] < 0).astype(int) +
        (df["-di"] > df["+di"]).astype(int) +
        (df["adx"] > 20).astype(int) +
        (df["k"] < df["k_smooth"]).astype(int) +

        # OB mitigation
        df["bearish_ob_mitigation"] +

        # Breaker Block
        df["bearish_bb"] +

        df["bearish_rb"] +

        df["bearish_mb"] +

        df["bearish_ifvg"] +

        (df["bullish"].shift(1) &
        df["exhaustion"]) + 

        df["bearish_ob"] +

        # MSS
        # (df["bearish_mss"] &
        # (df["bearish_trend"])
        # ).astype(int) +

        # MSS Retest
        df["bearish_mss_retest"] +

        # MSS 0.786 retracement
        df["bearish_mss_retracement"].astype(int) +

        df["bearish_choch"] +

        # Reversal block mitigating the last swing high
        # df["bearish_rb_mitigation"] +

        # FVG + MB/RB confluence
        (
            df["bearish_fvg_retracement"].shift(1) &
            (df["bearish_mb"] | df["bearish_rb"])
        ).astype(int) +

        # FVG retracement (standalone)
        # df["bearish_fvg_retracement"].astype(int) +

        # Liquidity
        ((df["asia_high_dist"] >= -3) & (df["asia_high_dist"] <= 0)).astype(int) +
        ((df["asia_low_dist"] >= 0) & (df["asia_low_dist"] <= 3)).astype(int) +
        ((df["pdh_dist"] >= -3) & (df["pdh_dist"] <= 0)).astype(int) +
        ((df["pdl_dist"] >= 0) & (df["pdl_dist"] <= 3)).astype(int) +

        # Equal highs
        df["eqh"] +

        # Equal lows retest
        df["eql_retest"].astype(int) +

        # Fib retracement from the swing low: 0.236 (5% tolerance), else
        # 0.786 — only while in a bearish trend.
        # (
        #     (
        #         df["bearish_fib236"].astype(bool)
        #         | df["bearish_fib786"].astype(bool)
        #     )
        #     & df["bearish_trend"].astype(bool)
        # ).astype(int) +

        # Not held by an ascending support line
        (~df["above_support"].astype(bool)).astype(int) +
        # df["below_vwap"] +
        # df["vwap_below_lower"] +
        (df["vwap_slope"] < 0) +
        (df["killzone"].isin([1, 2, 3])).astype(int)
    )

    return score

def add_indicators(df, include_higher_tf=True):
    # print("Loading indicators")

    if include_higher_tf:
        raw_1m = df.copy()

    df['adx'], df['+di'], df['-di'] = ADX(df)

    df['k'], df['k_smooth'] = STOCH(df)

    df['EMA7'] = EMA(df, 7)
    # df['EMA1'] = EMA(df, 1)
    # df['EMA1_Slope'] = df['EMA1'].diff()
    df['EMA7_Slope'] = df['EMA7'].diff()
    df['EMA21'] = EMA(df, 21)
    df['EMA21_Slope'] = df['EMA21'].diff()
    df['EMA721_DIFF'] = df['EMA7'] - df['EMA21']
    df['EMA7_DIST'] = df['Close'] - df['EMA7']
    # df['EMA50'] = EMA(df, 50)
    # df['EMA2150_DIFF'] = df['EMA21'] - df['EMA50']
    # df['EMA50_Slope'] = df['EMA50'].diff()
    # df['EMA200'] = EMA(df, 200)
    # df['EMA50200_DIFF'] = df['EMA50'] - df['EMA200']
    # df['EMA200_Slope'] = df['EMA200'].diff()
    # df['EMA1'] = EMA(df, 1)
    # df['EMA71_DIFF'] = df['EMA1'] - df['EMA7']

    # df["vwap"], df["vwap_upper"], df["vwap_lower"], df["above_vwap"], df["below_vwap"], df["vwap_above_upper"], df["vwap_below_lower"], df["vwap_slope"] = VWAP(df)
    _, _, _, df["above_vwap"], df["below_vwap"], df["vwap_above_upper"], df["vwap_below_lower"], df["vwap_slope"] = VWAP(df)

    df["range"] = GetRange(df)
    df["range_ma"] = RangeMA(df)

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

    df["swing_high"], df["swing_low"] = Swings(df)

    df["eqh"], df["eql"] = EQH_EQL(df)
    df["eqh_retest"], df["eql_retest"] = EQHEQLRetest(df)

    df["bearish_mb"] = BearishMB(df)
    df["bullish_mb"] = BullishMB(df)

    df["bullish_rb"], df["bearish_rb"] = RejectionBlocks(df)

    df["bullish_mss"], df["bearish_mss"] = MSS(df)
    df["bullish_mss_retest"], df["bearish_mss_retest"] = MSS_Retest(df)
    df["bullish_mss_retracement"], df["bearish_mss_retracement"] = MSSFibRetracement(df)

    df["bullish_bos"], df["bearish_bos"] = BoS(df)
    df["bullish_trend"], df["bearish_trend"] = Trend(df)
    df["bullish_choch"], df["bearish_choch"] = CHoCH(df)

    df["bullish_bb"] = BullishBB(df)
    df["bearish_bb"] = BearishBB(df)

    df["exhaustion"] = ExhaustionCandle(df)

    df["above_support"], df["below_resistance"] = TrendLines(df)

    df["bullish_fib236"], df["bullish_fib382"], df["bullish_fib500"], df["bullish_fib618"], df["bullish_fib786"], df["bearish_fib236"], df["bearish_fib382"], df["bearish_fib500"], df["bearish_fib618"], df["bearish_fib786"] = Fibonacci(df)

    df = add_irl_erl(df)

    df["killzone"] = GetKillzone(df)

    df["asia_high_dist"] = AsiaHighDistance(df)
    df["asia_low_dist"] = AsiaLowDistance(df)

    df["pdh_dist"] = PDHDistance(df)
    df["pdl_dist"] = PDLDistance(df)

    df["pd_poc_dist"] = PDPOCDistance(df)

    df["sell_score"] = SellScore(df)
    df["buy_score"] = BuyScore(df)

    df = df[["Open", "High", "Low", "Close",
            "k", "k_smooth", "adx", "+di", "-di", "EMA7", "EMA21", "EMA721_DIFF", "EMA7_DIST", "EMA7_Slope", "EMA21_Slope",
            # "EMA50", "EMA200", "EMA2150_DIFF", "EMA50_Slope", "EMA50200_DIFF", "EMA200_Slope",
            "indecision", "bullish_ob", "bearish_ob", "bullish_fvg", "bearish_fvg", "bullish_ifvg", "bearish_ifvg", "eqh", "eql", "bearish_mb", "bullish_mb","bullish_rb", "bearish_rb", "bullish_bb", "bearish_bb",
            "bullish_ob_mitigation", "bearish_ob_mitigation", "bullish_fvg_retracement", "bearish_fvg_retracement", "eqh_retest", "eql_retest",
            "swing_high", "swing_low", "bullish_mss", "bearish_mss", "bullish_mss_retest", "bearish_mss_retest", "bullish_mss_retracement", "bearish_mss_retracement",
            "bullish_bos", "bearish_bos", "bullish_trend", "bearish_trend", "bullish_choch", "bearish_choch",
            "range_high", "range_low", "erl_high", "erl_low", "irl",
            "exhaustion",
            "above_support", "below_resistance",
            "bullish_fib236", "bullish_fib382", "bullish_fib500", "bullish_fib618", "bullish_fib786", "bearish_fib236", "bearish_fib382", "bearish_fib500", "bearish_fib618", "bearish_fib786",
            "above_vwap", "below_vwap", "vwap_above_upper", "vwap_below_lower", "vwap_slope",
            "range_ma", "range",
            "bullish", "bearish",
            "asia_high_dist", "asia_low_dist",
            "pdh_dist", "pdl_dist",
            "pd_poc_dist",
            "killzone",
            "sell_score", "buy_score",
            ]].copy()

    # ==================================================================
    # HIGHER TIMEFRAME CANDLES, AGGREGATED FROM 1m AND MERGED BACK ONTO
    # EACH 1m ROW
    # ==================================================================
    # Right-labeled/left-closed resample so e.g. a 15m bin covering
    # [10:00, 10:15) is labeled 10:15 — i.e. it only becomes visible once
    # that candle has actually closed. reindex+ffill then means every 1m
    # bar only ever sees the most recently CLOSED higher-tf candle's
    # indicators, never the one still forming around it (no lookahead).
    if include_higher_tf:

        volume_col = (
            "Volume" if "Volume" in raw_1m.columns else
            "tick_volume" if "tick_volume" in raw_1m.columns else
            "real_volume"
        )

        for prefix, freq in (("15m", "15min"),):

            df_tf = raw_1m.resample(
                freq, label="right", closed="left"
            ).agg({
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                volume_col: "sum",
            }).dropna()

            if volume_col != "Volume":
                df_tf = df_tf.rename(columns={volume_col: "Volume"})

            df_tf = add_indicators(df_tf, include_higher_tf=False)
            df_tf = df_tf.add_prefix(f"{prefix}_")

            df = pd.concat(
                [df, df_tf.reindex(df.index, method="ffill")],
                axis=1
            )

    # print("Loaded indicators")

    df.dropna(inplace=True)
    return df

def fetch_dxy_history(cache_dir="download", max_age_hours=20):
    """
    Daily DXY (ICE US Dollar Index) close prices from Yahoo Finance,
    cached locally so training/live runs don't hit the network every time.
    Returns a pandas Series of closes indexed by (normalized, tz-naive) date.
    """
    cache_path = os.path.join(cache_dir, "dxy_history.csv")

    if os.path.exists(cache_path):
        age_hours = (time.time() - os.path.getmtime(cache_path)) / 3600
        if age_hours < max_age_hours:
            cached = pd.read_csv(cache_path, parse_dates=["date"], index_col="date")
            return cached["dxy"]

    url = "https://query1.finance.yahoo.com/v8/finance/chart/DX-Y.NYB?range=25y&interval=1d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    result = payload["chart"]["result"][0]
    dates = pd.to_datetime(result["timestamp"], unit="s", utc=True).normalize().tz_localize(None)
    closes = result["indicators"]["quote"][0]["close"]

    series = pd.Series(closes, index=dates, name="dxy").dropna()
    series = series[~series.index.duplicated(keep="last")].sort_index()

    os.makedirs(cache_dir, exist_ok=True)
    series.to_frame(name="dxy").rename_axis("date").to_csv(cache_path)

    return series

def fetch_real_yield_history(cache_dir="download", max_age_hours=20):
    """
    Daily 10-year real (TIPS) yield from FRED (series DFII10), cached
    locally the same way as fetch_dxy_history().
    """
    cache_path = os.path.join(cache_dir, "real_yield_history.csv")

    if os.path.exists(cache_path):
        age_hours = (time.time() - os.path.getmtime(cache_path)) / 3600
        if age_hours < max_age_hours:
            cached = pd.read_csv(cache_path, parse_dates=["date"], index_col="date")
            return cached["real_yield"]

    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFII10"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        raw = pd.read_csv(resp)

    raw.columns = ["date", "real_yield"]
    raw["date"] = pd.to_datetime(raw["date"])
    raw["real_yield"] = pd.to_numeric(raw["real_yield"], errors="coerce")
    raw = raw.dropna().set_index("date").sort_index()

    os.makedirs(cache_dir, exist_ok=True)
    raw.rename_axis("date").to_csv(cache_path)

    return raw["real_yield"]

def fetch_gld_holdings_history(cache_dir="download", max_age_hours=20):
    """
    Daily total GLD trust holdings in tonnes, from SPDR's own historical
    archive API. This is total physical gold held by the trust, which only
    moves via actual authorized-participant creation/redemption — unlike
    "ounces per share", it has no slow expense-ratio drift to de-trend, so
    the raw day-over-day change is already a clean net-flow signal.
    """
    cache_path = os.path.join(cache_dir, "gld_holdings_history.csv")

    if os.path.exists(cache_path):
        age_hours = (time.time() - os.path.getmtime(cache_path)) / 3600
        if age_hours < max_age_hours:
            cached = pd.read_csv(cache_path, parse_dates=["date"], index_col="date")
            return cached["tonnes"]

    url = "https://api.spdrgoldshares.com/api/v1/historical-archive?product=gld&exchange=NYSE&lang=en"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = resp.read()

    raw = pd.read_excel(BytesIO(data), sheet_name="US GLD Historical Archive")
    raw = raw.rename(columns={raw.columns[0]: "date", "Tonnes of Gold": "tonnes"})
    raw["date"] = pd.to_datetime(raw["date"], format="%d-%b-%Y", errors="coerce")
    raw["tonnes"] = pd.to_numeric(raw["tonnes"], errors="coerce")
    raw = raw.dropna(subset=["date", "tonnes"]).set_index("date").sort_index()

    os.makedirs(cache_dir, exist_ok=True)
    raw[["tonnes"]].rename_axis("date").to_csv(cache_path)

    return raw["tonnes"]

def add_macro_features(df, cache_dir="download", zscore_window=60):
    """
    Adds dxy_zscore / real_yield_zscore / gld_flow_zscore columns to df (an
    M5-indexed OHLCV+indicators frame). All three are daily macro series:
    z-scored on their own daily cadence (rolling zscore_window trading days)
    BEFORE being merged onto the M5 index, then shifted by one trading day
    so every bar only ever sees the PRIOR day's published value — never
    today's, which wouldn't actually be known yet at trade time. Missing/
    pre-history values default to 0.0 (neutral) rather than NaN.
    """
    dxy = fetch_dxy_history(cache_dir)
    real_yield = fetch_real_yield_history(cache_dir)
    gld_tonnes = fetch_gld_holdings_history(cache_dir)

    def zscore(s, window):
        mean = s.rolling(window).mean()
        std = s.rolling(window).std()
        return (s - mean) / std.replace(0, np.nan)

    dxy_z = zscore(dxy, zscore_window).shift(1)                    # yesterday's z-score only
    yield_z = zscore(real_yield, zscore_window).shift(1)
    gld_flow_z = zscore(gld_tonnes.diff(), zscore_window).shift(1) # z-score of the day-over-day tonnage change

    day_key = df.index.tz_convert(None).normalize() if df.index.tz is not None else df.index.normalize()

    df["dxy_zscore"] = day_key.map(dxy_z).astype(np.float32)
    df["real_yield_zscore"] = day_key.map(yield_z).astype(np.float32)
    df["gld_flow_zscore"] = day_key.map(gld_flow_z).astype(np.float32)

    macro_cols = ["dxy_zscore", "real_yield_zscore", "gld_flow_zscore"]
    df[macro_cols] = df[macro_cols].ffill().fillna(0.0)

    return df

class PPOLSTMNetwork(nn.Module):
    def __init__(self, state_size=12, hidden_size=64, action_size=3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=state_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=2
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

class WinRateModel:
    """
    Win-rate filter backed by XGBoost. Every closed trade's entry-time
    feature vector is stored along with whether it won, and periodically
    refit into an XGBClassifier. Before that, predict_win_rate() estimates
    a new setup's win probability — a trade is only worth taking if that
    estimate clears the breakeven win rate implied by the strategy's
    risk:reward (1 / (1 + RR)).

    Persisted as raw (X, y) samples rather than a pickled fitted model, so
    more samples can keep being appended and refit from run to run without
    depending on a specific xgboost version's pickle format. At 200-500
    trades/week this accumulates well past the point where a shallow,
    regularized boosted-tree fit is overfitting noise.
    """

    def __init__(self, symbol, min_samples=30):
        self.symbol = symbol
        self.min_samples = min_samples
        self.model = None
        self.X = []
        self.y = []

    def add_sample(self, state, won):
        self.X.append(np.asarray(state, dtype=np.float32))
        self.y.append(1 if won else 0)

    def fit(self):
        # Need both classes present, and enough rows that a shallow tree
        # ensemble isn't just memorizing individual trades.
        if len(self.X) < self.min_samples or len(set(self.y)) < 2:
            return False

        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42
        )
        self.model.fit(np.array(self.X), np.array(self.y))
        return True

    def predict_win_rate(self, state):
        if self.model is None:
            return None

        proba = self.model.predict_proba(
            np.asarray(state, dtype=np.float32).reshape(1, -1)
        )[0]

        classes = list(self.model.classes_)
        return proba[classes.index(1)] if 1 in classes else 0.0

    def allows(self, state, min_winrate):
        # No trained model yet (no file on disk, and this run hasn't
        # accumulated a first week of trades) — don't filter anything.
        if self.model is None:
            return True

        return self.predict_win_rate(state) >= min_winrate

    def _path(self):
        # Algorithm-agnostic name — WinRateModel's underlying model is
        # meant to be swappable, so the filename shouldn't hardcode
        # whatever algorithm happens to back it today.
        return os.path.join("LSTM-PPO-saves", f"{self.symbol}.winrate.pkl")

    def save(self):
        os.makedirs("LSTM-PPO-saves", exist_ok=True)
        with open(self._path(), "wb") as f:
            pickle.dump({"X": self.X, "y": self.y}, f)

    def load(self):
        path = self._path()
        if not os.path.exists(path):
            return False

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.X = data["X"]
        self.y = data["y"]

        self.fit()
        return True

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

def streak_stats(returns):
    """
    Average length of consecutive-win and consecutive-loss runs in
    `returns` (in the order given). A zero return breaks the current
    streak without starting a new one.
    """
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

def train_bot(symbol="XAUUSD"):

    print("Training bot")
    start = time.perf_counter()

    # load_last_mb_xauusd()'s default (mb=8) only loads the last ~22
    # weeks of XAU_1m_data.csv, not the full multi-year file — bumped up
    # here for meaningfully more history. add_indicators() runs once on
    # the whole loaded frame at startup (not per-week), so this trades a
    # longer one-time startup cost for more data per training pass;
    # raise/lower TRAIN_HISTORY_MB to trade off further.
    TRAIN_HISTORY_MB = 20
    df = load_last_mb_xauusd(mb=TRAIN_HISTORY_MB)
    print(f"Loading indicators... ({time.strftime('%H:%M')})")
    df = add_indicators(df)
    elapsed = int((time.perf_counter() - start) // 60)
    print(f"Loaded indicators (Elapsed: {elapsed}m)")

    print("Loading macro features (DXY, real yields, GLD)...")
    df = add_macro_features(df)
    print("Loaded macro features")

    SEQ_LEN = 15

    save_count = 1440 * 5

    # Every column add_indicators() produces per timeframe (excluding OHLC
    # itself) — also what gets "15m_"-prefixed for the merged higher-tf candle
    # indicators, since add_indicators() runs this same pipeline on the
    # resampled 15m data and prefixes its output with "15m_".
    BASE_INDICATOR_FEATURES = [
        "k", "k_smooth", "adx", "+di", "-di",
        "EMA7", "EMA21", "EMA721_DIFF", "EMA7_DIST", "EMA7_Slope", "EMA21_Slope",
        # "EMA50", "EMA200", "EMA2150_DIFF", "EMA50_Slope", "EMA50200_DIFF", "EMA200_Slope",
        "indecision", "bullish_ob", "bearish_ob", "bullish_fvg", "bearish_fvg", "bullish_ifvg", "bearish_ifvg", "eqh", "eql", "bearish_mb", "bullish_mb", "bullish_rb", "bearish_rb", "bullish_bb", "bearish_bb",
        "bullish_ob_mitigation", "bearish_ob_mitigation", "bullish_fvg_retracement", "bearish_fvg_retracement", "eqh_retest", "eql_retest",
        "swing_high", "swing_low", "bullish_mss", "bearish_mss", "bullish_mss_retest", "bearish_mss_retest", "bullish_mss_retracement", "bearish_mss_retracement",
        "bullish_bos", "bearish_bos", "bullish_trend", "bearish_trend", "bullish_choch", "bearish_choch",
        "range_high", "range_low", "erl_high", "erl_low", "irl",
        "exhaustion",
        "above_support", "below_resistance",
        "bullish_fib236", "bullish_fib382", "bullish_fib500", "bullish_fib618", "bullish_fib786", "bearish_fib236", "bearish_fib382", "bearish_fib500", "bearish_fib618", "bearish_fib786",
        "above_vwap", "below_vwap", "vwap_above_upper", "vwap_below_lower", "vwap_slope",
        "range_ma", "range",
        "bullish", "bearish",
        "asia_high_dist", "asia_low_dist",
        "pdh_dist", "pdl_dist",
        "pd_poc_dist",
        "killzone",
        "sell_score", "buy_score",
    ]

    FEATURES = (
        BASE_INDICATOR_FEATURES
        + ["15m_" + f for f in BASE_INDICATOR_FEATURES]
        + ["dxy_zscore", "real_yield_zscore", "gld_flow_zscore"]
    )

    agent = LSTMPPOAgent(
        state_size=len(FEATURES),
        hidden_size=64,
        action_size=3
    )

    winrate_model = WinRateModel(symbol)
    try:
        agent.loadcheckpoint(symbol)
        winrate_model.load()
        print(f"[{symbol}] Loaded checkpoint")
    except Exception as e:
        print(f"[{symbol}] Starting fresh ({e})")

    save_counter = 0

    in_position = False
    position_type = None

    entry_price = 0
    sl_price = 0
    tp_price = 0

    trade_returns = []

    SPREAD_AND_COMMISSION = 0.6

    # No fixed take-profit anymore — a trade rides until its trailing
    # stop is hit. TRAIL_PIPS is how far behind the best price reached
    # the stop trails; it only ratchets forward once price has moved at
    # least TRAIL_STEP_PIPS further since the last time it moved.

    sl_pips = 40

    # Breakeven win rate implied by the fixed RR below (tp is sl_pips/2,
    # so RR = 0.5) — a setup only clears the KNN filter once its
    # predicted win rate beats what's needed to be profitable at this RR.
    RR_RATIO = (sl_pips / 2) / sl_pips
    MIN_WINRATE = 1 / (1 + RR_RATIO)

    PIP_VALUE = 0.1

    state_buffer = deque(maxlen=SEQ_LEN)

    training_start_2 = time.time()
    training_start_3 = time.time()

    # preload sequence
    # for i in range(SEQ_LEN):
    #     row = df.iloc[i][FEATURES].values.astype(np.float32)
    #     state_buffer.append(row)
    # df["range"] = (df["High"] - df["Low"]) / PIP_VALUE

    loop_count = 0
    week_start_i = None
    week_features = None

    try:
        while True:
            loop_count += 1
            print(f"[{symbol}] [INFO] Starting pass {loop_count} over dataset")

            for i in range(SEQ_LEN, len(df)):

                # Preload this week's state_seq rows as one vectorized
                # slice instead of hitting df.iloc[i] per bar, whenever i
                # has moved outside the currently cached slice — not just
                # on save_counter's own weekly cadence, since save_counter
                # keeps counting across "while True" passes over the
                # dataset while i resets to SEQ_LEN at the start of each
                # new pass (using save_counter alone left week_start_i
                # stale across that reset, indexing far out of bounds).
                if (
                    week_start_i is None
                    or i < week_start_i
                    or i >= week_start_i + len(week_features)
                ):
                    week_features = df[FEATURES].iloc[i:i + save_count].values.astype(np.float32)
                    week_start_i = i

                current = df.iloc[i]

                current_price = current["Close"]
                high = current["High"]
                low = current["Low"]

                state = week_features[i - week_start_i]

                state_buffer.append(state)

                if len(state_buffer) < SEQ_LEN:
                    continue

                state_seq = np.array(state_buffer)

                # === Select action ============================================
                result = agent.select_action(state_seq, in_position, training=True)

                if result is None:
                    continue

                # action, logprob, value, rr_mult, sl_mult = result
                action, logprob, value = result

                # XGBoost win-rate filter: block a fresh entry if the setup's
                # predicted win rate doesn't clear breakeven for this RR.
                # allows() is a no-op (always True) until the model has
                # its first fit, so early training is unaffected.
                if action in (1, 2) and not winrate_model.allows(state, MIN_WINRATE):
                    action = 0

                pnl = 0.0
                done = False

                # ==============================================================
                # OPEN LONG
                # ==============================================================

                if action == 1:
                    in_position = True
                    position_type = "long"

                    entry_price = current_price
                    winrate_entry_state = state.copy()

                    sl_price = entry_price - sl_pips * PIP_VALUE
                    tp_price = entry_price + (sl_pips / 2) * PIP_VALUE

                    pnl = 0.0
                    done = False

                    agent.store_transition(
                        state_seq,
                        action,
                        logprob,
                        value,
                        pnl,
                        done
                    )

                    tp_hit = False

                # ==============================================================
                # OPEN SHORT
                # ==============================================================

                elif action == 2:
                    in_position = True
                    position_type = "short"

                    entry_price = current_price
                    winrate_entry_state = state.copy()

                    sl_price = entry_price + sl_pips * PIP_VALUE
                    tp_price = entry_price - (sl_pips / 2) * PIP_VALUE

                    pnl = 0.0
                    done = False

                    agent.store_transition(
                        state_seq,
                        action,
                        logprob,
                        value,
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

                        if high >= tp_price:
                            pnl = (tp_price - entry_price) / PIP_VALUE - SPREAD_AND_COMMISSION
                            trade_closed = True

                        if low <= sl_price:
                            # print(f"sl hit")
                            pnl = (sl_price - entry_price) / PIP_VALUE - SPREAD_AND_COMMISSION
                            trade_closed = True

                    # ==================================================
                    # SHORT
                    # ==================================================

                    elif position_type == "short":

                        if low <= tp_price:
                            pnl = (entry_price - tp_price) / PIP_VALUE - SPREAD_AND_COMMISSION
                            trade_closed = True

                        if high >= sl_price:
                            # print(f"sl hit")
                            pnl = (entry_price - sl_price) / PIP_VALUE - SPREAD_AND_COMMISSION
                            trade_closed = True

                    if trade_closed:

                        in_position = False
                        done = True
                        trade_returns.append(pnl)
                        winrate_model.add_sample(winrate_entry_state, pnl > 0)

                        agent.store_transition(
                            state_seq,
                            action,
                            logprob,
                            value,
                            pnl,
                            done
                        )

                        pnl = 0

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

                        std_ret = np.std(trade_returns)
                        zscore = np.mean(trade_returns) / std_ret if std_ret > 0 else 0.0

                        avg_win_streak, avg_loss_streak = streak_stats(trade_returns)

                        gross_profit = sum(wins)
                        gross_loss = abs(sum(losses))
                        profit_factor = (
                            gross_profit / gross_loss
                            if gross_loss > 0
                            else float("inf")
                        )
                        # avg_rr = sum(rr_list) / len(rr_list) if rr_list else 0
                        # avg_sl = sum(sl_list) / len(sl_list) if sl_list else 0
                        max_dd = max_drawdown(trade_returns)
                        R_pnl = weekly_pnl / sl_pips
                        # rf = (
                            # (len(trade_returns) * winrate * RR_RATIO)
                            # - (len(trade_returns) * (1 - winrate))
                        # ) / (max_dd/SL_PIPS)
                        #     R_pnl/(max_dd/avg_sl)
                        # )
                        rf = 0.0
                        if max_dd > 0 and sl_pips > 0:
                            rf = R_pnl / (max_dd / (sl_pips / 10))
                        else:
                            rf = 0

                        print()
                        print("================================================")
                        print(f"[{symbol}] WEEKLY PPO TRAINING")
                        print("================================================")
                        print(f"Trades:          {len(trade_returns)}")
                        print(f"Weekly PnL:      {weekly_pnl:.0f} pips")
                        print(f"Weekly R PnL:    {R_pnl:.2f}R")
                        print(f"Max DD:          {max_dd/sl_pips:.2f}R")
                        print(f"Winrate:         {winrate*100:.2f}%")
                        print(f"Mean Win:        {mean_win:.0f} pips")
                        print(f"Mean Loss:       {mean_loss:.0f} pips")
                        print(f"Avg Win Streak:  {avg_win_streak:.2f}")
                        print(f"Avg Loss Streak: {avg_loss_streak:.2f}")
                        print(f"Z-score:         {zscore:.2f}")
                        print(f"PF:              {profit_factor:.2f}")
                        print(f"RF:              {rf:.2f}")
                        # print(f"RF:              {round(rf, 0):.0f}")
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

                        # train() only clears the trajectory when it had >=32
                        # samples to train on; force a clear every week
                        # regardless, so a short week can't leak stale
                        # trajectory into the next one.
                        agent.trajectory.clear()

                        print(
                            f"[{symbol}] "
                            f"[INFO] Finished training PPO "
                            f"(Elapsed: {timedelta(seconds=int(time.time() - training_start))})"
                        )

                        # if save_counter % (1440 * 4) == 0:
                        agent.savecheckpoint(symbol)

                        if winrate_model.fit():
                            winrate_model.save()
                            print(f"[{symbol}] [INFO] XGBoost win-rate filter refit on {len(winrate_model.X)} trades and saved")

                        completed = int((save_counter / save_count))
                        total = int(round(len(df) / save_count, 0))

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

    except KeyboardInterrupt:
        print(f"[{symbol}] [INFO] KeyboardInterrupt received, stopping after {loop_count} pass(es) over the dataset")


    # ==============================================================
    # FINAL TRAINING
    # ==============================================================

    agent.train()
    agent.savecheckpoint(symbol)

    if winrate_model.fit():
        winrate_model.save()

    print(
        f"[{symbol}] "
        f"[INFO] Finished training "
    )

    return agent

def open_long(symbol, lot_size, sl_pips):
    """No take-profit — the position rides until trail_stop_loss() moves
    its SL into a fill, or close_trades() closes it."""

    tick = mt5.symbol_info_tick(symbol)

    entry = tick.ask

    sl = entry - sl_pips / 10 + 0.2
    tp = entry + (sl_pips / 2) / 10 + 0.2

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

def open_short(symbol, lot_size, sl_pips):
    """No take-profit — the position rides until trail_stop_loss() moves
    its SL into a fill, or close_trades() closes it."""

    tick = mt5.symbol_info_tick(symbol)

    entry = tick.bid

    sl = entry + sl_pips / 10 - 0.2

    tp = entry - (sl_pips / 2) / 10 - 0.2

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

def get_today_profit(symbol, day, now=None, magic=123456):
    """Net profit (profit + swap + commission) of every deal this EA has
    closed today, from MT5's own history — used for a daily profit cap."""
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
        # else:
        #     print(f"Closed {pos.ticket}")

def test_bot(symbol="XAUUSD"):
    SEQ_LEN = 15
    
    mt5.initialize()
    account = mt5.account_info()
    balance = account.balance

    # No fixed take-profit — trail_stop_loss() ratchets the SL once price
    # has moved TRAIL_PIPS in favor, in TRAIL_STEP_PIPS increments.
    sl_pips = 40
    RISK = 0.0025
    profit_target = RISK * 100 * 25

    # Same breakeven win rate the KNN filter was fit against during
    # training (see train_bot) — RR there is the fixed sl_pips/2 tp vs.
    # sl_pips sl, even though live trades ride a trailing stop instead.
    RR_RATIO = (sl_pips / 2) / sl_pips
    MIN_WINRATE = 1 / (1 + RR_RATIO)

    # Every column add_indicators() produces per timeframe (excluding OHLC
    # itself) — also what gets "15m_"-prefixed for the merged higher-tf candle
    # indicators, since add_indicators() runs this same pipeline on the
    # resampled 15m data and prefixes its output with "15m_".
    BASE_INDICATOR_FEATURES = [
        "k", "k_smooth", "adx", "+di", "-di",
        "EMA7", "EMA21", "EMA721_DIFF", "EMA7_DIST", "EMA7_Slope", "EMA21_Slope",
        # "EMA50", "EMA200", "EMA2150_DIFF", "EMA50_Slope", "EMA50200_DIFF", "EMA200_Slope",
        "indecision", "bullish_ob", "bearish_ob", "bullish_fvg", "bearish_fvg", "bullish_ifvg", "bearish_ifvg", "eqh", "eql", "bearish_mb", "bullish_mb", "bullish_rb", "bearish_rb", "bullish_bb", "bearish_bb",
        "bullish_ob_mitigation", "bearish_ob_mitigation", "bullish_fvg_retracement", "bearish_fvg_retracement", "eqh_retest", "eql_retest",
        "swing_high", "swing_low", "bullish_mss", "bearish_mss", "bullish_mss_retest", "bearish_mss_retest", "bullish_mss_retracement", "bearish_mss_retracement",
        "bullish_bos", "bearish_bos", "bullish_trend", "bearish_trend", "bullish_choch", "bearish_choch",
        "range_high", "range_low", "erl_high", "erl_low", "irl",
        "exhaustion",
        "above_support", "below_resistance",
        "bullish_fib236", "bullish_fib382", "bullish_fib500", "bullish_fib618", "bullish_fib786", "bearish_fib236", "bearish_fib382", "bearish_fib500", "bearish_fib618", "bearish_fib786",
        "above_vwap", "below_vwap", "vwap_above_upper", "vwap_below_lower", "vwap_slope",
        "range_ma", "range",
        "bullish", "bearish",
        "asia_high_dist", "asia_low_dist",
        "pdh_dist", "pdl_dist",
        "pd_poc_dist",
        "killzone",
        "sell_score", "buy_score",
    ]

    FEATURES = (
        BASE_INDICATOR_FEATURES
        + ["15m_" + f for f in BASE_INDICATOR_FEATURES]
        + ["dxy_zscore", "real_yield_zscore", "gld_flow_zscore"]
    )

    last_m1 = None

    agent = LSTMPPOAgent(
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

    winrate_model = WinRateModel(symbol)
    winrate_model.load()

    # ==========================================================
    # INITIAL LOAD
    # ==========================================================

    # The rolling live window needs enough 1m bars that resampling up to
    # 15m (the only higher timeframe merged in) still leaves a
    # comfortable margin over the largest indicator lookback (72 bars)
    # — 3000 1m bars ≈ 200 15m bars ≈ 2.08 days. Too small a window here
    # would make the 15m branch come back empty and silently wipe out
    # every row via add_indicators()'s dropna().
    LIVE_HISTORY_BARS = 600 * 5

    rates_m1 = mt5.copy_rates_from_pos(
        symbol,
        mt5.TIMEFRAME_M1,
        0,
        LIVE_HISTORY_BARS
    )

    df = pd.DataFrame(rates_m1)

    df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'time': 'Date',
        "tick_volume": "Volume"
    }, inplace=True)

    df["Date"] = (
    pd.to_datetime(df["Date"], unit="s", utc=True)
    )
    df.set_index("Date", inplace=True)

    raw_df = df

    df = add_indicators(df)
    df = add_macro_features(df)

    last_m1 = df.index[-1]
    last_trading_date = df.index[-1].date()

    # Balance snapshot at the start of each trading day, used to define
    # 1R for the daily profit cap below (R = day_start_balance * RISK).
    day_start_balance = mt5.account_info().balance

    # ==========================================================
    # MAIN LOOP
    # ==========================================================

    # print("entering main loop in test function")
    while True:

        now = datetime.now()

        seconds_until_next_minute = (
            60
            - now.second
            - now.microsecond / 1_000_000
        )
        if seconds_until_next_minute <= 0:
            seconds_until_next_minute += 0.25

        time.sleep(seconds_until_next_minute)

        tick = mt5.symbol_info_tick(symbol)

        # ======================================================
        # CHECK FOR NEW M15 CANDLE
        # ======================================================

        new_m1 = mt5.copy_rates_from_pos(
            symbol,
            mt5.TIMEFRAME_M1,
            0,
            1
        )

        current_m1 = new_m1[0]["time"]

        while True:
            if current_m1 != last_m1:
                break
            else:
                # time.sleep(0.25)
                new_m1 = mt5.copy_rates_from_pos(
                    symbol,
                    mt5.TIMEFRAME_M1,
                    0,
                    1
                )
                current_m1 = new_m1[0]["time"]

        if current_m1 != last_m1:

            last_m1 = current_m1

            # ==================================================
            # APPEND NEW CANDLE
            # ==================================================

            new_row = pd.DataFrame(new_m1)

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
            )
            new_row.set_index("Date", inplace=True)

            if new_row.index[-1] != df.index[-1]:

                raw_df = pd.concat(
                    [raw_df, new_row]
                )

                raw_df = raw_df.tail(LIVE_HISTORY_BARS)

                df = add_indicators(raw_df.copy())
                df = add_macro_features(df)  # cheap: fetch_*_history() cache for ~20h, no network hit every bar

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

            action, logprob, value = result 

            current_time = df.index[-1]  # or however you store timestamps

            # New trading day — reset the balance snapshot the daily
            # profit cap below is measured from.
            if current_time.date() != last_trading_date:
                last_trading_date = current_time.date()
                day_start_balance = mt5.account_info().balance

            if current_time.hour == 23 and current_time.minute >= 55:
                # Force close any open position
                if open_pos != 0:
                    close_trades()

                # Force HOLD
                action = 0

            # Daily profit cap: stop trading once today's realized profit
            # (from the EA's own closed trades, via get_today_profit() —
            # not an account-balance delta) reaches the target, where
            # 1R = day_start_balance * RISK and the target is profit_target R.
            today_profit = get_today_profit(symbol, last_trading_date, current_time)
            profit_target_pct = (today_profit / ((day_start_balance - today_profit) * RISK * profit_target)) * 100
            # print(f"profit target pct: {profit_target_pct:.2f}")
            # print(f"today profit: {today_profit}")
            # print(f"day start balance: {day_start_balance}")


            if profit_target_pct >= 100:
                if open_pos != 0:
                    close_trades()

                action = 0

            # print(f"Test action: {action}")

            # XGBoost win-rate filter: block a fresh entry if the setup's
            # predicted win rate doesn't clear breakeven for this RR.
            # allows() is a no-op (always True) until a winrate file exists.
            if action in (1, 2) and not winrate_model.allows(state_seq[-1], MIN_WINRATE):
                action = 0

            # ==================================================
            # OPEN NEW TRADE
            # ==================================================

            if open_pos == 0:

                account = mt5.account_info()

                balance = account.balance

                if action == 1:

                    # Lot size: hitting the initial SL costs RISK% of
                    # balance. There's no fixed TP anymore to size off of
                    # — the position rides on a trailing stop instead.
                    risk_per_position = min(
                        max((balance * RISK) / (sl_pips * 10), 0.01),
                        100.0
                    )
                    risk_per_position = round(risk_per_position, 2)

                    open_long(
                        symbol,
                        risk_per_position,
                        sl_pips
                    )

                elif action == 2:

                    # Lot size: hitting the initial SL costs RISK% of
                    # balance. There's no fixed TP anymore to size off of
                    # — the position rides on a trailing stop instead.
                    risk_per_position = min(
                        max((balance * RISK) / (sl_pips * 10), 0.01),
                        100.0
                    )
                    risk_per_position = round(risk_per_position, 2)

                    open_short(
                        symbol,
                        risk_per_position,
                        sl_pips
                    )

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
    parser.add_argument("--symbol", default="XAUUSD")

    args = parser.parse_args()

    # Separate OS processes, not threads: train_bot() and test_bot() are
    # both full of long-running Python for-loops (indicator computation,
    # bar-by-bar simulation), and threads share one GIL — so with two
    # threads, test_bot's once-a-minute indicator recompute can get
    # starved of execution time whenever it coincides with train_bot
    # mid-loop, causing an intermittent multi-second stall before an
    # order gets placed. Separate processes give each its own GIL.
    procs = []

    if args.train:
        p = multiprocessing.Process(
            target=train_bot,
            daemon=True
        )
        p.start()
        procs.append(p)

    if args.test:
        p = multiprocessing.Process(
            target=test_bot,
            args=(args.symbol,),
            daemon=True
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

if __name__ == "__main__":
    main()