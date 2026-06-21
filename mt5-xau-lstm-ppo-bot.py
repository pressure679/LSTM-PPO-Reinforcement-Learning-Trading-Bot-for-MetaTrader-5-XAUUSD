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
import  threading
import math
import glob
import MetaTrader5 as mt5
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# ready_event = threading.Event()
# process_counter = 0
# pd.set_option('future.no_silent_downcasting', True)

ACTIONS = ['hold', 'long', 'short', 'close']

def find_latest_dukascopy_csv():
    downloads = os.path.join(os.path.expanduser("~"), "downloads")

    pattern = os.path.join(downloads, "xauusd-m5-bid-*.csv")

    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError("No Dukascopy XAUUSD M5 CSV found in downloads.")

    return max(files, key=os.path.getmtime)

def load_last_mb_xauusd(file_path=None, mb=8, delimiter=',', col_names=None):
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
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

    # df = df.resample('15min').agg({
    #     'Open': 'first',
    #     'High': 'max',
    #     'Low': 'min',
    #     'Close': 'last'
    # }).dropna()
    
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

def EQH(df, tolerance=1):
    swing_high = (
        (df["High"] > df["High"].shift(1)) &
        (df["High"] > df["High"].shift(-1))
    )

    prev_swing_high = df["High"].where(swing_high).ffill().shift(1)

    return (
        swing_high &
        (abs(df["High"] - prev_swing_high) <= tolerance)
    ).astype(int)

def EQL(df, tolerance=1):
    swing_low = (
        (df["Low"] < df["Low"].shift(1)) &
        (df["Low"] < df["Low"].shift(-1))
    )

    prev_swing_low = df["Low"].where(swing_low).ffill().shift(1)

    return (
        swing_low &
        (abs(df["Low"] - prev_swing_low) <= tolerance)
    ).astype(int)

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

def BullishMB(df, multiplier=1.5):
    body = (df["Close"] - df["Open"]).abs()

    bearish = df["Close"] < df["Open"]
    next_bullish = df["Close"].shift(-1) > df["Open"].shift(-1)
    displacement = body.shift(-1) >= body * multiplier

    ob = bearish & next_bullish & displacement

    ob_high = df["High"].where(ob).ffill()
    ob_low = df["Low"].where(ob).ffill()

    return ((df["Low"] <= ob_high) &
            (df["High"] >= ob_low)).astype(int)

def BearishMB(df, multiplier=1.5):
    body = (df["Close"] - df["Open"]).abs()

    bullish = df["Close"] > df["Open"]
    next_bearish = df["Close"].shift(-1) < df["Open"].shift(-1)
    displacement = body.shift(-1) >= body * multiplier

    ob = bullish & next_bearish & displacement

    ob_high = df["High"].where(ob).ffill()
    ob_low = df["Low"].where(ob).ffill()

    return ((df["Low"] <= ob_high) &
            (df["High"] >= ob_low)).astype(int)

def AsiaHigh(df):
    # Asia session: 23:00-06:59 GMT
    asia = (df.index.hour >= 1) | (df.index.hour < 9)

    # Trading day starts at 23:00
    trade_day = (df.index - pd.Timedelta(hours=24)).date

    asia_high = (
        df["High"]
        .where(asia)
        .groupby(trade_day)
        .transform("max")
        .ffill()
    )

    return asia_high

def AsiaLow(df):
    asia = (df.index.hour >= 1) | (df.index.hour < 9)

    trade_day = (df.index - pd.Timedelta(hours=24)).date

    asia_low = (
        df["Low"]
        .where(asia)
        .groupby(trade_day)
        .transform("min")
        .ffill()
    )

    return asia_low

def PDH(df):
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

    return pdh

def PDL(df):
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

    return pdl

def VWAP(df, atr_period=14, atr_multiplier=1.0):

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

    session = df.index.normalize()

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

    upper = round(vwap + stddev * atr_multiplier, 2)
    lower = round(vwap - stddev * atr_multiplier, 2)

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
    elif "real_volume" in df.columns:
        volume = df["real_volume"]
    else:
        raise ValueError("No volume column found.")

    return round(volume.rolling(period).mean(), 2)

def BuyScore(df):

    return (
        (df["EMA7"] > df["EMA21"]).astype(int) * 2 +
        (df["EMA_DIFF"] > 0).astype(int) * 1 +
        (df["EMA7_Slope"] > 0).astype(int) * 1 +
        (df["+di"] > df["-di"]).astype(int) * 2 +
        (df["adx"] > 20).astype(int) * 1 +
        (df["k"] > df["k_smooth"]).astype(int) * 1 +
        (df["k"] < 20).astype(int) * 1 +
        df["bullish_ob"] * 2 +
        df["bullish_mb"] * 1 +
        df["bullish_fvg"] * 1 +
        df["eql"] * 1 +
        df["bullish_rb"] * 1 +
        df["above_vwap"] * 1 +
        df["vwap_above_upper"] * 1 -
        df["below_vwap"] * 1 -
        df["vwap_below_lower"] * 1 -
        df["bearish_rb"] * 1 -
        df["bearish_ob"] * 2 -
        df["bearish_fvg"] * 1 -
        df["eqh"] * 1 - 
        (df["k"] >= 80) * 1
    )

def SellScore(df):

    return (
        (df["EMA7"] < df["EMA21"]).astype(int) * 2 +
        (df["EMA_DIFF"] < 0).astype(int) * 1 +
        (df["EMA7_Slope"] < 0).astype(int) * 1 +
        (df["-di"] > df["+di"]).astype(int) * 2 +
        (df["adx"] > 20).astype(int) * 1 +
        (df["k"] < df["k_smooth"]).astype(int) * 1 +
        (df["k"] > 80).astype(int) * 1 +
        df["bearish_ob"] * 2 +
        df["bearish_mb"] * 1 +
        df["bearish_fvg"] * 1 +
        df["eqh"] * 1 +
        df["below_vwap"] * 1 +
        df["vwap_below_lower"] * 1 -
        df["above_vwap"] * 1 -
        df["vwap_above_upper"] * 1 -
        df["bearish_rb"] * 1 -
        df["bullish_rb"] * 1 -
        df["bullish_ob"] * 2 -
        df["bullish_fvg"] * 1 -
        df["eql"] * 1 -
        (df["k"] <= 20) * 1
    )

def add_indicators(df):
    df['adx'], df['+di'], df['-di'] = ADX(df)

    df['k'], df['k_smooth'] = STOCH(df)

    df['EMA7'] = EMA(df, 7)
    # df['EMA1'] = EMA(df, 1)
    # df['EMA1_Slope'] = df['EMA1'].diff()
    df['EMA7_Slope'] = df['EMA7'].diff()
    df['EMA21'] = EMA(df, 21)
    df['EMA_DIFF'] = df['EMA7'] - df['EMA21']

    df["indecision"] = Indecision(df)

    df["bullish_ob"] = BullishOB(df)
    df["bearish_ob"] = BearishOB(df)

    df["bullish_fvg"] = BullishFVG(df)
    df["bearish_fvg"] = BearishFVG(df)

    df["eqh"] = EQH(df)
    df["eql"] = EQL(df)

    df["bearish_mb"] = BearishMB(df)
    df["bullish_mb"] = BullishMB(df)

    df["bullish_rb"], df["bearish_rb"] = RejectionBlocks(df)

    df["vwap"], df["vwap_upper"], df["vwap_lower"], df["above_vwap"], df["below_vwap"], df["vwap_above_upper"], df["vwap_below_lower"], df["vwap_slope"] = VWAP(df)

    df["sell_score"] = SellScore(df)
    df["buy_score"] = BuyScore(df)

    df["volume_ma"] = VolumeMA(df)

    df["asia_high"] = AsiaHigh(df)
    df["asia_low"] = AsiaLow(df)

    df["pdh"] = PDH(df)
    df["pdl"] = PDL(df)

    df = df[["Open", "High", "Low", "Close",
            "k", "k_smooth", "adx", "+di", "-di", "EMA7", "EMA21", "EMA_DIFF", "EMA7_Slope",
            "indecision", "bullish_ob", "bearish_ob", "bullish_fvg", "bearish_fvg", "eqh", "eql", "bearish_mb", "bullish_mb", "bullish_rb", "bearish_rb",
            "vwap", "vwap_upper", "vwap_lower", "above_vwap", "below_vwap", "vwap_above_upper", "vwap_below_lower", "vwap_slope",
            "volume_ma",
            "asia_high", "asia_low",
            "pdh", "pdl",
            "sell_score", "buy_score"]].copy()
    # df = df[["Open", "High", "Low", "Close", "EMA_crossover", "macd_zone", "macd_line", "macd_signal", "macd_line_diff", "macd_signal_diff", "macd_line_slope", "macd_signal_line_slope" , "macd_osma", "macd_crossover", "bb_sma", "bb_upper", "bb_lower", "RSI_zone", "ADX_zone", "+DI_val", "-DI_val", "ATR", "order_block_type"]].copy()

    df.dropna(inplace=True)
    # print(df.isna().sum())
    # ready_event.set()
    return df

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

    df = load_last_mb_xauusd()
    df = add_indicators(df)

    SEQ_LEN = 12

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
        "EMA_DIFF",
        "EMA7_Slope",
        "indecision",
        "bullish_ob",
        "bearish_ob",
        "bullish_fvg",
        "bearish_fvg",
        "eqh",
        "eql",
        "bearish_mb",
        "bullish_mb",
        "bullish_rb",
        "bearish_rb",
        # "vwap",
        # "vwap_upper",
        # "vwap_lower",
        "above_vwap",
        "below_vwap",
        "vwap_above_upper",
        "vwap_below_lower",
        "vwap_slope",
        "volume_ma",
        # "asia_high",
        # "asia_low",
        # "pdh",
        # "pdl",
        "sell_score",
        "buy_score"
    ]

    agent = LSTMPPOAgent(
        state_size=len(FEATURES),
        hidden_size=64,
        action_size=4
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
    tp_price = 0

    entry_price = 0
    sl_price = 0

    tp1_price = 0

    position_size = 0.0
    realized_reward = 0.0

    tp1_hit = False

    tp1_sl_moved = False

    trade_returns = []

    # STANDARD_SL_PIPS = 100
    RR_RATIO = 0.5
    SPREAD_AND_COMMISSION = 0.6

    SL_PIPS = 40

    PIP_VALUE = 0.1

    SPREAD_AND_COMMISSION = 0.6

    state_buffer = deque(maxlen=SEQ_LEN)

    training_start_2 = time.time()
    training_start_3 = time.time()
    prob_multiplier = 1.0

    # preload sequence
    for i in range(SEQ_LEN):
        row = df.iloc[i][FEATURES].values.astype(np.float32)
        state_buffer.append(row)

    for i in range(SEQ_LEN, len(df)):

        current = df.iloc[i]

        current_price = current["Close"]
        high = current["High"]
        low = current["Low"]
        TP1_PIPS = round(SL_PIPS * RR_RATIO, 0)

        state = current[FEATURES].values.astype(np.float32)

        state_buffer.append(state)

        if len(state_buffer) < SEQ_LEN:
            continue

        state_seq = np.array(state_buffer)

        # === Select action ============================================
        result = agent.select_action(state_seq, in_position, training=True)

        if result is None:
            continue

        action, logprob, value = result

        if df["adx"].iloc[i] < 20:
            action = 0

        pnl = 0.0
        reward = 0.0
        done = False

        # ==============================================================
        # OPEN LONG
        # ==============================================================

        # if action == 1 and not in_position and df["+di"].iloc[i] > df["-di"].iloc[i] and df["EMA_DIFF"].iloc[i] > 0 and df["k"].iloc[i] < 80:
        # if action == 1 and not in_position and df["EMA7"].iloc[i] > df["EMA21"].iloc[i] and df["k"].iloc[i] < 80:
        if action == 1 and not in_position:
            in_position = True
            position_type = "long"

            entry_price = current_price

            sl_price = entry_price - (SL_PIPS * 0.1)
            tp_price = entry_price + (
                SL_PIPS * RR_RATIO * 0.1
            )

            entry_price = current_price

            sl_price = entry_price - (SL_PIPS * PIP_VALUE)

            tp1_price = entry_price + (TP1_PIPS * PIP_VALUE)

            position_size = 1.0
            realized_reward = 0.0
            pnl = 0.0
            reward = 0.0
            done = False

            agent.store_transition(
                state_seq,
                action,
                logprob,
                value,
                reward,
                done
            )

            tp1_hit = False

        # ==============================================================
        # OPEN SHORT
        # ==============================================================

        # elif action == 2 and not in_position and df["-di"].iloc[i] > df["+di"].iloc[i] and df["EMA_DIFF"].iloc[i] < 0 and df["k"].iloc[i] > 20:
        # elif action == 2 and not in_position and df["buy_score"].iloc[i] < df["sell_score"].iloc[i]:
        # elif action == 2 and not in_position and df["EMA7"].iloc[i] < df["EMA21"].iloc[i] and df["k"].iloc[i] > 20:
        elif action == 2 and not in_position:
            in_position = True
            position_type = "short"

            entry_price = current_price

            sl_price = entry_price + (SL_PIPS * 0.1)
            tp_price = entry_price - (
                SL_PIPS * RR_RATIO * 0.1
            )

            entry_price = current_price

            sl_price = entry_price + (SL_PIPS * PIP_VALUE)

            tp1_price = entry_price - (TP1_PIPS * PIP_VALUE)

            position_size = 1.0
            realized_reward = 0.0
            reward = 0.0
            pnl = 0.0
            done = False

            agent.store_transition(
                state_seq,
                action,
                logprob,
                value,
                reward,
                done
            )

            tp1_hit = False

        # ==============================================================
        # MANAGE POSITION
        # ==============================================================
        if not in_position:
            done = False

        if in_position:

            trade_closed = False

            # ==================================================
            # LONG
            # ==================================================

            if position_type == "long":

                if not tp1_hit and high >= tp1_price:

                    # realized_reward += SL_PIPS - SPREAD_AND_COMMISSION
                    pnl += (SL_PIPS * RR_RATIO) - SPREAD_AND_COMMISSION
                    reward += (SL_PIPS * RR_RATIO) - SPREAD_AND_COMMISSION

                    tp1_hit = True

                    sl_price = entry_price
                    trade_closed = True

                    # print(f"tp1 hit, +{SL_PIPS - SPREAD_AND_COMMISSION:.0f}")

                if not trade_closed and low <= sl_price:

                    remaining_pips = (
                        (sl_price - entry_price)
                        / PIP_VALUE
                    )

                    # realized_reward += (
                    reward += (
                        # remaining_pips * (position_size / 0.25) - SPREAD_AND_COMMISSION * (position_size / 0.25)
                        remaining_pips - SPREAD_AND_COMMISSION
                    )
                    pnl += (
                        remaining_pips - SPREAD_AND_COMMISSION
                        # remaining_pips - SPREAD_AND_COMMISSION * (position_size / 0.25)
                    )

                    # print(f"sl hit, (+){remaining_pips * (position_size / 0.25) - SPREAD_AND_COMMISSION * (position_size / 0.25):.0f}")
                    # print(f"remaining pips: {remaining_pips:.0f}, positions: {position_size / 0.25:.0f}")
                    # reward = realized_reward
                    # if sl_price < entry_price:
                    #     realized_reward = (SL_PIPS * 2) * -1
                    trade_closed = True

            # ==================================================
            # SHORT
            # ==================================================

            elif position_type == "short":

                if not tp1_hit and low <= tp1_price:

                    # realized_reward += SL_PIPS * 2 - SPREAD_AND_COMMISSION
                    reward += (SL_PIPS * RR_RATIO) - SPREAD_AND_COMMISSION
                    pnl += (SL_PIPS * RR_RATIO) - SPREAD_AND_COMMISSION
                    # position_size -= 0.25

                    tp1_hit = True

                    sl_price = entry_price
                    trade_closed = True
                    # print(f"tp1 hit, +{SL_PIPS - SPREAD_AND_COMMISSION:.0f}")

                if not trade_closed and high >= sl_price:

                    remaining_pips = (
                        (entry_price - sl_price)
                        / PIP_VALUE
                    )

                    # realized_reward += (
                    reward += (
                        remaining_pips - SPREAD_AND_COMMISSION
                    )
                    pnl += (
                        remaining_pips - SPREAD_AND_COMMISSION
                    )

                    # print(f"sl hit, (+){remaining_pips * (position_size / 0.25) - SPREAD_AND_COMMISSION * (position_size / 0.25):.0f}")
                    # print(f"remaining pips: {remaining_pips:.0f}, positions: {position_size / 0.25:.0f}")
                    # reward = realized_reward
                    # if sl_price > entry_price:
                    #     realized_reward = (SL_PIPS * 2) * -1
                    trade_closed = True

            if trade_closed:

                in_position = False
                done = True
                action = 3
                trade_returns.append(pnl)

                # ==============================================================
                # STORE PPO TRANSITION
                # ==============================================================
            else:
                action = 0

            agent.store_transition(
                state_seq,
                action,
                logprob,
                value,
                reward,
                done
            )

        save_counter += 1

        # ==============================================================
        # WEEKLY TRAINING
        # ==============================================================

        if save_counter % save_count == 0:
        # if len(agent.trajectory) >= 512:

            # knn._fit()
            # knn.save()

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
                max_dd = max_drawdown(trade_returns)
                R_pnl = weekly_pnl / (SL_PIPS)

                print()
                print("================================================")
                print(f"[{symbol}] WEEKLY PPO TRAINING")
                print("================================================")
                print(f"Trades:          {len(trade_returns)}")
                print(f"Weekly PnL:      {weekly_pnl:.0f} pips")
                print(f"Winrate:         {winrate*100:.2f}%")
                print(f"Mean Win:        {mean_win:.0f} pips")
                print(f"Mean Loss:       {mean_loss:.0f} pips")
                print(f"Max DD:          {max_dd/(SL_PIPS):.2f}R")
                print(f"PF:              {profit_factor:.2f}")
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
                if save_counter % (1440 * 4) == 0:
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

def open_long(symbol, lot_size):

    tick = mt5.symbol_info_tick(symbol)

    entry = tick.ask

    sl = entry - 4

    tp1 = entry + 2
    # tp2 = entry + 10
    # tp3 = entry + 15
    # tp4 = entry + 20

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
            "comment": "bot trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC
        }

        result = mt5.order_send(request)

        # print(result)

def open_short(symbol, lot_size):

    tick = mt5.symbol_info_tick(symbol)

    entry = tick.bid

    sl = entry + 4

    tp1 = entry - 2
    # tp2 = entry - 10
    # tp3 = entry - 15
    # tp4 = entry - 20

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
            "comment": "bot trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC
        }

        result = mt5.order_send(request)

        # print(result)

def open_positions(symbol):
    positions = mt5.positions_get(symbol=symbol)
    positions = [
        p
        for p in positions
        if p.magic == 123456
    ]
    return len(positions)

def test_bot(symbol="XAUUSD"):
    SEQ_LEN = 12
    
    mt5.initialize()
    account = mt5.account_info()
    # if account is None:
    #     print("Failed to get account info")
    #     print(mt5.last_error())
    #     return
    balance = account.balance
    RISK = 0.005
    # risk_per_position = max(balance * RISK / 500 / 4, 0.01)

    # tick = mt5.symbol_info_tick(symbol)
    # SL_PIPS = round(tick.bid * 0.00125 * 10, 0)
    # risk_per_position = min(
    #     max((balance * RISK) / (SL_PIPS * 10) / 4, 0.01),
    #     100.0
    # )
    # risk_per_position = round(risk_per_position, 2)
    # print(f"volume: {risk_per_position}")
    # print(f"sl pips: {SL_PIPS}")

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
        "EMA_DIFF",
        "EMA7_Slope",
        "indecision",
        "bullish_ob",
        "bearish_ob",
        "bullish_fvg",
        "bearish_fvg",
        "eqh",
        "eql",
        "bearish_mb",
        "bullish_mb",
        "bullish_rb",
        "bearish_rb",
        # "vwap",
        # "vwap_upper",
        # "vwap_lower",
        "above_vwap",
        "below_vwap",
        "vwap_above_upper",
        "vwap_below_lower",
        "vwap_slope",
        "volume_ma",
        # "asia_high",
        # "asia_low",
        # "pdh",
        # "pdl",
        "sell_score",
        "buy_score"
    ]

    # last_m15 = None
    last_m5 = None

    agent = LSTMPPOAgent(
        state_size=len(FEATURES),
        hidden_size=64,
        action_size=4
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

    # rates_m1 = mt5.copy_rates_from_pos(
    #     symbol,
    #     mt5.TIMEFRAME_M15,
    #     0,
    #     500
    # )

    df = pd.DataFrame(rates_m5)

    df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'time': 'Date'
    }, inplace=True)

    df["Date"] = pd.to_datetime(df["Date"], unit="s")
    df.set_index("Date", inplace=True)

    raw_df = df

    df = add_indicators(df)

    # last_m1 = rates_m1[-1]["time"]
    last_m5 = df.index[-1]

    # last_m5 = None
    # last_m1 = None

    # ==========================================================
    # MAIN LOOP
    # ==========================================================

    while True:

        now = datetime.now()

        seconds_until_next_5m = (
            (5 - now.minute % 5) * 60
            - now.second
            - now.microsecond / 1_000_000
        )
        # print(f"sleeping {seconds_until_next_5m:.0f} seconds, current time: {datetime.now()}")
        if seconds_until_next_5m <= 0:
            seconds_until_next_5m += 300

        time.sleep(seconds_until_next_5m)
        # print(f"slept {seconds_until_next_5m:.0f} seconds, current time: {datetime.now()}")

        tick = mt5.symbol_info_tick(symbol)
        # SL_PIPS = round(tick.bid * 0.00125 * 10, 0)
        SL_PIPS = 40
        risk_per_position = min(
            max((balance * RISK) / (SL_PIPS * 10), 0.01),
            100.0
        )
        risk_per_position = round(risk_per_position, 2)

        # ======================================================
        # MANAGE POSITIONS EVERY NEW M1 CANDLE
        # ======================================================

        # rates_m1 = mt5.copy_rates_from_pos(
        #     symbol,
        #     mt5.TIMEFRAME_M1,
        #     0,
        #     2
        # )

        # current_m1 = rates_m1[-1]["time"]

        # if current_m1 != last_m1:

           #  last_m1 = current_m1

        # manage_positions(symbol, round(SL_PIPS / 10 / 7, 2))
        # print("exited manage_positions()")

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
                'time': 'Date'
            }, inplace=True)

            new_row["Date"] = pd.to_datetime(new_row["Date"], unit="s")
            new_row.set_index("Date", inplace=True)

            if new_row.index[-1] != df.index[-1]:

                # df = pd.concat(
                #     [df, new_row]
                #     # ignore_index=True
                # )

                # df.set_index("Date", inplace=True)

                # df = (
                #     df.tail(200)
                #     # .reset_index(drop=True)
                # )

                raw_df = pd.concat(
                    [raw_df, new_row]
                    # ignore_index=True
                )

                # raw_df = raw_df.tail(200).reset_index(drop=True)
                raw_df = raw_df.tail(600)

                # print("Before indicators:", len(df))
                # df = add_indicators(raw_df.copy())
                # print(type(raw_df.index))
                # print(raw_df.index[:5])
                df = add_indicators(raw_df.copy())
                # print("After indicators:", len(df))
                # print(df.tail())
                # print(df.shape)

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

            # print("df shape:", df.shape)
            # print("state_seq shape:", state_seq.shape)
            # print("len(df):", len(df))

            # if len(df) < SEQ_LEN:
            #     print(f"Skipping: len(df)={len(df)}")
            #     continue

            # state_seq = (
            #     df[FEATURES]
            #     .tail(SEQ_LEN)
            #     .values
            #     .astype(np.float32)
            # )

            if state_seq.shape[0] != SEQ_LEN:
                print(f"Bad state shape: {state_seq.shape}")
                continue

            action, _, _ = agent.select_action(
                state_seq,
                open_pos > 0,
                training=False
            )

            if df["adx"].iloc[-1] < 20:
                action = 0

            print(f"Test action: {action}")

            # ==================================================
            # OPEN NEW TRADE
            # ==================================================

            if open_pos == 0:

                account = mt5.account_info()

                balance = account.balance

                # risk_per_position = max(
                #     balance * RISK / 500 / 4,
                #     0.01
                # )

                # if action == 1 and df["adx"].iloc[-1] > 20 and df["+di"].iloc[-1] > df["-di"].iloc[-1] and df["EMA_DIFF"].iloc[-1] > 0 and df["k"].iloc[-1] < 80:
                # if action == 1 and df["buy_score"].iloc[-1] > df["sell_score"].iloc[-1]:
                if action == 1 and df["EMA7"].iloc[-1] > df["EMA21"].iloc[-1] and df["k"].iloc[-1] < 80:
                # if action == 1:
                    # print(
                    #     f"[{symbol}] PPO BUY"
                    # )

                    open_long(
                        symbol,
                        risk_per_position
                    )

                # elif action == 2 and df["adx"].iloc[-1] > 20 and df["-di"].iloc[-1] > df["+di"].iloc[-1] and df["EMA_DIFF"].iloc[-1] < 0 and df["k"].iloc[-1] > 20:
                # elif action == 2 and df["buy_score"].iloc[-1] < df["sell_score"].iloc[-1]:
                elif action == 2 and df["EMA7"].iloc[-1] < df["EMA21"].iloc[-1] and df["k"].iloc[-1] > 20:
                # elif action == 2:
                    # print(
                    #     f"[{symbol}] PPO SELL"
                    # )

                    open_short(
                        symbol,
                        risk_per_position
                    )

                # else:

                #     print(
                #         f"[{symbol}] PPO HOLD"
                #     )

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
