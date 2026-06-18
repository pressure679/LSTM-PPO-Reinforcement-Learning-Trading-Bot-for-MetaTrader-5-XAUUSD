import pandas as pd
import numpy as np
import os
import pickle
from io import StringIO
import random
from collections import deque
from datetime import datetime, timedelta
import subprocess
# import requests
# import threading
# from multiprocessing import Process
import time
# from time import timezone
# from decimal import Decimal
# from pybit.unified_trading import HTTP
# from pybit.unified_trading import WebSocket
# from sklearn.neighbors import NearestNeighbors
# from sklearn.metrics.pairwise import cosine_similarity
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import subprocess
# import glob
# import shutil
import MetaTrader5 as mt5
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# ready_event = threading.Event()
# process_counter = 0
# pd.set_option('future.no_silent_downcasting', True)

ACTIONS = ['hold', 'long', 'short', 'close']

# capital = 800

def load_last_mb_xauusd(file_path="C:\\Users\\Vittus Mikiassen\\Desktop\\XAU_5m_data.csv", mb=15, delimiter=';', col_names=None):
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
        
    df = pd.read_csv(StringIO(data), delimiter=delimiter, header=None, engine='python')
    
    #if col_names:
    df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    
    # Convert columns if needed, e.g.:
    df["Date"] = pd.to_datetime(df["Date"], format="%Y.%m.%d %H:%M", errors='coerce')
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close']].copy()

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

def Indecision(df, threshold=0.2):
    body = (df["Close"] - df["Open"]).abs()
    candle_range = (df["High"] - df["Low"]).replace(0, 1e-9)

    return (body / candle_range < threshold).astype(int)

def EQH(df, lookback=36, tolerance=1):
    highs = df["High"]

    return highs.rolling(lookback).apply(
        lambda x: int((abs(x[-1] - x[:-1]) <= tolerance).any()),
        raw=True
    ).fillna(0).astype(int)

def EQL(df, lookback=36, tolerance=1):
    lows = df["Low"]

    return lows.rolling(lookback).apply(
        lambda x: int((abs(x[-1] - x[:-1]) <= tolerance).any()),
        raw=True
    ).fillna(0).astype(int)

def RejectionBlock(df, wick_ratio=2.0):
    body = (df["Close"] - df["Open"]).abs()

    upper = df["High"] - df[["Open","Close"]].max(axis=1)
    lower = df[["Open","Close"]].min(axis=1) - df["Low"]

    return ((upper > body * wick_ratio) |
            (lower > body * wick_ratio)).astype(int)

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

def add_indicators(df):
    df['adx'], df['+di'], df['-di'] = ADX(df)

    df['k'], df['k_smooth'] = STOCH(df)

    df['EMA7'] = EMA(df, 7)
    df['EMA21'] = EMA(df, 21)
    df['EMA_DIFF'] = df['EMA7'] - df['EMA21']

    df["indecision"] = Indecision(df)
    df["rb"] = RejectionBlock(df)

    df["bullish_ob"] = BullishOB(df)
    df["bearish_ob"] = BearishOB(df)

    df["bullish_fvg"] = BullishFVG(df)
    df["bearish_fvg"] = BearishFVG(df)

    df["eqh"] = EQH(df)
    df["eql"] = EQL(df)

    df["bearish_mb"] = BearishMB(df)
    df["bullish_mb"] = BullishMB(df)

    df = df[["Open", "High", "Low", "Close", "k", "k_smooth", "adx", "+di", "-di", "EMA7", "EMA21", "EMA_DIFF",
            "indecision", "rb", "bullish_ob", "bearish_ob", "bullish_fvg", "bearish_fvg", "eqh", "eql", "bearish_mb", "bullish_mb"]].copy()
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

def calc_lot_size(risk_amount, stop_loss_pips=300, pip_value_per_lot=0.01):
    if stop_loss_pips <= 0:
        raise ValueError("stop_loss_pips must be positive")
    if pip_value_per_lot <= 0:
        raise ValueError("pip_value_per_lot must be positive")
    if risk_amount <= 0:
        return 0.0

    lots = (risk_amount / (stop_loss_pips * pip_value_per_lot))
    # print(f"lots: {round(lots, 0)} with risk amount ${risk_amount}")
    lot_size = round(lots, 0) * 0.01
    # print(f"lot size: {lot_size} with risk amount ${risk_amount}")
    return lot_size

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

    SEQ_LEN = 12 * 8

    FEATURES = [
        "Open",
        "High",
        "Low",
        "Close",
        "k",
        "k_smooth",
        "adx",
        "+di",
        "-di",
        "EMA7",
        "EMA21",
        "EMA_DIFF",
        "indecision",
        "rb",
        "bullish_ob",
        "bearish_ob",
        "bullish_fvg",
        "bearish_fvg",
        "eqh",
        "eql",
        "bearish_mb",
        "bullish_mb"
    ]

    agent = LSTMPPOAgent(
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
    tp_price = 0

    entry_price = 0
    sl_price = 0

    tp1_price = 0
    tp2_price = 0
    tp3_price = 0
    tp4_price = 0

    position_size = 0.0
    realized_reward = 0.0

    tp1_hit = False
    tp2_hit = False
    tp3_hit = False
    tp4_hit = False

    tp1_sl_moved = False
    tp2_sl_moved = False
    tp3_sl_moved = False

    trade_returns = []

    # STANDARD_SL_PIPS = 100
    RR_RATIO = 0.2
    # SPREAD_AND_COMMISSION = 1.2

    # SL_PIPS = 50

    # TP1_PIPS = 50
    # TP2_PIPS = 100
    # TP3_PIPS = 150
    # TP4_PIPS = 200

    PIP_VALUE = 0.1

    SPREAD_AND_COMMISSION = 0

    state_buffer = deque(maxlen=SEQ_LEN)

    # preload sequence
    for i in range(SEQ_LEN):
        row = df.iloc[i][FEATURES].values.astype(np.float32)
        state_buffer.append(row)

    for i in range(SEQ_LEN, len(df)):

        current = df.iloc[i]

        current_price = current["Close"]
        high = current["High"]
        low = current["Low"]
        # SL_PIPS = round(current_price * 0.00125 * 10, 0)
        SL_PIPS = 50
        TP1_PIPS = round(SL_PIPS * 0.26, 0)
        # TP2_PIPS = round(SL_PIPS * 2, 0)
        # TP3_PIPS = round(SL_PIPS * 3, 0)
        # TP4_PIPS = round(SL_PIPS * 4, 0)
        # SL_MOVE_BUFFER = round(SL_PIPS / 7, 0)
        # SL_PIPS = 30
        # TP1_PIPS = 100
        # TP2_PIPS = 200

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

        # if action == 0 and in_position:
        #     if position_type == "long":
        #         reward = round((current_price - entry_price) * 10, 0)

        # ==============================================================
        # OPEN LONG
        # ==============================================================

        # if action == 1 and not in_position and df["+di"].iloc[i] > df["-di"].iloc[i] and df["EMA_DIFF"].iloc[i] > 0 and df["k"].iloc[i] < 80:
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
            # tp2_price = entry_price + (TP2_PIPS * PIP_VALUE)
            # tp3_price = entry_price + (TP3_PIPS * PIP_VALUE)
            # tp4_price = entry_price + (TP4_PIPS * PIP_VALUE)

            position_size = 1.0
            realized_reward = 0.0
            pnl = 0.0
            reward = 0.0

            tp1_hit = False
            # tp2_hit = False
            # tp3_hit = False
            # tp4_hit = False

            # tp1_sl_moved = False
            # tp2_sl_moved = False
            # tp3_sl_moved = False

            # print('opened long')

        # ==============================================================
        # OPEN SHORT
        # ==============================================================

        # elif action == 2 and not in_position and df["-di"].iloc[i] > df["+di"].iloc[i] and df["EMA_DIFF"].iloc[i] < 0 and df["k"].iloc[i] > 20:
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
            # tp2_price = entry_price - (TP2_PIPS * PIP_VALUE)
            # tp3_price = entry_price - (TP3_PIPS * PIP_VALUE)
            # tp4_price = entry_price - (TP4_PIPS * PIP_VALUE)

            position_size = 1.0
            realized_reward = 0.0
            reward = 0.0
            pnl = 0.0

            tp1_hit = False
            # tp2_hit = False
            # tp3_hit = False
            # tp4_hit = False

            # tp1_sl_moved = False
            # tp2_sl_moved = False
            # tp3_sl_moved = False

            # print('opened short')

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
                    pnl += SL_PIPS * 0.2 - SPREAD_AND_COMMISSION
                    reward += SL_PIPS * 0.2 - SPREAD_AND_COMMISSION
                    position_size -= 0.25

                    tp1_hit = True

                    sl_price = entry_price
                    trade_closed = True

                    # print(f"tp1 hit, +{SL_PIPS - SPREAD_AND_COMMISSION:.0f}")
                """
                if not tp2_hit and high >= tp2_price:

                    # realized_reward += SL_PIPS * 2 - SPREAD_AND_COMMISSION
                    pnl += SL_PIPS * 2 - SPREAD_AND_COMMISSION
                    reward += SL_PIPS * 2 - SPREAD_AND_COMMISSION
                    position_size -= 0.25

                    tp2_hit = True

                    sl_price = tp1_price
                    # print(f"tp2 hit, +{SL_PIPS * 2 - SPREAD_AND_COMMISSION:.0f}")
                    # trade_closed = True
                
                if not tp3_hit and high >= tp3_price:

                    # realized_reward += SL_PIPS * 3 - SPREAD_AND_COMMISSION
                    pnl += SL_PIPS * 3 - SPREAD_AND_COMMISSION
                    reward += SL_PIPS * 3 - SPREAD_AND_COMMISSION
                    position_size -= 0.25

                    tp3_hit = True

                    sl_price = tp2_price
                    # print(f"tp3 hit, +{SL_PIPS * 3 - SPREAD_AND_COMMISSION:.0f}")

                if not tp4_hit and high >= tp4_price:

                    # realized_reward += SL_PIPS * 4 - SPREAD_AND_COMMISSION
                    pnl += SL_PIPS * 4 - SPREAD_AND_COMMISSION
                    reward += SL_PIPS * 4 - SPREAD_AND_COMMISSION

                    # reward = realized_reward

                    # print(f"tp4 hit, +{SL_PIPS * 4 - SPREAD_AND_COMMISSION:.0f}")
                    trade_closed = True
                """
                if not trade_closed and low <= sl_price:

                    remaining_pips = (
                        (sl_price - entry_price)
                        / PIP_VALUE
                    )

                    # realized_reward += (
                    # reward += (
                    #     # remaining_pips * (position_size / 0.25) - SPREAD_AND_COMMISSION * (position_size / 0.25)
                    #     remaining_pips - SPREAD_AND_COMMISSION * (position_size / 0.25)
                    # )
                    pnl += (
                        remaining_pips * (position_size) - SPREAD_AND_COMMISSION * (position_size)
                        # remaining_pips - SPREAD_AND_COMMISSION * (position_size / 0.25)
                    )

                    # print(f"sl hit, (+){remaining_pips * (position_size / 0.25) - SPREAD_AND_COMMISSION * (position_size / 0.25):.0f}")
                    # print(f"remaining pips: {remaining_pips:.0f}, positions: {position_size / 0.25:.0f}")
                    # reward = realized_reward
                    # if sl_price < entry_price:
                    #     realized_reward = (SL_PIPS * 2) * -1
                    trade_closed = True
                """
                # TP1 reached +10 pips
                if tp1_hit and not tp1_sl_moved:

                    if high >= tp1_price + (SL_MOVE_BUFFER * PIP_VALUE):

                        sl_price = tp1_price

                        tp1_sl_moved = True

                # TP2 reached +10 pips
                if tp2_hit and not tp1_sl_moved:

                    if high >= tp2_price + (SL_MOVE_BUFFER * PIP_VALUE):

                        sl_price = tp2_price

                        tp2_sl_moved = True
                
                # TP3 reached +10 pips
                if tp3_hit and not tp3_sl_moved:

                    if high >= tp3_price + (SL_MOVE_BUFFER * PIP_VALUE):

                        sl_price = tp3_price

                        tp3_sl_moved = True
                """
            # ==================================================
            # SHORT
            # ==================================================

            elif position_type == "short":

                if not tp1_hit and low <= tp1_price:

                    # realized_reward += SL_PIPS * 2 - SPREAD_AND_COMMISSION
                    reward += SL_PIPS * 0.2 - SPREAD_AND_COMMISSION
                    pnl += SL_PIPS * 0.2 - SPREAD_AND_COMMISSION
                    # position_size -= 0.25

                    tp1_hit = True

                    sl_price = entry_price
                    trade_closed = True
                    # print(f"tp1 hit, +{SL_PIPS - SPREAD_AND_COMMISSION:.0f}")
                """
                if not tp2_hit and low <= tp2_price:

                    # realized_reward += SL_PIPS * 2 - SPREAD_AND_COMMISSION
                    reward += SL_PIPS * 2 - SPREAD_AND_COMMISSION
                    pnl += SL_PIPS * 2 - SPREAD_AND_COMMISSION
                    position_size -= 0.25

                    tp2_hit = True

                    # print(f"tp2 hit, +{SL_PIPS * 2 - SPREAD_AND_COMMISSION:.0f}")
                    sl_price = tp1_price
                    # realized_reward = 300 - SPREAD_AND_COMMISSION * 2
                    # trade_closed = True

                if not tp3_hit and low <= tp3_price:

                    # realized_reward += SL_PIPS * 3 - SPREAD_AND_COMMISSION
                    reward += SL_PIPS * 3 - SPREAD_AND_COMMISSION
                    pnl += SL_PIPS * 3 - SPREAD_AND_COMMISSION
                    position_size -= 0.25

                    tp3_hit = True
                    # print(f"tp3 hit, +{SL_PIPS * 3 - SPREAD_AND_COMMISSION:.0f}")
                    sl_price = tp2_price

                if not tp4_hit and low <= tp4_price:

                    # realized_reward += SL_PIPS * 4 - SPREAD_AND_COMMISSION
                    reward += SL_PIPS * 3 - SPREAD_AND_COMMISSION
                    pnl += SL_PIPS * 3 - SPREAD_AND_COMMISSION
                    # print(f"tp4 hit, +{SL_PIPS * 4 - SPREAD_AND_COMMISSION:.0f}")
                    # reward = realized_reward

                    trade_closed = True
                """
                if not trade_closed and high >= sl_price:

                    remaining_pips = (
                        (entry_price - sl_price)
                        / PIP_VALUE
                    )

                    # realized_reward += (
                    # reward += (
                    #     remaining_pips - SPREAD_AND_COMMISSION * (position_size / 0.25)
                    # )
                    pnl += (
                        remaining_pips * (position_size) - SPREAD_AND_COMMISSION * (position_size)
                    )

                    # print(f"sl hit, (+){remaining_pips * (position_size / 0.25) - SPREAD_AND_COMMISSION * (position_size / 0.25):.0f}")
                    # print(f"remaining pips: {remaining_pips:.0f}, positions: {position_size / 0.25:.0f}")
                    # reward = realized_reward
                    # if sl_price > entry_price:
                    #     realized_reward = (SL_PIPS * 2) * -1
                    trade_closed = True
                """
                # TP1 reached +10 pips
                if tp1_hit and not tp1_sl_moved:

                    if low <= tp1_price - (SL_MOVE_BUFFER * PIP_VALUE):

                        sl_price = tp1_price

                        tp1_sl_moved = True

                # TP2 reached +10 pips
                if tp2_hit and not tp1_sl_moved:

                    if low <= tp2_price - (SL_MOVE_BUFFER * PIP_VALUE):

                        sl_price = tp2_price

                        tp2_sl_moved = True
                
                # TP1 reached +10 pips
                if tp3_hit and not tp3_sl_moved:

                    if low <= tp3_price - (SL_MOVE_BUFFER * PIP_VALUE):

                        sl_price = tp3_price

                        tp3_sl_moved = True
                """
            if trade_closed:

                in_position = False
                done = True
                # trade_returns.append(realized_reward)
                trade_returns.append(pnl)
                # print(
                #     entry_price,
                #     sl_price,
                #     tp1_price,
                #     tp2_price,
                #     tp3_price,
                #     tp4_price,
                #     realized_reward
                # )
                # print(f"closed position, pnl: {realized_reward:.0f}")

                # ==============================================================
                # STORE PPO TRANSITION
                # ==============================================================

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

        if save_counter % 1440 == 0:
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

                trade_returns = []

            print(
                f"[{symbol}] "
                f"[INFO] Training PPO on step "
                f"{save_counter}..."
            )
            agent.train()
            agent.savecheckpoint(symbol)

    # ==============================================================
    # FINAL TRAINING
    # ==============================================================

    agent.train()

    agent.savecheckpoint(symbol)

    # knn._fit()

    # knn.save()

    print(f"[{symbol}] Training complete.")

    return agent

def open_long(symbol, lot_size):

    tick = mt5.symbol_info_tick(symbol)

    entry = tick.ask

    sl = entry - 5

    tp1 = entry + 1.3
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

    sl = entry + 5

    tp1 = entry - 1.3
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

def get_ppo_positions(symbol):

    positions = mt5.positions_get(symbol=symbol)

    return [
        p
        for p in positions
        if p.magic == 123456
    ]

def move_all_stops(symbol, new_sl):
    print("in move_all_stops")

    positions = get_ppo_positions(symbol)

    for pos in positions:

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": pos.ticket,
            "sl": new_sl,
            "tp": pos.tp
        }

        result = mt5.order_send(request)

        print(
            f"SL moved ticket "
            f"{pos.ticket} -> "
            f"{new_sl}"
        )

def manage_positions(symbol, SL_MOVE_BUFFER):
    positions = get_ppo_positions(symbol)

    if not positions:
        return

    count = len(positions)

    direction = positions[0].type
    entry = positions[0].price_open

    tick = mt5.symbol_info_tick(symbol)

    if direction == mt5.ORDER_TYPE_BUY:
        current_price = tick.bid

        positions.sort(key=lambda p: p.tp)

        # TP1 hit
        if count == 3:
            move_all_stops(symbol, entry)

        # TP + buffer hit
        for pos in positions:
            if pos.tp > 0 and current_price >= pos.tp + SL_MOVE_BUFFER:
                move_all_stops(symbol, pos.tp)

    else:
        current_price = tick.ask

        positions.sort(key=lambda p: p.tp, reverse=True)

        # TP1 hit
        if count == 3:
            move_all_stops(symbol, entry)

        # TP + buffer hit
        for pos in positions:
            if pos.tp > 0 and current_price <= pos.tp - SL_MOVE_BUFFER:
                move_all_stops(symbol, pos.tp)

def test_bot(symbol="XAUUSD"):
    SEQ_LEN = 12 * 8
    
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
        "Open",
        "High",
        "Low",
        "Close",
        "k",
        "k_smooth",
        "adx",
        "+di",
        "-di",
        "EMA7",
        "EMA21",
        "EMA_DIFF",
        "indecision",
        "rb",
        "bullish_ob",
        "bearish_ob",
        "bullish_fvg",
        "bearish_fvg",
        "eqh",
        "eql",
        "bearish_mb",
        "bullish_mb"
    ]

    # last_m15 = None
    last_m5 = None

    agent = LSTMPPOAgent(
        state_size=len(FEATURES),
        hidden_size=64,
        action_size=3
    )

    # agent.model.debug = True

    agent.loadcheckpoint("XAUUSD")

    # ==========================================================
    # INITIAL LOAD
    # ==========================================================

    rates_m5 = mt5.copy_rates_from_pos(
        symbol,
        mt5.TIMEFRAME_M5,
        0,
        200
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
        SL_PIPS = 50
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

            if new_row.index[-1] != df.index[-1]:

                df = pd.concat(
                    [df, new_row],
                    ignore_index=True
                )

                df = (
                    df.tail(200)
                    .reset_index(drop=True)
                )

                """
                raw_df = pd.concat(
                    [raw_df, new_row],
                    ignore_index=True
                )

                raw_df = raw_df.tail(200).reset_index(drop=True)
                """

                # print("Before indicators:", len(df))
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

            state_seq = (
                df[FEATURES]
                .tail(SEQ_LEN)
                .values
                .astype(np.float32)
            )

            if state_seq.shape[0] != SEQ_LEN:
                print(f"Bad state shape: {state_seq.shape}")
                continue

            action, _, _ = agent.select_action(
                state_seq,
                open_pos > 0,
                training=True
            )

            if df["adx"].iloc[-1] < 20:
                action = 0

            # print(f"action: {action}")

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
                if action == 1:
                    print(
                        f"[{symbol}] PPO BUY"
                    )

                    open_long(
                        symbol,
                        risk_per_position
                    )

                # elif action == 2 and df["adx"].iloc[-1] > 20 and df["-di"].iloc[-1] > df["+di"].iloc[-1] and df["EMA_DIFF"].iloc[-1] < 0 and df["k"].iloc[-1] > 20:
                elif action == 2:
                    print(
                        f"[{symbol}] PPO SELL"
                    )

                    open_short(
                        symbol,
                        risk_per_position
                    )

                # else:

                #     print(
                #         f"[{symbol}] PPO HOLD"
                #     )

CSV_FILE = "XAU_5m_data.csv"

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
            - timedelta(days=365 * 5)
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
    # update_xauusd_data()
    train_bot("XAUUSD")
    
    # test_bot(symbol="XAUUSD-VIP")

main()
