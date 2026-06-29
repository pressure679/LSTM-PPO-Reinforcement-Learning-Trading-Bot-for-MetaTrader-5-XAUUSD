# xLSTM PPO Reinforcement Learning Trading Bot for MetaTrader 5 (XAUUSD)

An AI-powered trading bot that combines **Extended LSTM (xLSTM)** and **Proximal Policy Optimization (PPO)** to learn profitable trading strategies on XAUUSD.

Unlike traditional indicator-based Expert Advisors, this project learns directly from historical market data using reinforcement learning while incorporating Smart Money Concepts (SMC), technical indicators, market structure, and adaptive risk management.

---

## Features

### Reinforcement Learning

- xLSTM policy network
- Proximal Policy Optimization (PPO)
- Sequence-based learning
- Reinforcement learning trading environment
- Reward shaping
- GPU/CPU training support

---

## Smart Money Concepts (SMC)

The bot automatically detects:

- Order Blocks (OB)
- Mitigated Order Blocks
- Breaker Blocks
- Mitigation Blocks (MB)
- Rejection Blocks (RB)
- Fair Value Gaps (FVG)
- Inverse Fair Value Gaps (IFVG)
- FVG Retracements
- Equal Highs (EQH)
- Equal Lows (EQL)
- Market Structure Shifts (MSS)
- Swing Highs
- Swing Lows
- Previous Day High (PDH)
- Previous Day Low (PDL)
- Asia High
- Asia Low
- Trend Lines
- Liquidity Sweeps

---

## Technical Indicators

Included features include:

- EMA 7
- EMA 21
- EMA Difference
- EMA Slopes
- ADX
- +DI
- -DI
- Stochastic Oscillator
- VWAP
- VWAP Upper Band
- VWAP Lower Band
- Volume Moving Average
- Session Detection
- Indecision Candles

---

## Adaptive Trade Management

The bot automatically manages:

- Adaptive Stop Loss
- Adaptive Take Profit
- Partial Profit Taking
- Break-even Movement
- Dynamic Position Sizing
- Risk-based Lot Calculation

---

## Machine Learning Features

The model learns from more than 50 engineered features including:

- Smart Money Concepts
- Trend
- Momentum
- Liquidity
- Session information
- Price distances
- Volume
- Volatility
- Historical sequences

---

## Trading Environment

The custom Gymnasium environment supports:

- Buy
- Sell
- Hold

Rewards consider:

- Profit
- Drawdown
- Risk
- Trade quality
- Spread
- Commission

---

## Project Structure

```
.
├── bot.py                  # Main application
├── download/               # Historical CSV data
├── models/                 # Saved models
├── logs/                   # Training logs
├── scaler.pkl              # Feature scaler
└── README.md
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/pressure679/LSTM-PPO-Reinforcement-Learning-Trading-Bot-for-MetaTrader-5-XAUUSD-
cd LSTM-PPO-Reinforcement-Learning-Trading-Bot-for-MetaTrader-5-XAUUSD-
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Requirements

- Python 3.11+
- MetaTrader 5

Python packages:

```
torch
stable-baselines3
sb3-contrib
gymnasium
MetaTrader5
numpy
pandas
scikit-learn
ta
xlstm
```

---

## Historical Data

Place historical data inside the `download/` folder.

Example:

```
download/
└── xauusd-m5-bid-2023-01-01-2026-06-01.csv
```

The bot automatically calculates all technical indicators and Smart Money Concept features before training.

---

## Training

Train the reinforcement learning model:

```bash
python bot.py --train
```

Training automatically:

- Loads historical data
- Calculates technical indicators
- Builds SMC features
- Normalizes inputs
- Creates the trading environment
- Trains the PPO agent
- Saves the trained model

---

## Live Trading

Run the live trading bot:

```bash
python bot.py --test
```

The bot will:

- Connect to MetaTrader 5
- Read live candles
- Generate feature vectors
- Predict actions
- Execute trades
- Manage open positions automatically

---

## Feature List

Current feature set includes:

```
k
k_smooth
adx
+di
-di
EMA721_DIFF
EMA7_Slope
EMA21_Slope
indecision

bullish_ob
bearish_ob
bullish_fvg
bearish_fvg
bullish_ifvg
bearish_ifvg

eqh
eql

bullish_mb
bearish_mb

bullish_rb
bearish_rb

bullish_bb
bearish_bb

bullish_mss
bearish_mss

bullish_ob_mitigation
bearish_ob_mitigation

bullish_fvg_retracement
bearish_fvg_retracement

bullish
bearish

above_vwap
below_vwap

pdh
pdl

asia_high
asia_low

pdh_distance
pdl_distance

asia_high_distance
asia_low_distance

vwap
vwap_upper
vwap_lower

bullish_ifvg_retracement
bearish_ifvg_retracement

swing_high
swing_low

support_trend_line
resistance_trend_line

seq_len
session
```

---

## Risk Management

Supports:

- Fixed percentage risk
- Dynamic lot sizing
- Adaptive stop losses
- Adaptive take profits
- Partial exits
- Break-even protection
- Maximum open positions

---

## Training Statistics

The bot reports:

- Win Rate
- Profit Factor
- Sharpe Ratio
- Sortino Ratio
- Weekly Return
- Maximum Drawdown
- Mean Win
- Mean Loss
- Average R
- Trade Count

---

## Roadmap

- [ ] Multi-symbol training
- [ ] Multi-timeframe observations
- [ ] Prioritized Experience Replay
- [ ] Hyperparameter optimization
- [ ] Walk-forward validation
- [ ] ONNX export
- [ ] Transformer/xLSTM hybrid policy
- [ ] Trade visualization dashboard
- [ ] Pattern similarity search
- [ ] Explainable AI trade analysis

---

## Disclaimer

This project is intended for educational and research purposes only.

Trading financial markets involves substantial risk. Past performance does not guarantee future results.

Always test on a demo account before trading with real funds.

---

## License

This project is licensed under the MIT License.
