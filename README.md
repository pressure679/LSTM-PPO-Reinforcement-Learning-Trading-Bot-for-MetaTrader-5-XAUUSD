# MT5 XAUUSD LSTM PPO Trading Bot

A MetaTrader 5 reinforcement learning trading bot for XAUUSD (Gold).

The bot combines an LSTM neural network for sequence learning with Proximal Policy Optimization (PPO) to learn trading decisions directly from historical market data.

Unlike traditional bots that rely on fixed rules, the PPO agent learns when to Buy, Sell or Hold from thousands of market examples.

The program should be in a folder or the desktop where the 5m csv file with OHLC data from https://www.kaggle.com/datasets/novandraanugrah/xauusd-gold-price-historical-data-2004-2024 is, in training it will then make a LSTM-PPO-saves folder where the training is saved (used for making decisions, also in testing).

---

## Features

### Reinforcement Learning

- LSTM policy network
- PPO training
- Continuous online training
- Live inference on MT5
- Automatic checkpoint saving/loading

### Technical Indicators

- EMA 7
- EMA 21
- EMA Difference (Momentum)
- ADX
- +DI
- -DI
- Stochastic
- VWAP
- VWAP Bands
- VWAP Slope
- Volume Moving Average

### Smart Money Concepts

- Bullish Order Blocks
- Bearish Order Blocks
- Bullish Fair Value Gaps
- Bearish Fair Value Gaps
- Bullish Rejection Blocks
- Bearish Rejection Blocks
- Equal Highs
- Equal Lows
- Market Breaks
- Indecision Candles

### PPO State Features

Current state contains:

- OHLC
- EMA trend
- Momentum
- ADX trend strength
- DI Direction
- Stochastic
- VWAP
- VWAP Bands
- VWAP Position
- VWAP Slope
- Volume MA
- Order Blocks
- Fair Value Gaps
- Rejection Blocks
- Equal Highs/Lows
- Buy Score
- Sell Score

---

## Current Strategy

Current reward structure:

- Take Profit: 20 pips
- Stop Loss: 40 pips
- Risk/Reward: 1 : 0.5

The current focus is high-probability momentum trades rather than large swing trades.

---

## Training

The PPO agent trains continuously over historical MT5 data.

Example metrics during training:

- Win rate: 70–85%
- Profit Factor: 1.5–3+
- Weekly performance: typically 10–30R during training (varies by market conditions)

These figures are training statistics only and are not guarantees of future performance.

### Quarterly stats

|                     Quarter | Avg Weekly R |   Avg PF | Avg Max DD |    Avg RF |
| --------------------------: | -----------: | -------: | ---------: | --------: |
|             Q1 (Weeks 1–13) |    **6.20R** | **1.10** |  **8.80R** |  **2.04** |
|            Q2 (Weeks 14–26) |    **7.39R** | **1.12** |  **9.80R** |  **2.25** |
|            Q3 (Weeks 27–39) |   **16.89R** | **1.25** |  **8.14R** |  **5.42** |
|            Q4 (Weeks 40–52) |   **43.73R** | **1.29** | **10.81R** |  **9.20** |
|            Q5 (Weeks 53–65) |   **36.54R** | **1.38** |  **8.94R** |  **8.72** |
|            Q6 (Weeks 66–78) |   **75.07R** | **1.38** |  **8.62R** | **19.42** |
|            Q7 (Weeks 79–91) |  **164.37R** | **1.82** |  **7.23R** | **54.69** |
| Q8 (Weeks 92–102, 11 weeks) |  **126.55R** | **1.49** |  **8.76R** | **34.54** |

---

## Requirements

- Python 3.11+
- MetaTrader 5
- MetaTrader5
- pandas
- numpy
- torch

Install dependencies:

```bash
pip install MetaTrader5 pandas numpy torch
```

---

## Running

Train:

```bash
python mt5-xau-lstm-ppo-bot.py --train
```

Live trading:

```bash
python mt5-xau-lstm-ppo-bot.py --test
```

Train and trade simultaneously:

```bash
python mt5-xau-lstm-ppo-bot.py --train --test
```

---

To train download a m5 xauusd csv file form kaggle or dukascopy and place the folder in "download" relative to the directory the mt5-xau-lstm-ppo-bot.py is in.

## Project Goals

Current work focuses on:

- Improving PPO policy learning
- Better feature engineering
- Dynamic trade management
- VWAP and volume analysis
- Smart Money Concept detection
- MFE/MAE prediction research
- Higher timeframe context

---

## Disclaimer

This project is for educational and research purposes only.

Trading leveraged products involves substantial risk. Always test thoroughly on historical data and demo accounts before risking real capital.
