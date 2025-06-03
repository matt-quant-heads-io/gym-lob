# 📈 LOB Trading Environment

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)

**A professional-grade Limit Order Book (LOB) trading environment with LSTM-Transformer PPO agents and real-time DOM visualization**

[Features](#-features) •
[Installation](#-installation) •
[Quick Start](#-quick-start) •
[GUI Demo](#-gui-visualization) •
[Documentation](#-documentation)

</div>

---

## 🚀 **Features**

### **Trading Environment**
- **🎯 Professional DOM Interface**: 20×6 order book representation mimicking real trading terminals
- **⚡ Sophisticated Order Management**: Passive orders with expiry, fill detection, and realistic sizing
- **💰 Realistic Market Simulation**: Transaction costs, slippage, and position limits
- **🔄 Discrete Action Space**: 7 trading actions (passive/market orders, cancellations, position management)

### **AI Agent Architecture**
- **🧠 LSTM + Transformer**: Sequential processing with attention mechanisms for temporal market patterns
- **🎭 Actor-Critic Design**: Separate policy and value networks for efficient learning
- **🏆 PPO Training**: State-of-the-art reinforcement learning with GAE and gradient clipping

### **Professional Visualization**
- **📊 Real-time DOM Ladder**: Live order book with agent order placement visualization
- **📈 Position & P&L Tracking**: Real-time position monitoring and performance metrics
- **🎨 Trading Terminal UI**: Professional dark theme with color-coded bid/ask sides

---

## 📦 **Installation**

### **Prerequisites**
- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster training)

### **Setup**
```bash
# Clone the repository
git clone https://github.com/your-username/lob-trading-env.git
cd lob-trading-env

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print('PyTorch version:', torch.__version__)"
```

### **Dependencies**
```
torch>=2.0.0          # Deep learning framework
gymnasium>=0.29.0     # RL environment interface
numpy>=1.24.0         # Numerical computing
pandas>=2.0.0         # Data manipulation
pyyaml>=6.0           # Configuration management
matplotlib>=3.7.0     # Visualization
tensorboard>=2.13.0   # Training monitoring
```

---

## ⚡ **Quick Start**

### **1. Train Your First Agent**
```bash
# Start training with default configuration
python train.py

# Training will automatically:
# ✓ Initialize the LOB environment
# ✓ Create LSTM-Transformer model
# ✓ Run PPO training loop
# ✓ Save model checkpoints
```

### **2. Evaluate Performance**
```bash
# Evaluate trained model
python eval.py

# Generates:
# ✓ Performance statistics
# ✓ Visualization plots
# ✓ Episode analysis
```

### **3. Launch GUI Visualization**
```bash
# Real-time DOM visualization
python run_visualization.py

# Watch your agent trade live!
# ✓ See order placements
# ✓ Track P&L changes
# ✓ Monitor position evolution
```

---

## 🎮 **GUI Visualization**

<div align="center">

### **Professional DOM Ladder Interface**

```
┌─────────────────────────────────────────────────────────────────┐
│              🏆 LOB TRADING AGENT - DOM VISUALIZER              │
├─────────────────────────────────────────────────────────────────┤
│ 📊 POS: +25.3  │  💰 P&L: $127.45  │  💵 CASH: $9,872  │ ⏱️ STEP: 456 │
├──────────────────────────────────────────┬──────────────────────┤
│                DOM LADDER                │     AGENT STATUS     │
│ ┌──────────────────────────────────────┐ │ ┌──────────────────┐ │
│ │BID VOL│BID SIZE│BID PRICE│ASK PRICE │ │ │  Working Orders  │ │
│ │ 234.7 │  87.3  │ 99.98   │ 100.01   │ │ │🔵 BID: 25@$99.95 │ │
│ │ 189.2 │  65.1  │ 99.97   │ 100.02   │ │ │   ASK: None      │ │
│ │ 156.8 │  43.9  │ 99.96   │ 100.03   │ │ │                  │ │
│ │🔵25.0 │        │         │          │ │ │   Market Info    │ │
│ │ 98.4  │  32.7  │ 99.95   │ 100.04   │ │ │📈 MID: $99.995   │ │
│ └──────────────────────────────────────┘ │ │📏 SPREAD: $0.03  │ │
│                                          │ └──────────────────┘ │
└──────────────────────────────────────────┴──────────────────────┘
```

</div>

### **GUI Features**
- **🎯 Real-time Order Book**: Live 20-level DOM with bid/ask visualization
- **🔵 Agent Order Tracking**: See exactly where passive orders are placed
- **📊 Performance Metrics**: Live P&L, position, and cash tracking
- **🎨 Professional Styling**: Trading terminal aesthetics with color coding
- **⚡ High-Frequency Updates**: 10 FPS refresh rate for smooth visualization

---

## 📁 **Project Structure**

```
lob_trading_env/
├── 📄 README.md                    # You are here!
├── 📄 requirements.txt             # Python dependencies
├── 🎯 train.py                     # Main training script
├── 📊 eval.py                      # Model evaluation
├── 🖥️  run_visualization.py        # GUI launcher
│
├── 📁 config/
│   └── 📄 default.yaml            # Hyperparameters & settings
│
├── 📁 env/                         # Trading Environment
│   ├── 📄 __init__.py
│   ├── 🏭 orderbook_env.py         # Main Gym environment
│   └── 🔧 utils.py                 # DOM formatting & order management
│
├── 📁 model/                       # AI Architecture
│   ├── 📄 __init__.py
│   ├── 🧠 actor_critic.py          # LSTM + Transformer network
│   └── 🎓 ppo.py                   # PPO trainer & rollout buffer
│
├── 📁 gui/                         # Visualization Interface
│   ├── 📄 __init__.py
│   ├── 🎨 dom_visualizer.py        # Professional DOM GUI
│   └── 🎮 interactive_eval.py      # Live evaluation interface
│
└── 📁 data/                        # Market Data (optional)
    └── 📁 orderbooks/
        └── 📄 (market data files)
```

---

## ⚙️ **Configuration**

The environment is highly configurable via `config/default.yaml`:

### **Order Management**
```yaml
order:
  size_pct_of_level: 0.5              # Order size as % of DOM level volume
  passive_order_ticks_offset: 2       # Ticks away from trigger price
  passive_order_expiry_steps: 10      # Max order lifetime in steps
```

### **PPO Training**
```yaml
ppo:
  learning_rate: 0.0003               # Adam optimizer learning rate
  clip_param: 0.2                     # PPO clipping parameter
  entropy_coeff: 0.01                 # Exploration bonus
  batch_size: 64                      # Mini-batch size
  num_epochs: 10                      # PPO update epochs
```

### **Model Architecture**
```yaml
model:
  lstm_hidden_size: 128               # LSTM hidden dimensions
  transformer_heads: 4                # Multi-head attention
  transformer_layers: 2               # Transformer depth
  dropout: 0.1                        # Regularization
```

### **Environment Settings**
```yaml
env:
  episode_length: 1000                # Steps per episode
  tick_size: 0.01                     # Minimum price increment
  max_position: 100                   # Position limits
  transaction_cost: 0.001             # Trading fees
```

---

## 🎯 **Action Space**

The agent can execute 7 discrete trading actions:

| Action | Description | Use Case |
|--------|-------------|----------|
| `DO_NOTHING` | No action taken | Market observation |
| `PASSIVE_LIMIT_BUY_ORDER` | Place passive bid | Liquidity provision |
| `PASSIVE_LIMIT_SELL_ORDER` | Place passive ask | Liquidity provision |
| `CANCEL_PASSIVE_ORDERS` | Cancel all resting orders | Risk management |
| `MARKET_BUY` | Immediate buy execution | Aggressive entry |
| `MARKET_SELL` | Immediate sell execution | Aggressive exit |
| `CLOSE_ALL` | Close all positions | Risk management |

---

## 📊 **Observation Space**

**Total Features**: 122 dimensions

### **DOM Representation** (120 features)
- **20×6 matrix** flattened to 120 features
- **Bid side** (columns 0-2): Volume, Size, Price (descending)
- **Ask side** (columns 3-5): Volume, Size, Price (ascending)

### **State Information** (2 features)
- **Current Position**: Scalar value (positive=long, negative=short)
- **Passive Orders**: [bid_size, ask_size] vector