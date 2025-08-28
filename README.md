# 🧠 ALE AI Trading System

**Advanced AI-Powered Trading with Spot, Futures & All Bitget Features**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com)
[![Bitget](https://img.shields.io/badge/Bitget-API-orange.svg)](https://bitget.com)
[![AI](https://img.shields.io/badge/AI-Trading-purple.svg)](https://github.com)

## 🚀 Features

### ✅ **Complete Trading System**
- **Spot Trading**: Market & Limit orders for all major cryptocurrencies
- **Futures Trading**: Leveraged trading with 1x-20x options
- **Real-time Execution**: Live order placement and management
- **Portfolio Management**: Multi-asset portfolio tracking
- **Risk Management**: Professional-grade safety controls

### 🧠 **Advanced AI Features**
- **AI Consciousness System**: Dynamic intelligence scaling
- **Market Analysis**: Trend detection & volatility analysis
- **Signal Generation**: AI-powered trading signals
- **Portfolio Optimization**: Automatic rebalancing
- **Performance Learning**: AI improves over time

### 🛡️ **Enterprise Security**
- **Secure Credential Storage**: Encrypted API key management
- **IP Whitelisting**: Enhanced API security
- **Risk Controls**: Stop-loss, take-profit, position limits
- **Audit Logging**: Complete activity tracking
- **Demo Mode**: Safe testing environment

## 📋 Prerequisites

- **Python 3.8+**
- **Bitget Account** with API access
- **Windows 10/11** (tested on Windows 10)
- **Internet Connection** for real-time trading

## 🛠️ Installation

### 1. **Clone Repository**
```bash
git clone <repository-url>
cd AI_Trading_Suite
```

### 2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 3. **Set Up API Credentials**
```bash
python setup_api.py
```

**Follow the prompts to enter your Bitget API credentials:**
- API Key
- API Secret  
- API Passphrase

## 🔐 API Setup Guide

### **Getting Bitget API Credentials**

1. **Log into Bitget Account**
   - Visit [bitget.com](https://bitget.com)
   - Sign in to your account

2. **Navigate to API Management**
   - Go to Account Settings
   - Select "API Management"
   - Click "Create API Key"

3. **Configure API Permissions**
   - ✅ **Spot Trading**: Enable for spot orders
   - ✅ **Futures Trading**: Enable for futures orders
   - ✅ **Read Only**: Enable for account data
   - ❌ **Withdrawals**: Disable for security

4. **Set IP Restrictions**
   - Add your current IP address
   - Enable IP whitelisting

5. **Save Credentials**
   - Copy API Key, Secret, and Passphrase
   - Store securely (never share!)

### **Quick Setup Commands**

```bash
# Setup API credentials
python setup_api.py

# Check configuration
python setup_api.py check

# Show help
python setup_api.py help
```

## 🚀 Running the System

### **Start Trading System**
```bash
python web_ui_singularity.py
```

### **Access Dashboard**
Open your browser and navigate to:
```
http://localhost:5000
```

## 🎯 Trading Features

### **Spot Trading**
- **Market Orders**: Instant execution at current price
- **Limit Orders**: Set specific price targets
- **Multiple Cryptocurrencies**: BTC, ETH, SOL, ADA, DOT
- **Real-time Validation**: Input validation and risk checks

### **Futures Trading**
- **Leverage Options**: 1x, 2x, 5x, 10x, 20x
- **Long/Short Positions**: Full directional trading
- **Risk Management**: Built-in position sizing
- **Margin Control**: Automatic margin calculations

### **AI Trading**
- **Automatic Signals**: AI analyzes markets 24/7
- **Smart Execution**: Automatic trade placement
- **Portfolio Rebalancing**: AI-driven optimization
- **Performance Learning**: Continuous improvement

## 🛡️ Risk Management

### **Safety Features**
- **Daily Loss Limits**: Configurable maximum daily losses
- **Position Size Limits**: Maximum position percentages
- **Stop Loss**: Automatic stop-loss execution
- **Take Profit**: Automatic take-profit execution
- **Correlation Control**: Portfolio diversification

### **Risk Settings**
```bash
MAX_DAILY_LOSS=5        # 5% maximum daily loss
MAX_POSITION_SIZE=10     # 10% maximum position size
STOP_LOSS=2             # 2% automatic stop-loss
TAKE_PROFIT=6           # 6% automatic take-profit
```

## 📊 Dashboard Features

### **Real-time Monitoring**
- **System Status**: Trading engine status
- **AI Consciousness**: Intelligence level tracking
- **Active Orders**: Current order management
- **Positions**: Open position tracking
- **Performance Metrics**: Win rate, P&L, Sharpe ratio

### **Trading Controls**
- **Start/Stop Trading**: One-click activation
- **Emergency Stop**: Immediate halt to all trading
- **Order Management**: Place, modify, cancel orders
- **Risk Settings**: Update risk parameters

### **Market Analysis**
- **Live Price Feeds**: Real-time cryptocurrency prices
- **Technical Indicators**: Advanced chart analysis
- **Market Sentiment**: AI-powered sentiment analysis
- **Volatility Tracking**: Market volatility monitoring

## 🔧 Configuration

### **Environment Variables**
The system automatically loads configuration from:
```
~/.ale_ai_trading/.env
```

### **Key Configuration Options**
```bash
# Trading Mode
TRADING_MODE=paper      # demo, paper, live

# AI Settings
AI_CONFIDENCE_THRESHOLD=0.75
AI_UPDATE_INTERVAL=30
AUTO_REBALANCE=true

# Risk Management
MAX_DAILY_LOSS=5
MAX_POSITION_SIZE=10
STOP_LOSS=2
TAKE_PROFIT=6
```

### **Updating Configuration**
```bash
# Use the configuration manager
python config.py

# Or edit .env file directly
# (Located at ~/.ale_ai_trading/.env)
```

## 📈 Getting Started

### **1. Paper Trading (Recommended)**
```bash
# Set trading mode to paper
TRADING_MODE=paper

# Start system
python web_ui_singularity.py

# Test features without real money
```

### **2. Live Trading**
```bash
# Set trading mode to live
TRADING_MODE=live

# Ensure proper risk settings
MAX_DAILY_LOSS=2        # Conservative 2%
MAX_POSITION_SIZE=5     # Small 5% positions

# Start with small amounts
```

### **3. AI Trading**
```bash
# Enable AI features
AUTO_REBALANCE=true
CONSCIOUSNESS_LEARNING=true

# Set confidence threshold
AI_CONFIDENCE_THRESHOLD=0.8  # High confidence only
```

## 🚨 Safety Guidelines

### **Before Live Trading**
1. **Test Thoroughly**: Use paper trading mode first
2. **Start Small**: Begin with minimal position sizes
3. **Set Limits**: Configure strict risk parameters
4. **Monitor Closely**: Watch AI decisions initially
5. **Have Exit Plan**: Know how to stop trading quickly

### **Risk Warnings**
- ⚠️ **Cryptocurrency trading is highly risky**
- ⚠️ **Never invest more than you can afford to lose**
- ⚠️ **AI trading can result in significant losses**
- ⚠️ **Past performance doesn't guarantee future results**
- ⚠️ **Always test thoroughly before live trading**

## 🐛 Troubleshooting

### **Common Issues**

#### **API Connection Failed**
```bash
# Check credentials
python setup_api.py check

# Verify API permissions
# Ensure IP whitelisting is correct
```

#### **Trading Engine Not Available**
```bash
# Check system initialization
python -c "import web_ui_singularity; print('Import successful')"

# Verify dependencies
pip install -r requirements.txt
```

#### **Permission Errors**
```bash
# Check file permissions
# Ensure .ale_ai_trading directory is accessible
# Run as administrator if needed
```

### **Log Files**
- **Main Log**: `ai_trading.log`
- **Configuration**: `~/.ale_ai_trading/`
- **Error Details**: Check console output

## 📚 API Reference

### **Trading Endpoints**
```bash
POST /api/trading/start          # Start AI trading
POST /api/trading/stop           # Stop AI trading
POST /api/trading/place-order    # Place spot order
POST /api/trading/place-futures-order  # Place futures order
GET  /api/trading/status         # Get trading status
GET  /api/trading/logs           # Get AI activity logs
```

### **Market Endpoints**
```bash
GET /api/market/ticker/<symbol>  # Get market ticker
GET /api/market/analysis         # Get AI market analysis
```

## 🔮 Future Features

### **Planned Enhancements**
- **Advanced Charting**: Interactive trading charts
- **Mobile App**: iOS/Android trading app
- **Social Trading**: Copy successful traders
- **Backtesting**: Historical strategy testing
- **Multi-Exchange**: Support for other exchanges

### **AI Improvements**
- **Deep Learning**: Neural network strategies
- **Sentiment Analysis**: News and social media integration
- **Portfolio Optimization**: Advanced allocation algorithms
- **Risk Prediction**: AI-powered risk assessment

## 🤝 Support

### **Getting Help**
1. **Check Documentation**: This README and code comments
2. **Review Logs**: Check `ai_trading.log` for errors
3. **Test Configuration**: Use `python setup_api.py check`
4. **Verify Setup**: Ensure all prerequisites are met

### **Reporting Issues**
- **Bug Reports**: Include error logs and steps to reproduce
- **Feature Requests**: Describe desired functionality
- **Security Issues**: Report privately and responsibly

## 📄 License

This project is for educational and research purposes. Use at your own risk.

## ⚠️ Disclaimer

**This software is provided "as is" without warranty. Cryptocurrency trading involves substantial risk and may result in the loss of your invested capital. The developers are not responsible for any financial losses incurred through the use of this software.**

**Always test thoroughly in a safe environment before using real funds. Never invest more than you can afford to lose.**

---

## 🎉 **Ready to Start Trading?**

Your ALE AI Trading System is now fully configured and ready for action!

```bash
# Start the system
python web_ui_singularity.py

# Open dashboard
http://localhost:5000

# Begin your AI trading journey! 🚀
```

**Happy Trading! 🧠💹**
