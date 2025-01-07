# 🤖 ALPHA MATRIX PROFESSIONAL

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange)
![MetaTrader5](https://img.shields.io/badge/MetaTrader-5-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

ALPHA MATRIX PROFESSIONAL is an advanced AI-powered forex trading prediction system that utilizes deep learning to analyze market patterns and generate trading signals. The system combines multiple timeframe analysis (H1 and M5) with various technical indicators to provide sophisticated market insights.

## 🌟 Features

### Multi-Timeframe Analysis
- H1 (1-hour) timeframe for main signals
- M5 (5-minute) timeframe for precision entry/exit
- Synchronized data processing across timeframes

### Advanced Technical Indicators
- EMA (Exponential Moving Average) - Long and Short periods
- Bollinger Bands with dynamic standard deviations
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Volume Analysis with ratio-based filtering
- Volatility Measurements

### Market Session Awareness
- Asian Session Analysis
- London Session Analysis
- New York Session Analysis
- Session Overlap Detection
- Dynamic threshold adjustment based on session volatility

### Deep Learning Architecture
- Bidirectional LSTM layers for sequence learning
- Conv1D layers for pattern detection
- Multi-Head Attention mechanism
- Batch Normalization for stable training
- Dropout layers for regularization

### Risk Management
- Dynamic Risk/Reward calculation
- Session-based threshold adjustment
- Volatility-based position sizing
- Multiple confirmation levels for signals

## 📊 Signal Classification

The system provides various signal strengths:
- Strong Buy/Sell signals (±1)
- Medium Buy/Sell signals (±0.5)
- Weak Buy/Sell signals (±0.1)
- No Trade signals (-1)

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/Edimar18/AlphaMatrix-Professional.git
cd ALPHA-MATRIX-PROFESSIONAL
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Configure MetaTrader 5:
- Enable Expert Advisors
- Allow algorithmic trading
- Configure API access

## 🚀 Usage

1. Prepare training data:
```bash
python training_data_maker.py
```

2. Train the model:
```bash
python ai_training.py
```

3. Run live predictions:
```bash
python main.py
```

## 📁 Project Structure

```
ALPHA-MATRIX-PROFESSIONAL/
├── config.py           # Configuration and parameters
├── mt5_util.py        # MetaTrader 5 utilities
├── training_data_maker.py  # Data preparation
├── ai_training.py     # Model training
├── main.py           # Live prediction system
├── requirements.txt   # Dependencies
└── README.md         # Documentation
```

## ⚙️ Configuration

Key parameters can be adjusted in `config.py`:
- Timeframe settings
- Technical indicator parameters
- Model architecture
- Trading thresholds
- Risk management settings

## 📈 Performance Metrics

The system evaluates predictions based on:
- Directional accuracy
- Risk/Reward ratio
- Market session performance
- Volatility adaptation
- Signal strength distribution

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading forex carries significant risks, and past performance does not guarantee future results. Always practice proper risk management and consult with financial professionals.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions and feedback, please open an issue in the GitHub repository.

---
Made with ❤️ by EDIMAR MOSQUIDA