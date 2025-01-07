import os
import time as time_module
import MetaTrader5 as mt5
import numpy as np
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from mt5_util import MT5DataFetcher
from config import *
import colorama
from colorama import Fore, Back, Style
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, MultiHeadAttention

class AlphaMatrixPro:
    def __init__(self):
        colorama.init()
        
        # Initialize MT5 connection
        if not mt5.initialize():
            raise Exception("Failed to initialize MT5. Please check if MetaTrader 5 is running.")
        
        print(f"{Fore.GREEN}Successfully connected to MetaTrader 5{Style.RESET_ALL}")
        
        # Create data fetcher
        self.mt5_fetcher = MT5DataFetcher()
        
        # Create required directories
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
        
        # Load model
        print("Loading model...")
        self.model = self._load_latest_model(SYMBOLS[0])
        
        # Initialize state variables
        self.last_prediction_time = None
        self.last_prediction = None
        self.last_session = None
        self.last_volatility_ratio = None
        self.last_risk_reward = None
        
        print(f"{Fore.GREEN}Alpha Matrix Professional initialized successfully!{Style.RESET_ALL}\n")
    
    def _load_latest_model(self, symbol):
        """Load the most recent trained model"""
        # Find the latest model file
        model_files = [f for f in os.listdir(MODEL_SAVE_PATH) if f.startswith(symbol) and f.endswith('.keras')]
        if not model_files:
            raise Exception(f"No trained model found for {symbol}")
        
        latest_model = max(model_files)
        model_path = os.path.join(MODEL_SAVE_PATH, latest_model)
        
        try:
            # Import all required layers
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import (
                Input, LSTM, Conv1D, Dense, Dropout, concatenate,
                BatchNormalization, Bidirectional, GlobalAveragePooling1D,
                MultiHeadAttention
            )
            
            # Create a new instance of MarketPredictor to get the model architecture
            from ai_training import MarketPredictor
            predictor = MarketPredictor()
            fresh_model = predictor._build_model()
            
            # Load weights from saved model
            fresh_model.load_weights(model_path)
            
            # Verify the model loaded correctly
            print("Model loaded successfully!")
            print(f"Model input names: {[input.name for input in fresh_model.inputs]}")
            print(f"Model output names: {[output.name for output in fresh_model.outputs]}")
            
            return fresh_model
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise e
    
    def _reshape_input(self, data, sequence_length):
        """Reshape input data for model"""
        # Ensure data is a numpy array
        data = np.array(data)
        
        # Handle OHLC data (4 columns)
        if len(data.shape) == 2 and data.shape[1] == 4:
            # Already in correct shape for OHLC, just add batch dimension
            return data.reshape(1, sequence_length, 4)
        
        # Handle 1D data (indicators)
        if len(data.shape) == 1:
            # Reshape to (batch_size, sequence_length, 1)
            return data.reshape(1, sequence_length, 1)
        
        # If already 2D but not OHLC
        if len(data.shape) == 2:
            if data.shape[0] == sequence_length:
                return data.reshape(1, sequence_length, 1)
            else:
                return data.reshape(1, sequence_length, 1)
        
        # Return as is if already in correct shape
        return data
    
    def _prepare_model_inputs(self, latest_data):
        """Prepare and reshape all inputs for model prediction"""
        return {
            'h1_ema_long_input': self._reshape_input(latest_data['h1_ema_long'], H1_SEQUENCE_LENGTH),
            'h1_ema_short_input': self._reshape_input(latest_data['h1_ema_short'], H1_SEQUENCE_LENGTH),
            'h1_bb_upper_input': self._reshape_input(latest_data['h1_bb_upper'], H1_SEQUENCE_LENGTH),
            'h1_bb_middle_input': self._reshape_input(latest_data['h1_bb_middle'], H1_SEQUENCE_LENGTH),
            'h1_bb_lower_input': self._reshape_input(latest_data['h1_bb_lower'], H1_SEQUENCE_LENGTH),
            'h1_rsi_input': self._reshape_input(latest_data['h1_rsi'], H1_SEQUENCE_LENGTH),
            'h1_macd_input': self._reshape_input(latest_data['h1_macd'], H1_SEQUENCE_LENGTH),
            'h1_macd_signal_input': self._reshape_input(latest_data['h1_macd_signal'], H1_SEQUENCE_LENGTH),
            'h1_macd_diff_input': self._reshape_input(latest_data['h1_macd_diff'], H1_SEQUENCE_LENGTH),
            'h1_volume_ratio_input': self._reshape_input(latest_data['h1_volume_ratio'], H1_SEQUENCE_LENGTH),
            'h1_volatility_ratio_input': self._reshape_input(latest_data['h1_volatility_ratio'], H1_SEQUENCE_LENGTH),
            'h1_ohlc_input': self._reshape_input(latest_data['h1_ohlc'], H1_SEQUENCE_LENGTH),
            'm5_ema_long_input': self._reshape_input(latest_data['m5_ema_long'], M5_SEQUENCE_LENGTH),
            'm5_ema_short_input': self._reshape_input(latest_data['m5_ema_short'], M5_SEQUENCE_LENGTH),
            'm5_bb_upper_input': self._reshape_input(latest_data['m5_bb_upper'], M5_SEQUENCE_LENGTH),
            'm5_bb_middle_input': self._reshape_input(latest_data['m5_bb_middle'], M5_SEQUENCE_LENGTH),
            'm5_bb_lower_input': self._reshape_input(latest_data['m5_bb_lower'], M5_SEQUENCE_LENGTH),
            'm5_rsi_input': self._reshape_input(latest_data['m5_rsi'], M5_SEQUENCE_LENGTH),
            'm5_macd_input': self._reshape_input(latest_data['m5_macd'], M5_SEQUENCE_LENGTH),
            'm5_macd_signal_input': self._reshape_input(latest_data['m5_macd_signal'], M5_SEQUENCE_LENGTH),
            'm5_macd_diff_input': self._reshape_input(latest_data['m5_macd_diff'], M5_SEQUENCE_LENGTH),
            'm5_volume_ratio_input': self._reshape_input(latest_data['m5_volume_ratio'], M5_SEQUENCE_LENGTH),
            'm5_volatility_ratio_input': self._reshape_input(latest_data['m5_volatility_ratio'], M5_SEQUENCE_LENGTH),
            'm5_ohlc_input': self._reshape_input(latest_data['m5_ohlc'], M5_SEQUENCE_LENGTH)
        }
    
    def _format_prediction(self, pred, session, volatility_ratio, risk_reward):
        """Format prediction value into signal strength and color with enhanced information"""
        signal = ""
        if pred == -1:
            signal = Back.RED + Fore.WHITE + "NO TRADE" + Style.RESET_ALL
        elif pred >= 0.8:
            signal = Back.GREEN + Fore.WHITE + "STRONG BUY" + Style.RESET_ALL
        elif pred >= 0.3:
            signal = Fore.GREEN + "BUY" + Style.RESET_ALL
        elif pred >= 0.1:
            signal = Fore.GREEN + "WEAK BUY" + Style.RESET_ALL
        elif pred <= -0.8:
            signal = Back.RED + Fore.WHITE + "STRONG SELL" + Style.RESET_ALL
        elif pred <= -0.3:
            signal = Fore.RED + "SELL" + Style.RESET_ALL
        elif pred <= -0.1:
            signal = Fore.RED + "WEAK SELL" + Style.RESET_ALL
        else:
            signal = Fore.YELLOW + "NEUTRAL" + Style.RESET_ALL
            
        # Add session and market condition information
        session_info = f"{Fore.CYAN}[{session} SESSION]"
        volatility_info = (
            f"{Fore.RED}[HIGH VOLATILITY]" if volatility_ratio > VOLATILITY_THRESHOLD
            else f"{Fore.GREEN}[NORMAL VOLATILITY]"
        )
        risk_reward_info = (
            f"{Fore.GREEN}[GOOD R/R: {risk_reward:.2f}]" if risk_reward >= MIN_RISK_REWARD_RATIO
            else f"{Fore.YELLOW}[LOW R/R: {risk_reward:.2f}]"
        )
        
        return f"{signal} {session_info} {volatility_info} {risk_reward_info}{Style.RESET_ALL}"
    
    def _print_header(self):
        """Print professional header"""
        os.system('cls' if os.name == 'nt' else 'clear')
        print(Back.BLUE + Fore.WHITE + "╔══════════════════════════════════════════════════════════════╗" + Style.RESET_ALL)
        print(Back.BLUE + Fore.WHITE + "║                   ALPHA MATRIX PROFESSIONAL                   ║" + Style.RESET_ALL)
        print(Back.BLUE + Fore.WHITE + "║                H1 Market Prediction System                    ║" + Style.RESET_ALL)
        print(Back.BLUE + Fore.WHITE + "╚══════════════════════════════════════════════════════════════╝" + Style.RESET_ALL)
        print()
    
    def _print_market_info(self, symbol, current_price, prediction, next_update, session, volatility_ratio, risk_reward, latest_data=None):
        """Print market information in a professional format with enhanced details"""
        print(f"{'='*100}")
        print(f"Symbol: {Fore.CYAN}{symbol}{Style.RESET_ALL}")
        print(f"Current Price: {Fore.YELLOW}{current_price:.5f}{Style.RESET_ALL}")
        print(f"Signal: {self._format_prediction(prediction, session, volatility_ratio, risk_reward)}")
        
        # Technical Analysis Summary
        if latest_data is not None and abs(prediction) >= 0.3:  # Only show for strong signals
            print("\nTechnical Analysis:")
            self._print_technical_summary(latest_data)
        
        print(f"\nNext Update: {Fore.MAGENTA}{next_update.strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}")
        print(f"{'='*100}\n")
    
    def _print_technical_summary(self, data):
        """Print summary of technical indicators"""
        # RSI Analysis
        rsi = data['h1_rsi'][-1]
        rsi_color = (
            Fore.RED if rsi > RSI_OVERBOUGHT
            else Fore.GREEN if rsi < RSI_OVERSOLD
            else Fore.YELLOW
        )
        print(f"RSI (H1): {rsi_color}{rsi:.1f}{Style.RESET_ALL}")
        
        # MACD Analysis
        macd = data['h1_macd'][-1]
        macd_signal = data['h1_macd_signal'][-1]
        macd_color = Fore.GREEN if macd > macd_signal else Fore.RED
        print(f"MACD (H1): {macd_color}{macd:.5f}{Style.RESET_ALL}")
        
        # Volume Analysis
        volume_ratio = data['h1_volume_ratio'][-1]
        volume_color = Fore.GREEN if volume_ratio > VOLUME_THRESHOLD else Fore.YELLOW
        print(f"Volume: {volume_color}{volume_ratio:.1f}x Average{Style.RESET_ALL}")
    
    def run(self):
        """Main prediction loop"""
        while True:
            try:
                self._print_header()
                current_time = datetime.now()
                
                # Check if we need to wait for next H1 candle
                if self.last_prediction_time:
                    next_hour = self.last_prediction_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
                    if current_time < next_hour:
                        for symbol in SYMBOLS:
                            self._print_market_info(
                                symbol,
                                mt5.symbol_info_tick(symbol).ask,
                                self.last_prediction,
                                next_hour,
                                self.last_session,
                                self.last_volatility_ratio,
                                self.last_risk_reward
                            )
                        time_module.sleep(1)
                        continue
                
                # Get latest data and make prediction
                for symbol in SYMBOLS:
                    print(f"\nProcessing {symbol}...")
                    
                    # Fetch latest data
                    latest_data = self.mt5_fetcher.get_latest_data(symbol)
                    if latest_data is None:
                        print(f"{Fore.RED}Failed to fetch data for {symbol}{Style.RESET_ALL}")
                        continue
                    
                    try:
                        # Prepare input data
                        print("Preparing model inputs...")
                        model_inputs = self._prepare_model_inputs(latest_data)
                        
                        # Verify input shapes
                        for name, tensor in model_inputs.items():
                            print(f"Input shape for {name}: {tensor.shape}")
                        
                        # Make prediction
                        print("Making prediction...")
                        prediction = self.model.predict(model_inputs, verbose=0)[0][0]
                        print(f"Raw prediction value: {prediction}")
                        
                        # Get current market conditions
                        session = self.mt5_fetcher._get_market_session(current_time)
                        volatility_ratio = latest_data['h1_volatility_ratio'][-1]
                        
                        # Calculate potential risk/reward
                        last_close = latest_data['h1_ohlc'][-1][-1]
                        stop_loss = volatility_ratio * STOP_LOSS_MULTIPLIER
                        potential_move = abs(prediction * BASE_SIGNIFICANT_MOVE)
                        risk_reward = (potential_move * 2) / stop_loss if stop_loss != 0 else 0
                        
                        # Store values for next iteration
                        self.last_prediction = prediction
                        self.last_prediction_time = current_time
                        self.last_session = session
                        self.last_volatility_ratio = volatility_ratio
                        self.last_risk_reward = risk_reward
                        
                        # Calculate next update time
                        next_update = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
                        
                        # Print market information
                        self._print_market_info(
                            symbol,
                            mt5.symbol_info_tick(symbol).ask,
                            prediction,
                            next_update,
                            session,
                            volatility_ratio,
                            risk_reward,
                            latest_data
                        )
                        
                    except Exception as e:
                        print(f"{Fore.RED}Error during prediction: {str(e)}")
                        print("Input shapes:")
                        for name, tensor in model_inputs.items():
                            print(f"{name}: {tensor.shape}")
                        raise e
                
                time_module.sleep(1)
                
            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}Shutting down Alpha Matrix Professional...{Style.RESET_ALL}")
                break
            except Exception as e:
                print(f"{Fore.RED}Fatal Error: {str(e)}{Style.RESET_ALL}")
                time_module.sleep(5)

if __name__ == "__main__":
    try:
        alpha_matrix = AlphaMatrixPro()
        alpha_matrix.run()
    except Exception as e:
        print(f"{Fore.RED}Fatal Error: {str(e)}{Style.RESET_ALL}")
    finally:
        colorama.deinit() 