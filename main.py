import os
import time
import numpy as np
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from mt5_util import MT5DataFetcher
from config import *
import colorama
from colorama import Fore, Back, Style

class AlphaMatrixPro:
    def __init__(self):
        colorama.init()
        self.mt5_fetcher = MT5DataFetcher()
        self.model = self._load_latest_model(SYMBOLS[0])  # Load model for first symbol
        self.last_prediction_time = None
        
    def _load_latest_model(self, symbol):
        """Load the most recent trained model"""
        model_files = [f for f in os.listdir(MODEL_SAVE_PATH) if f.startswith(symbol)]
        if not model_files:
            raise Exception(f"No trained model found for {symbol}")
        
        latest_model = max(model_files)
        return load_model(os.path.join(MODEL_SAVE_PATH, latest_model))
    
    def _format_prediction(self, pred):
        """Format prediction value into signal strength and color"""
        if pred == -1:
            return Back.RED + Fore.WHITE + "NO TRADE" + Style.RESET_ALL
        elif pred >= 0.8:
            return Back.GREEN + Fore.WHITE + "STRONG BUY" + Style.RESET_ALL
        elif pred >= 0.3:
            return Fore.GREEN + "BUY" + Style.RESET_ALL
        elif pred >= 0.1:
            return Fore.GREEN + "WEAK BUY" + Style.RESET_ALL
        elif pred <= -0.8:
            return Back.RED + Fore.WHITE + "STRONG SELL" + Style.RESET_ALL
        elif pred <= -0.3:
            return Fore.RED + "SELL" + Style.RESET_ALL
        elif pred <= -0.1:
            return Fore.RED + "WEAK SELL" + Style.RESET_ALL
        else:
            return Fore.YELLOW + "NEUTRAL" + Style.RESET_ALL
    
    def _print_header(self):
        """Print professional header"""
        os.system('cls' if os.name == 'nt' else 'clear')
        print(Back.BLUE + Fore.WHITE + "╔══════════════════════════════════════════════════════════════╗" + Style.RESET_ALL)
        print(Back.BLUE + Fore.WHITE + "║                   ALPHA MATRIX PROFESSIONAL                   ║" + Style.RESET_ALL)
        print(Back.BLUE + Fore.WHITE + "║                H1 Market Prediction System                    ║" + Style.RESET_ALL)
        print(Back.BLUE + Fore.WHITE + "╚══════════════════════════════════════════════════════════════╝" + Style.RESET_ALL)
        print()
    
    def _print_market_info(self, symbol, current_price, prediction, next_update):
        """Print market information in a professional format"""
        print(f"{'='*70}")
        print(f"Symbol: {Fore.CYAN}{symbol}{Style.RESET_ALL}")
        print(f"Current Price: {Fore.YELLOW}{current_price:.5f}{Style.RESET_ALL}")
        print(f"Signal: {self._format_prediction(prediction)}")
        print(f"Next Update: {Fore.MAGENTA}{next_update.strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}")
        print(f"{'='*70}\n")
    
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
                                next_hour
                            )
                        time.sleep(1)
                        continue
                
                # Get latest data and make prediction
                for symbol in SYMBOLS:
                    latest_data = self.mt5_fetcher.get_latest_data(symbol)
                    if latest_data is None:
                        print(f"{Fore.RED}Failed to fetch data for {symbol}{Style.RESET_ALL}")
                        continue
                    
                    # Prepare input data
                    model_inputs = {
                        'h1_ema_long_input': latest_data['h1_ema_long'][None, :],
                        'h1_ema_short_input': latest_data['h1_ema_short'][None, :],
                        'h1_bb_upper_input': latest_data['h1_bb_upper'][None, :],
                        'h1_bb_middle_input': latest_data['h1_bb_middle'][None, :],
                        'h1_bb_lower_input': latest_data['h1_bb_lower'][None, :],
                        'h1_ohlc_input': latest_data['h1_ohlc'][None, :],
                        'm5_ema_long_input': latest_data['m5_ema_long'][None, :],
                        'm5_ema_short_input': latest_data['m5_ema_short'][None, :],
                        'm5_bb_upper_input': latest_data['m5_bb_upper'][None, :],
                        'm5_bb_middle_input': latest_data['m5_bb_middle'][None, :],
                        'm5_bb_lower_input': latest_data['m5_bb_lower'][None, :],
                        'm5_ohlc_input': latest_data['m5_ohlc'][None, :]
                    }
                    
                    # Make prediction
                    prediction = self.model.predict(model_inputs, verbose=0)[0][0]
                    self.last_prediction = prediction
                    self.last_prediction_time = current_time
                    
                    # Calculate next update time
                    next_update = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
                    
                    # Print market information
                    self._print_market_info(
                        symbol,
                        mt5.symbol_info_tick(symbol).ask,
                        prediction,
                        next_update
                    )
                
                time.sleep(1)
                
            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}Shutting down Alpha Matrix Professional...{Style.RESET_ALL}")
                break
            except Exception as e:
                print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
                time.sleep(5)

if __name__ == "__main__":
    try:
        alpha_matrix = AlphaMatrixPro()
        alpha_matrix.run()
    except Exception as e:
        print(f"{Fore.RED}Fatal Error: {str(e)}{Style.RESET_ALL}")
    finally:
        colorama.deinit() 