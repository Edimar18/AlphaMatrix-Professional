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
    
    def _print_market_info(self, symbol, current_price, prediction, next_update, session, volatility_ratio, risk_reward):
        """Print market information in a professional format with enhanced details"""
        print(f"{'='*100}")
        print(f"Symbol: {Fore.CYAN}{symbol}{Style.RESET_ALL}")
        print(f"Current Price: {Fore.YELLOW}{current_price:.5f}{Style.RESET_ALL}")
        print(f"Signal: {self._format_prediction(prediction, session, volatility_ratio, risk_reward)}")
        
        # Technical Analysis Summary
        if abs(prediction) >= 0.3:  # Only show for strong signals
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
                        'h1_rsi_input': latest_data['h1_rsi'][None, :],
                        'h1_macd_input': latest_data['h1_macd'][None, :],
                        'h1_macd_signal_input': latest_data['h1_macd_signal'][None, :],
                        'h1_macd_diff_input': latest_data['h1_macd_diff'][None, :],
                        'h1_volume_ratio_input': latest_data['h1_volume_ratio'][None, :],
                        'h1_volatility_ratio_input': latest_data['h1_volatility_ratio'][None, :],
                        'h1_ohlc_input': latest_data['h1_ohlc'][None, :],
                        'm5_ema_long_input': latest_data['m5_ema_long'][None, :],
                        'm5_ema_short_input': latest_data['m5_ema_short'][None, :],
                        'm5_bb_upper_input': latest_data['m5_bb_upper'][None, :],
                        'm5_bb_middle_input': latest_data['m5_bb_middle'][None, :],
                        'm5_bb_lower_input': latest_data['m5_bb_lower'][None, :],
                        'm5_rsi_input': latest_data['m5_rsi'][None, :],
                        'm5_macd_input': latest_data['m5_macd'][None, :],
                        'm5_macd_signal_input': latest_data['m5_macd_signal'][None, :],
                        'm5_macd_diff_input': latest_data['m5_macd_diff'][None, :],
                        'm5_volume_ratio_input': latest_data['m5_volume_ratio'][None, :],
                        'm5_volatility_ratio_input': latest_data['m5_volatility_ratio'][None, :],
                        'm5_ohlc_input': latest_data['m5_ohlc'][None, :]
                    }
                    
                    # Make prediction
                    prediction = self.model.predict(model_inputs, verbose=0)[0][0]
                    
                    # Get current market conditions
                    session = self.mt5_fetcher._get_market_session(current_time)
                    volatility_ratio = latest_data['h1_volatility_ratio'][-1]
                    
                    # Calculate potential risk/reward
                    last_close = latest_data['h1_ohlc'][-1][-1]
                    potential_move = abs(prediction * BASE_SIGNIFICANT_MOVE)
                    stop_loss = latest_data['h1_volatility'][-1] * STOP_LOSS_MULTIPLIER
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
                        risk_reward
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