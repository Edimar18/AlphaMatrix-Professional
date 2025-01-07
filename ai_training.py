import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Conv1D, Dense, Dropout, concatenate,
    BatchNormalization, Bidirectional, GlobalAveragePooling1D,
    Attention, MultiHeadAttention
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from datetime import datetime
from config import *

class MarketPredictor:
    def __init__(self):
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(LOG_PATH, exist_ok=True)
        self.model = self._build_model()
        
    def _build_model(self):
        """Build the neural network model with enhanced architecture"""
        # Input layers for H1 timeframe
        h1_ema_long_input = Input(shape=(H1_SEQUENCE_LENGTH, 1), name='h1_ema_long_input')
        h1_ema_short_input = Input(shape=(H1_SEQUENCE_LENGTH, 1), name='h1_ema_short_input')
        h1_bb_upper_input = Input(shape=(H1_SEQUENCE_LENGTH, 1), name='h1_bb_upper_input')
        h1_bb_middle_input = Input(shape=(H1_SEQUENCE_LENGTH, 1), name='h1_bb_middle_input')
        h1_bb_lower_input = Input(shape=(H1_SEQUENCE_LENGTH, 1), name='h1_bb_lower_input')
        h1_rsi_input = Input(shape=(H1_SEQUENCE_LENGTH, 1), name='h1_rsi_input')
        h1_macd_input = Input(shape=(H1_SEQUENCE_LENGTH, 1), name='h1_macd_input')
        h1_macd_signal_input = Input(shape=(H1_SEQUENCE_LENGTH, 1), name='h1_macd_signal_input')
        h1_macd_diff_input = Input(shape=(H1_SEQUENCE_LENGTH, 1), name='h1_macd_diff_input')
        h1_volume_ratio_input = Input(shape=(H1_SEQUENCE_LENGTH, 1), name='h1_volume_ratio_input')
        h1_volatility_ratio_input = Input(shape=(H1_SEQUENCE_LENGTH, 1), name='h1_volatility_ratio_input')
        h1_ohlc_input = Input(shape=(H1_SEQUENCE_LENGTH, 4), name='h1_ohlc_input')
        
        # Input layers for M5 timeframe
        m5_ema_long_input = Input(shape=(M5_SEQUENCE_LENGTH, 1), name='m5_ema_long_input')
        m5_ema_short_input = Input(shape=(M5_SEQUENCE_LENGTH, 1), name='m5_ema_short_input')
        m5_bb_upper_input = Input(shape=(M5_SEQUENCE_LENGTH, 1), name='m5_bb_upper_input')
        m5_bb_middle_input = Input(shape=(M5_SEQUENCE_LENGTH, 1), name='m5_bb_middle_input')
        m5_bb_lower_input = Input(shape=(M5_SEQUENCE_LENGTH, 1), name='m5_bb_lower_input')
        m5_rsi_input = Input(shape=(M5_SEQUENCE_LENGTH, 1), name='m5_rsi_input')
        m5_macd_input = Input(shape=(M5_SEQUENCE_LENGTH, 1), name='m5_macd_input')
        m5_macd_signal_input = Input(shape=(M5_SEQUENCE_LENGTH, 1), name='m5_macd_signal_input')
        m5_macd_diff_input = Input(shape=(M5_SEQUENCE_LENGTH, 1), name='m5_macd_diff_input')
        m5_volume_ratio_input = Input(shape=(M5_SEQUENCE_LENGTH, 1), name='m5_volume_ratio_input')
        m5_volatility_ratio_input = Input(shape=(M5_SEQUENCE_LENGTH, 1), name='m5_volatility_ratio_input')
        m5_ohlc_input = Input(shape=(M5_SEQUENCE_LENGTH, 4), name='m5_ohlc_input')
        
        # Process H1 technical indicators
        h1_indicators = concatenate([
            h1_ema_long_input,
            h1_ema_short_input,
            h1_bb_upper_input,
            h1_bb_middle_input,
            h1_bb_lower_input,
            h1_rsi_input,
            h1_macd_input,
            h1_macd_signal_input,
            h1_macd_diff_input,
            h1_volume_ratio_input,
            h1_volatility_ratio_input
        ], axis=2)
        
        # Process M5 technical indicators
        m5_indicators = concatenate([
            m5_ema_long_input,
            m5_ema_short_input,
            m5_bb_upper_input,
            m5_bb_middle_input,
            m5_bb_lower_input,
            m5_rsi_input,
            m5_macd_input,
            m5_macd_signal_input,
            m5_macd_diff_input,
            m5_volume_ratio_input,
            m5_volatility_ratio_input
        ], axis=2)
        
        # Enhanced H1 processing branch
        h1_conv = Conv1D(MODEL_CONFIG['conv_filters'][0], 3, activation='relu', name='h1_conv_1')(h1_indicators)
        h1_conv = BatchNormalization(name='h1_bn_1')(h1_conv)
        h1_conv = Conv1D(MODEL_CONFIG['conv_filters'][1], 3, activation='relu', name='h1_conv_2')(h1_conv)
        h1_conv = BatchNormalization(name='h1_bn_2')(h1_conv)
        h1_conv = GlobalAveragePooling1D(name='h1_gap')(h1_conv)
        
        h1_ohlc_lstm = Bidirectional(LSTM(MODEL_CONFIG['lstm_units'][0], return_sequences=True, name='h1_lstm_1'), name='h1_bilstm_1')(h1_ohlc_input)
        h1_ohlc_attention = MultiHeadAttention(num_heads=4, key_dim=MODEL_CONFIG['lstm_units'][0], name='h1_attention')(
            h1_ohlc_lstm, h1_ohlc_lstm
        )
        h1_ohlc_lstm = Bidirectional(LSTM(MODEL_CONFIG['lstm_units'][1], name='h1_lstm_2'), name='h1_bilstm_2')(h1_ohlc_attention)
        
        # Enhanced M5 processing branch
        m5_conv = Conv1D(MODEL_CONFIG['conv_filters'][0], 3, activation='relu', name='m5_conv_1')(m5_indicators)
        m5_conv = BatchNormalization(name='m5_bn_1')(m5_conv)
        m5_conv = Conv1D(MODEL_CONFIG['conv_filters'][1], 3, activation='relu', name='m5_conv_2')(m5_conv)
        m5_conv = BatchNormalization(name='m5_bn_2')(m5_conv)
        m5_conv = GlobalAveragePooling1D(name='m5_gap')(m5_conv)
        
        m5_ohlc_lstm = Bidirectional(LSTM(MODEL_CONFIG['lstm_units'][0], return_sequences=True, name='m5_lstm_1'), name='m5_bilstm_1')(m5_ohlc_input)
        m5_ohlc_attention = MultiHeadAttention(num_heads=4, key_dim=MODEL_CONFIG['lstm_units'][0], name='m5_attention')(
            m5_ohlc_lstm, m5_ohlc_lstm
        )
        m5_ohlc_lstm = Bidirectional(LSTM(MODEL_CONFIG['lstm_units'][1], name='m5_lstm_2'), name='m5_bilstm_2')(m5_ohlc_attention)
        
        # Combine all features with attention
        combined = concatenate([h1_conv, h1_ohlc_lstm, m5_conv, m5_ohlc_lstm], name='feature_concat')
        
        # Enhanced dense layers
        x = Dense(MODEL_CONFIG['dense_units'][0], activation='relu', name='dense_1')(combined)
        x = BatchNormalization(name='dense_bn_1')(x)
        x = Dropout(MODEL_CONFIG['dropout_rate'], name='dropout_1')(x)
        
        x = Dense(MODEL_CONFIG['dense_units'][1], activation='relu', name='dense_2')(x)
        x = BatchNormalization(name='dense_bn_2')(x)
        x = Dropout(MODEL_CONFIG['dropout_rate'], name='dropout_2')(x)
        
        x = Dense(MODEL_CONFIG['dense_units'][2], activation='relu', name='dense_3')(x)
        x = BatchNormalization(name='dense_bn_3')(x)
        x = Dropout(MODEL_CONFIG['dropout_rate'], name='dropout_3')(x)
        
        # Output layer
        output = Dense(1, activation='tanh', name='output')(x)  # tanh for [-1, 1] range
        
        # Create model
        model = Model(
            inputs=[
                h1_ema_long_input, h1_ema_short_input,
                h1_bb_upper_input, h1_bb_middle_input, h1_bb_lower_input,
                h1_rsi_input, h1_macd_input, h1_macd_signal_input, h1_macd_diff_input,
                h1_volume_ratio_input, h1_volatility_ratio_input, h1_ohlc_input,
                m5_ema_long_input, m5_ema_short_input,
                m5_bb_upper_input, m5_bb_middle_input, m5_bb_lower_input,
                m5_rsi_input, m5_macd_input, m5_macd_signal_input, m5_macd_diff_input,
                m5_volume_ratio_input, m5_volatility_ratio_input, m5_ohlc_input
            ],
            outputs=output
        )
        
        model.compile(
            optimizer=Adam(learning_rate=MODEL_CONFIG['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, symbol):
        """Train the model on prepared data"""
        # Load data
        data_path = os.path.join(DATA_PATH, f"{symbol}_processed.npz")
        data = np.load(data_path)
        
        # Prepare inputs
        inputs = {
            'h1_ema_long_input': data['h1_ema_long'],
            'h1_ema_short_input': data['h1_ema_short'],
            'h1_bb_upper_input': data['h1_bb_upper'],
            'h1_bb_middle_input': data['h1_bb_middle'],
            'h1_bb_lower_input': data['h1_bb_lower'],
            'h1_rsi_input': data['h1_rsi'],
            'h1_macd_input': data['h1_macd'],
            'h1_macd_signal_input': data['h1_macd_signal'],
            'h1_macd_diff_input': data['h1_macd_diff'],
            'h1_volume_ratio_input': data['h1_volume_ratio'],
            'h1_volatility_ratio_input': data['h1_volatility_ratio'],
            'h1_ohlc_input': data['h1_ohlc'],
            'm5_ema_long_input': data['m5_ema_long'],
            'm5_ema_short_input': data['m5_ema_short'],
            'm5_bb_upper_input': data['m5_bb_upper'],
            'm5_bb_middle_input': data['m5_bb_middle'],
            'm5_bb_lower_input': data['m5_bb_lower'],
            'm5_rsi_input': data['m5_rsi'],
            'm5_macd_input': data['m5_macd'],
            'm5_macd_signal_input': data['m5_macd_signal'],
            'm5_macd_diff_input': data['m5_macd_diff'],
            'm5_volume_ratio_input': data['m5_volume_ratio'],
            'm5_volatility_ratio_input': data['m5_volatility_ratio'],
            'm5_ohlc_input': data['m5_ohlc']
        }
        
        labels = data['labels']
        
        # Split data
        train_inputs = {}
        val_inputs = {}
        
        for key in inputs:
            train_data, val_data = train_test_split(
                inputs[key], test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED
            )
            train_inputs[key] = train_data
            val_inputs[key] = val_data
        
        train_labels, val_labels = train_test_split(
            labels, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED
        )
        
        # Callbacks
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(MODEL_SAVE_PATH, f"{symbol}_{timestamp}.keras")
        log_dir = os.path.join(LOG_PATH, f"{symbol}_{timestamp}")
        
        callbacks = [
            ModelCheckpoint(
                model_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            TensorBoard(log_dir=log_dir)
        ]
        
        # Train model
        history = self.model.fit(
            train_inputs,
            train_labels,
            validation_data=(val_inputs, val_labels),
            epochs=MODEL_CONFIG['epochs'],
            batch_size=MODEL_CONFIG['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        # Print final metrics
        print("\nTraining completed!")
        print(f"Best validation loss: {min(history.history['val_loss']):.4f}")
        print(f"Best validation MAE: {min(history.history['val_mae']):.4f}")
        print(f"Model saved to: {model_path}")
        print(f"Logs saved to: {log_dir}")

if __name__ == "__main__":
    predictor = MarketPredictor()
    for symbol in SYMBOLS:
        print(f"\nTraining model for {symbol}...")
        predictor.train(symbol) 