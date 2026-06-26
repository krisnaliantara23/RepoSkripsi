import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# LOAD DATA
df = pd.read_csv('XAUUSD_H1_1Tahun.csv', sep='\t')
df['Date'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'])
df.set_index('Date', inplace=True)
df.rename(columns={'<OPEN>': 'Open', '<HIGH>': 'High', '<LOW>': 'Low', '<CLOSE>': 'Close', '<TICKVOL>': 'Volume'}, inplace=True)

# SMC & INDICATORS (simplified for speed)
df.ta.ema(length=20, append=True)
df.ta.ema(length=50, append=True)
df.ta.rsi(length=14, append=True)

def swing_high_low(df, window=5):
    df['swing_high'] = df['High'].rolling(window=window, center=True).max()
    df['swing_low'] = df['Low'].rolling(window=window, center=True).min()
    return df

df = swing_high_low(df)
df['swing_high_pct'] = (df['swing_high'] - df['Close']) / df['Close']
df['swing_low_pct'] = (df['Close'] - df['swing_low']) / df['Close']
df['EMA_diff'] = (df['EMA_20'] - df['EMA_50']) / df['Close']
df['RSI_norm'] = df['RSI_14'] / 100.0

future_step = 3
df['Target_return'] = df['Close'].pct_change(periods=future_step).shift(-future_step)
df.ffill().bfill().dropna(inplace=True)
df.dropna(inplace=True)

train_size = int(len(df) * 0.8)
val_size = int(len(df) * 0.1)
train_df = df[:train_size]
val_df   = df[train_size:train_size+val_size]
test_df  = df[train_size+val_size:]

def create_dataset(X, y, window=60):
    Xs, ys = [], []
    for i in range(window, len(X)):
        Xs.append(X[i-window:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

window_size = 60

def run_experiment(fitur_list, label):
    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    scaler_x.fit(train_df[fitur_list])
    scaler_y.fit(train_df[['Target_return']])
    
    X_train, y_train = create_dataset(scaler_x.transform(train_df[fitur_list]), scaler_y.transform(train_df[['Target_return']]), window_size)
    X_test, y_test   = create_dataset(scaler_x.transform(test_df[fitur_list]), scaler_y.transform(test_df[['Target_return']]), window_size)
    
    model = Sequential([
        LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(1, activation='tanh')
    ])
    model.compile(optimizer=Adam(0.001), loss='huber')
    model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=0)
    
    y_pred_scaled = model.predict(X_test)
    y_true = scaler_y.inverse_transform(y_test).flatten()
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
    
    acc = accuracy_score((y_true > 0), (y_pred > 0))
    print(f"{label} Accuracy: {acc:.4f}")

run_experiment(['Open', 'High', 'Low', 'Close'], "OHLC")
run_experiment(['Open', 'High', 'Low', 'Close', 'EMA_diff', 'RSI_norm'], "Indicators")
run_experiment(['Open', 'High', 'Low', 'Close', 'EMA_diff', 'RSI_norm', 'swing_high_pct', 'swing_low_pct'], "SMC")
