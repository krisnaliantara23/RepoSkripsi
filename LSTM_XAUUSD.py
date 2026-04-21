import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# load Data==========
df = pd.read_csv('XAUUSD_H1_1Tahun.csv', sep='\t')

df['Date'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'])
df.set_index('Date', inplace=True)

df.rename(columns={
    '<OPEN>': 'Open',
    '<HIGH>': 'High',
    '<LOW>': 'Low',
    '<CLOSE>': 'Close',
    '<TICKVOL>': 'Volume'
}, inplace=True)

# Bagian Indikator=========
df.ta.ema(length=20, append=True)
df.ta.ema(length=50, append=True)
df.ta.rsi(length=14, append=True)
df.ta.atr(length=14, append=True)

psar = df.ta.psar()
df['PSAR'] = psar.iloc[:, 0].fillna(psar.iloc[:, 1])

# Smart Money Concept
# ========================
# SMC (SMART MONEY CONCEPT)
# ========================

# 1. SWING HIGH & LOW
def swing_high_low(df, window=5):
    df['swing_high'] = np.where(
        df['High'] == df['High'].rolling(window=window, center=True).max(),
        df['High'],
        np.nan
    )
    
    df['swing_low'] = np.where(
        df['Low'] == df['Low'].rolling(window=window, center=True).min(),
        df['Low'],
        np.nan
    )
    
    return df


# 2. ORDER BLOCK (Sederhana tapi valid)
def order_block(df):
    # Bullish OB → candle bearish lalu bullish
    df['OB_bull'] = np.where(
        (df['Close'] > df['Open']) & (df['Close'].shift(1) < df['Open'].shift(1)),
        df['Low'],
        0
    )
    
    # Bearish OB → candle bullish lalu bearish
    df['OB_bear'] = np.where(
        (df['Close'] < df['Open']) & (df['Close'].shift(1) > df['Open'].shift(1)),
        df['High'],
        0
    )
    
    return df


# 3. FAIR VALUE GAP (FVG)
def fair_value_gap(df):
    # Gap naik (bullish imbalance)
    df['FVG_up'] = np.where(
        df['Low'] > df['High'].shift(2),
        df['Low'] - df['High'].shift(2),
        0
    )

    # Gap turun (bearish imbalance)
    df['FVG_down'] = np.where(
        df['High'] < df['Low'].shift(2),
        df['Low'].shift(2) - df['High'],
        0
    )
    
    return df


# ========================
# APPLY SMC
# ========================
df = swing_high_low(df)
df = order_block(df)
df = fair_value_gap(df)

# ========================
# HANDLE NaN
# ========================
df.fillna(0, inplace=True)

# TARGET (REGRESI t+1)
df['Target'] = df['Close'].shift(-1)

df.dropna(inplace=True)

# FITUR DI SMC
fitur = [
    'Open', 'High', 'Low', 'Close',
    'EMA_20', 'EMA_50', 'RSI_14', 'ATRr_14',
    'PSAR',
    
    # SMC FEATURES
    'swing_high', 'swing_low',
    'OB_bull', 'OB_bear',
    'FVG_up', 'FVG_down'
]

#Split Data
train_size = int(len(df) * 0.8)
val_size = int(len(df) * 0.1)

train_df = df[:train_size]
val_df = df[train_size:train_size+val_size]
test_df = df[train_size+val_size:]

# Scaling
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

scaler_x.fit(train_df[fitur])
scaler_y.fit(train_df[['Target']])

train_x = scaler_x.transform(train_df[fitur])
val_x = scaler_x.transform(val_df[fitur])
test_x = scaler_x.transform(test_df[fitur])

train_y = scaler_y.transform(train_df[['Target']])
val_y = scaler_y.transform(val_df[['Target']])
test_y = scaler_y.transform(test_df[['Target']])

# Sliding Window
def create_dataset(X, y, window=60):
    Xs, ys = [], []
    for i in range(window, len(X)):
        Xs.append(X[i-window:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

window_size = 60

X_train, y_train = create_dataset(train_x, train_y, window_size)
X_val, y_val = create_dataset(val_x, val_y, window_size)
X_test, y_test = create_dataset(test_x, test_y, window_size)

# Modelling 
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1, activation='linear')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse'
)

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Uji Latih
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Prediksi
y_pred_scaled = model.predict(X_test)

y_true = scaler_y.inverse_transform(y_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

y_true = y_true.flatten()
y_pred = y_pred.flatten()

bias = np.mean(y_true - y_pred)
y_pred = y_pred + bias

# Evaluasi Regresi
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# EVALUASI ARAH (KLASIFIKASI TURUNAN)
current_price = test_df['Close'].iloc[window_size:].values

y_true_dir = (y_true > current_price).astype(int)
y_pred_dir = (y_pred > current_price).astype(int)

acc = accuracy_score(y_true_dir, y_pred_dir)
prec = precision_score(y_true_dir, y_pred_dir)
rec = recall_score(y_true_dir, y_pred_dir)

print(f"Akurasi Arah: {acc:.2f}")
print(f"Precision: {prec:.2f}")
print(f"Recall: {rec:.2f}")

# VISUALISASI
plt.figure(figsize=(12,6))
plt.plot(y_true[-100:], label='Real Price')
plt.plot(y_pred[-100:], label='Predicted Price')
plt.legend()
plt.title("Real vs Predicted Price")
plt.show()
