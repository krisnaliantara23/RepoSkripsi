import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv('XAUUSD_H1_1Tahun.csv', sep='\t')
df['Date'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'])
df.set_index('Date', inplace=True)
df.rename(columns={
    '<OPEN>': 'Open', '<HIGH>': 'High',
    '<LOW>': 'Low', '<CLOSE>': 'Close', '<TICKVOL>': 'Volume'
}, inplace=True)

# ============================================================
# INDIKATOR TEKNIKAL
# ============================================================
df.ta.ema(length=20, append=True)
df.ta.ema(length=50, append=True)
df.ta.rsi(length=14, append=True)
df.ta.atr(length=14, append=True)
df.ta.macd(append=True)       # tambahan: MACD
bb_cols = df.ta.bbands(length=20, append=True)  # tambahan: Bollinger Bands
bbu_col = bb_cols.filter(like='BBU').columns[0]
bbl_col = bb_cols.filter(like='BBL').columns[0]

psar = df.ta.psar()
df['PSAR'] = psar.iloc[:, 0].fillna(psar.iloc[:, 1])

# ============================================================
# SMC (SMART MONEY CONCEPT)
# ============================================================
def swing_high_low(df, window=5):
    df['swing_high'] = np.where(
        df['High'] == df['High'].rolling(window=window, center=True).max(),
        df['High'], np.nan
    )
    df['swing_low'] = np.where(
        df['Low'] == df['Low'].rolling(window=window, center=True).min(),
        df['Low'], np.nan
    )
    return df

def order_block(df):
    df['OB_bull'] = np.where(
        (df['Close'] > df['Open']) & (df['Close'].shift(1) < df['Open'].shift(1)),
        df['Low'], 0
    )
    df['OB_bear'] = np.where(
        (df['Close'] < df['Open']) & (df['Close'].shift(1) > df['Open'].shift(1)),
        df['High'], 0
    )
    return df

def fair_value_gap(df):
    df['FVG_up'] = np.where(df['Low'] > df['High'].shift(2), df['Low'] - df['High'].shift(2), 0)
    df['FVG_down'] = np.where(df['High'] < df['Low'].shift(2), df['Low'].shift(2) - df['High'], 0)
    return df

df = swing_high_low(df)
df = order_block(df)
df = fair_value_gap(df)

# ============================================================
# FITUR ENGINEERING — KUNCI PERBAIKAN
# ============================================================

# 1. Return & log return (stasioner, lebih mudah dipelajari LSTM)
df['Return'] = df['Close'].pct_change()
df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
# 2. Fitur relatif (normalisasi alami terhadap harga saat ini)
print("DEBUG - Columns available:", df.columns.tolist())
df['EMA_diff'] = (df['EMA_20'] - df['EMA_50']) / df['Close']   # dalam % harga
df['PSAR_diff'] = (df['Close'] - df['PSAR']) / df['Close']      # dalam % harga
df['BB_width'] = (df[bbu_col] - df[bbl_col]) / df['Close']  # lebar BB relatif
df['BB_position'] = (df['Close'] - df[bbl_col]) / (df[bbu_col] - df[bbl_col] + 1e-9)
df['ATR_pct'] = df['ATRr_14'] / df['Close']                     # ATR relatif

# 3. Volatility & range relatif
df['Volatility'] = (df['High'] - df['Low']) / df['Close']
df['Body_size'] = abs(df['Close'] - df['Open']) / df['Close']
df['Upper_shadow'] = (df['High'] - df[['Open','Close']].max(axis=1)) / df['Close']
df['Lower_shadow'] = (df[['Open','Close']].min(axis=1) - df['Low']) / df['Close']

# 4. RSI normalisasi 0-1
df['RSI_norm'] = df['RSI_14'] / 100.0

# 5. MACD normalisasi
df['MACD_norm'] = df['MACD_12_26_9'] / df['Close']
df['MACDs_norm'] = df['MACDs_12_26_9'] / df['Close']

# 6. Swing dalam % harga
df['swing_high_pct'] = (df['swing_high'] - df['Close']) / df['Close']
df['swing_low_pct'] = (df['Close'] - df['swing_low']) / df['Close']

# 7. OB & FVG normalisasi
df['OB_bull_pct'] = df['OB_bull'] / (df['Close'] + 1e-9)
df['OB_bear_pct'] = df['OB_bear'] / (df['Close'] + 1e-9)
df['FVG_up_pct'] = df['FVG_up'] / (df['Close'] + 1e-9)
df['FVG_down_pct'] = df['FVG_down'] / (df['Close'] + 1e-9)

# 8. Volume normalisasi (jika ada)
if 'Volume' in df.columns:
    df['Volume_norm'] = df['Volume'] / (df['Volume'].rolling(20).mean() + 1e-9)

# ============================================================
# TARGET — PREDIKSI RETURN, BUKAN HARGA ABSOLUT
# ============================================================
# Memprediksi % perubahan harga (return) jauh lebih stabil
# daripada prediksi harga absolut yang rentan scale mismatch
df['Target_return'] = df['Close'].pct_change().shift(-1)  # return 1 bar ke depan

# Handle NaN & Inf
df[['swing_high','swing_low']] = df[['swing_high','swing_low']].ffill().bfill()
df[['swing_high_pct','swing_low_pct']] = df[['swing_high_pct','swing_low_pct']].ffill().bfill()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# ============================================================
# DEFINISI FITUR (semua sudah relatif/ternormalisasi)
# ============================================================
fitur = [
    # Indikator normalisasi relatif
    'EMA_diff', 'PSAR_diff', 'BB_width', 'BB_position',
    'ATR_pct', 'RSI_norm', 'MACD_norm', 'MACDs_norm',

    # SMC normalisasi
    'swing_high_pct', 'swing_low_pct',
    'OB_bull_pct', 'OB_bear_pct',
    'FVG_up_pct', 'FVG_down_pct',

    # Price action relatif
    'Return', 'Log_Return',
    'Volatility', 'Body_size', 'Upper_shadow', 'Lower_shadow',
]

# Tambah volume jika ada
if 'Volume_norm' in df.columns:
    fitur.append('Volume_norm')

print(f"Total fitur: {len(fitur)}")
print(f"Total data: {len(df)}")

# ============================================================
# SPLIT DATA
# ============================================================
train_size = int(len(df) * 0.8)
val_size = int(len(df) * 0.1)

train_df = df[:train_size]
val_df   = df[train_size:train_size+val_size]
test_df  = df[train_size+val_size:]

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# ============================================================
# SCALING — hanya fitur X yang perlu discale
# Target (return) sudah kecil, tapi tetap discale agar stabil
# ============================================================
scaler_x = MinMaxScaler(feature_range=(-1, 1))  # range -1,1 lebih baik untuk LSTM
scaler_y = MinMaxScaler(feature_range=(-1, 1))

scaler_x.fit(train_df[fitur])
scaler_y.fit(train_df[['Target_return']])

train_x = scaler_x.transform(train_df[fitur])
val_x   = scaler_x.transform(val_df[fitur])
test_x  = scaler_x.transform(test_df[fitur])

train_y = scaler_y.transform(train_df[['Target_return']])
val_y   = scaler_y.transform(val_df[['Target_return']])
test_y  = scaler_y.transform(test_df[['Target_return']])

# ============================================================
# SLIDING WINDOW
# ============================================================
def create_dataset(X, y, window=60):
    Xs, ys = [], []
    for i in range(window, len(X)):
        Xs.append(X[i-window:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

window_size = 60

X_train, y_train = create_dataset(train_x, train_y, window_size)
X_val,   y_val   = create_dataset(val_x,   val_y,   window_size)
X_test,  y_test  = create_dataset(test_x,  test_y,  window_size)

print(f"Shape X_train: {X_train.shape}")

# ============================================================
# MODEL — dengan BatchNormalization untuk stabilitas
# ============================================================
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    BatchNormalization(),
    Dropout(0.3),

    LSTM(64, return_sequences=True),
    BatchNormalization(),
    Dropout(0.2),

    LSTM(32, return_sequences=False),
    Dropout(0.2),

    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='huber')  # Huber loss lebih robust terhadap outlier
model.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# ============================================================
# PREDIKSI — rekonstruksi harga dari prediksi return
# ============================================================
y_pred_scaled = model.predict(X_test)

# Inverse transform: kembali ke return asli
y_true_return = scaler_y.inverse_transform(y_test).flatten()
y_pred_return = scaler_y.inverse_transform(y_pred_scaled).flatten()

# Rekonstruksi harga dari return prediksi
# current_price = harga penutupan saat ini (bar ke-window_size dan seterusnya)
current_price = test_df['Close'].iloc[window_size:].values

# Harga aktual berikutnya
y_true_price = current_price * (1 + y_true_return)

# Harga prediksi berikutnya
y_pred_price = current_price * (1 + y_pred_return)

# ============================================================
# EVALUASI REGRESI
# ============================================================
mae  = mean_absolute_error(y_true_price, y_pred_price)
rmse = np.sqrt(mean_squared_error(y_true_price, y_pred_price))
mape = mean_absolute_percentage_error(y_true_price, y_pred_price) * 100

print("\n========== EVALUASI REGRESI ==========")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.4f}%")

# ============================================================
# EVALUASI ARAH (KLASIFIKASI)
# ============================================================
y_true_dir = (y_true_return > 0).astype(int)
y_pred_dir = (y_pred_return > 0).astype(int)

acc  = accuracy_score(y_true_dir, y_pred_dir)
prec = precision_score(y_true_dir, y_pred_dir, zero_division=0)
rec  = recall_score(y_true_dir, y_pred_dir, zero_division=0)

print("\n========== EVALUASI ARAH ==========")
print(f"Akurasi Arah: {acc:.4f}")
print(f"Precision:    {prec:.4f}")
print(f"Recall:       {rec:.4f}")

# ============================================================
# VISUALISASI
# ============================================================
fig, axes = plt.subplots(3, 2, figsize=(16, 18))

# Plot 1: Training history
axes[0, 0].plot(history.history['loss'], label='Train Loss', color='blue')
axes[0, 0].plot(history.history['val_loss'], label='Val Loss', color='orange')
axes[0, 0].set_title('Training History')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Huber Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Real vs Predicted Price (100 data terakhir)
n = 200
axes[0, 1].plot(y_true_price[-n:], label='Real Price', color='blue', linewidth=1.5)
axes[0, 1].plot(y_pred_price[-n:], label='Predicted Price', color='red', linewidth=1.2, alpha=0.8)
axes[0, 1].set_title(f'Real vs Predicted Price (last {n} bars)')
axes[0, 1].set_xlabel('Bar')
axes[0, 1].set_ylabel('Price (USD)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Predicted Return vs True Return
axes[1, 0].plot(y_true_return[-n:], label='True Return', color='blue', linewidth=1.2, alpha=0.7)
axes[1, 0].plot(y_pred_return[-n:], label='Predicted Return', color='red', linewidth=1.2, alpha=0.8)
axes[1, 0].axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
axes[1, 0].set_title(f'Return Prediction (last {n} bars)')
axes[1, 0].set_xlabel('Bar')
axes[1, 0].set_ylabel('Return')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Scatter Plot Actual vs Predicted Return
axes[1, 1].scatter(y_true_return, y_pred_return, alpha=0.5, color='purple', s=10)
axes[1, 1].plot([y_true_return.min(), y_true_return.max()], [y_true_return.min(), y_true_return.max()], 'r--', lw=2)
axes[1, 1].set_title(f'Scatter: Actual vs Predicted Return (Corr: {np.corrcoef(y_true_return, y_pred_return)[0,1]:.4f})')
axes[1, 1].set_xlabel('Actual Return')
axes[1, 1].set_ylabel('Predicted Return')
axes[1, 1].grid(True, alpha=0.3)

# Plot 5: Error Distribution
errors = y_true_price - y_pred_price
axes[2, 0].hist(errors, bins=50, color='teal', alpha=0.7, edgecolor='black')
axes[2, 0].axvline(x=0, color='red', linestyle='--')
axes[2, 0].set_title(f'Price Error Distribution (Mean Error: {np.mean(errors):.4f})')
axes[2, 0].set_xlabel('Error (Actual - Predicted)')
axes[2, 0].set_ylabel('Frequency')
axes[2, 0].grid(True, alpha=0.3)

# Plot 6: Cumulative Return (Strategy Simple)
# Jika prediksi > 0 beli, jika < 0 jual
strategy_returns = np.sign(y_pred_return) * y_true_return
cum_strategy = np.cumsum(strategy_returns)
cum_market = np.cumsum(y_true_return)

axes[2, 1].plot(cum_strategy, label='Strategy (Long/Short)', color='green')
axes[2, 1].plot(cum_market, label='Market (Buy & Hold)', color='gray', alpha=0.5)
axes[2, 1].set_title('Cumulative Return Strategy (Backtest Sederhana)')
axes[2, 1].set_xlabel('Bar')
axes[2, 1].set_ylabel('Cumulative Return')
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('xauusd_prediction_result.png', dpi=150, bbox_inches='tight')
# plt.show()

print("\nSelesai! Plot disimpan ke 'xauusd_prediction_result.png'")