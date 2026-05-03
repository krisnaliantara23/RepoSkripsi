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
# FITUR ENGINEERING
# ============================================================
df['EMA_diff'] = (df['EMA_20'] - df['EMA_50']) / df['Close']
df['PSAR_diff'] = (df['Close'] - df['PSAR']) / df['Close']
df['BB_width'] = (df[bbu_col] - df[bbl_col]) / df['Close']
df['BB_position'] = (df['Close'] - df[bbl_col]) / (df[bbu_col] - df[bbl_col] + 1e-9)
df['ATR_pct'] = df['ATRr_14'] / df['Close']
df['RSI_norm'] = df['RSI_14'] / 100.0
df['MACD_norm'] = df['MACD_12_26_9'] / df['Close']
df['MACDs_norm'] = df['MACDs_12_26_9'] / df['Close']
df['swing_high_pct'] = (df['swing_high'] - df['Close']) / df['Close']
df['swing_low_pct'] = (df['Close'] - df['swing_low']) / df['Close']
df['OB_bull_pct'] = df['OB_bull'] / (df['Close'] + 1e-9)
df['OB_bear_pct'] = df['OB_bear'] / (df['Close'] + 1e-9)
df['FVG_up_pct'] = df['FVG_up'] / (df['Close'] + 1e-9)
df['FVG_down_pct'] = df['FVG_down'] / (df['Close'] + 1e-9)

future_step = 3
df['Target_return'] = df['Close'].pct_change(periods=future_step).shift(-future_step)
df[['swing_high','swing_low']] = df[['swing_high','swing_low']].ffill().bfill()
df[['swing_high_pct','swing_low_pct']] = df[['swing_high_pct','swing_low_pct']].ffill().bfill()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

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
# UTILITY FUNCTIONS
# ============================================================
def create_dataset(X, y, window=60):
    Xs, ys = [], []
    for i in range(window, len(X)):
        Xs.append(X[i-window:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

window_size = 60

# ============================================================
# EXPERIMENT SETUP
# ============================================================

# 1. OHLC ONLY
fitur_ohlc = ['Open', 'High', 'Low', 'Close']

# 2. OHLC + INDIKATOR
fitur_indikator = [
    'Open', 'High', 'Low', 'Close',
    'EMA_diff', 'PSAR_diff', 'BB_width', 'BB_position',
    'ATR_pct', 'RSI_norm', 'MACD_norm', 'MACDs_norm',
]

# 3. OHLC + INDIKATOR + SMC (FULL MODEL)
fitur_smc = [
    'Open', 'High', 'Low', 'Close',
    'EMA_diff', 'PSAR_diff', 'BB_width', 'BB_position',
    'ATR_pct', 'RSI_norm', 'MACD_norm', 'MACDs_norm',
    'swing_high_pct', 'swing_low_pct',
    'OB_bull_pct', 'OB_bear_pct',
    'FVG_up_pct', 'FVG_down_pct',
]

def run_experiment(fitur_list, label, epochs=30):
    print(f"\n===== RUNNING EXPERIMENT: {label} =====")
    
    # Scalers
    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    
    scaler_x.fit(train_df[fitur_list])
    scaler_y.fit(train_df[['Target_return']])
    
    train_x = scaler_x.transform(train_df[fitur_list])
    val_x   = scaler_x.transform(val_df[fitur_list])
    test_x  = scaler_x.transform(test_df[fitur_list])
    
    train_y = scaler_y.transform(train_df[['Target_return']])
    val_y   = scaler_y.transform(val_df[['Target_return']])
    test_y  = scaler_y.transform(test_df[['Target_return']])
    
    # Datasets
    X_train, y_train = create_dataset(train_x, train_y, window_size)
    X_val, y_val     = create_dataset(val_x, val_y, window_size)
    X_test, y_test   = create_dataset(test_x, test_y, window_size)
    
    # Model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='tanh')
    ])
    
    model.compile(optimizer=Adam(0.001), loss='huber')
    
    # Train
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        verbose=0,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
    )
    
    # Predict
    y_pred_scaled = model.predict(X_test)
    y_true = scaler_y.inverse_transform(y_test).flatten()
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
    
    # Reconstruct Prices
    current_price = test_df['Close'].iloc[window_size:].values
    y_true_price = current_price * (1 + y_true)
    y_pred_price = current_price * (1 + y_pred)
    
    # Metrics
    acc = accuracy_score((y_true > 0), (y_pred > 0))
    mae = mean_absolute_error(y_true_price, y_pred_price)
    rmse = np.sqrt(mean_squared_error(y_true_price, y_pred_price))
    mape = mean_absolute_percentage_error(y_true_price, y_pred_price) * 100
    
    print(f"Accuracy: {acc:.4f}")
    print(f"MAE:      {mae:.4f}")
    print(f"RMSE:     {rmse:.4f}")
    print(f"MAPE:     {mape:.4f}%")
    
    return {
        'acc': acc, 'mae': mae, 'rmse': rmse, 'mape': mape,
        'y_true': y_true_price, 'y_pred': y_pred_price
    }

# Run the 3 experiments
res1 = run_experiment(fitur_ohlc, "OHLC", epochs=30)
res2 = run_experiment(fitur_indikator, "OHLC + Indikator", epochs=30)
res3 = run_experiment(fitur_smc, "OHLC + Indikator + SMC", epochs=30)

# ============================================================
# TABEL RINGKASAN ERROR
# ============================================================
results_df = pd.DataFrame({
    'Model': ['OHLC', 'OHLC + Indikator', 'OHLC + Indikator + SMC'],
    'Accuracy': [res1['acc'], res2['acc'], res3['acc']],
    'MAE': [res1['mae'], res2['mae'], res3['mae']],
    'RMSE': [res1['rmse'], res2['rmse'], res3['rmse']],
    'MAPE (%)': [res1['mape'], res2['mape'], res3['mape']]
})
print("\n===== RINGKASAN HASIL EVALUASI =====")
print(results_df.to_string(index=False))

# ============================================================
# GRAFIK PERBANDINGAN AKURASI & ERROR
# ============================================================
labels = ['OHLC', 'OHLC + Indikator', 'OHLC + Indikator + SMC']
metrics = ['Accuracy', 'MAE', 'RMSE']
data = [
    [res1['acc'], res2['acc'], res3['acc']],
    [res1['mae'], res2['mae'], res3['mae']],
    [res1['rmse'], res2['rmse'], res3['rmse']]
]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

colors = ['skyblue', 'lightgreen', 'salmon']

for i, metric in enumerate(metrics):
    bars = axes[i].bar(labels, data[i], color=colors)
    axes[i].set_title(f'Perbandingan {metric}')
    axes[i].set_ylabel(metric)
    axes[i].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Tambahkan label nilai di atas bar
    for bar in bars:
        yval = bar.get_height()
        axes[i].text(bar.get_x() + bar.get_width()/2, yval * 1.01, f'{yval:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('comparison_metrics.png', dpi=150)

# ============================================================
# GRAFIK HARGA (REAL VS PREDICTED)
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(12, 18))

# Plot Experiment 1
axes[0].plot(res1['y_true'], label='Real Price', color='blue', alpha=0.7)
axes[0].plot(res1['y_pred'], label='Predicted Price', color='red', linestyle='--', alpha=0.7)
axes[0].set_title(f"OHLC - Real vs Predicted Price (MAE: {res1['mae']:.2f})")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot Experiment 2
axes[1].plot(res2['y_true'], label='Real Price', color='blue', alpha=0.7)
axes[1].plot(res2['y_pred'], label='Predicted Price', color='green', linestyle='--', alpha=0.7)
axes[1].set_title(f"OHLC + Indikator - Real vs Predicted Price (MAE: {res2['mae']:.2f})")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot Experiment 3
axes[2].plot(res3['y_true'], label='Real Price', color='blue', alpha=0.7)
axes[2].plot(res3['y_pred'], label='Predicted Price', color='orange', linestyle='--', alpha=0.7)
axes[2].set_title(f"OHLC + Indikator + SMC - Real vs Predicted Price (MAE: {res3['mae']:.2f})")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comparison_prices.png', dpi=150)

print("\nEksperimen selesai! Grafik perbandingan disimpan ke 'comparison_metrics.png' dan 'comparison_prices.png'")

