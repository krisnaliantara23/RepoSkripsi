import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, accuracy_score,
    precision_score, recall_score, mean_absolute_percentage_error,
    f1_score, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dropout, Dense, BatchNormalization,
    Bidirectional, Conv1D, MaxPooling1D, Flatten, Input, concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import tensorflow as tf
import random
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# REPRODUCIBILITY
# ============================================================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv('XAUUSD.vxc_H1.csv', sep='\t')
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
df.ta.macd(append=True)
df.ta.bbands(length=20, append=True)
df.ta.stoch(append=True)        # Stochastic Oscillator — konfirmasi momentum
df.ta.adx(append=True)          # ADX — kekuatan trend
df.ta.cci(length=20, append=True)  # CCI — overbought/oversold

# Ambil nama kolom secara dinamis
bbu_col  = [c for c in df.columns if 'BBU' in c][0]
bbl_col  = [c for c in df.columns if 'BBL' in c][0]
ema20_col = [c for c in df.columns if 'EMA_20' in c][0]
ema50_col = [c for c in df.columns if 'EMA_50' in c][0]
stochk_col = [c for c in df.columns if 'STOCHk' in c][0]
stochd_col = [c for c in df.columns if 'STOCHd' in c][0]
adx_col   = [c for c in df.columns if c.startswith('ADX_')][0]
cci_col   = [c for c in df.columns if 'CCI' in c][0]

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
    df['FVG_up']   = np.where(df['Low'] > df['High'].shift(2), df['Low'] - df['High'].shift(2), 0)
    df['FVG_down'] = np.where(df['High'] < df['Low'].shift(2), df['Low'].shift(2) - df['High'], 0)
    return df

df = swing_high_low(df)
df = order_block(df)
df = fair_value_gap(df)

# ============================================================
# FITUR ENGINEERING
# ============================================================
df['EMA_diff']    = (df[ema20_col] - df[ema50_col]) / df['Close']
df['PSAR_diff']   = (df['Close'] - df['PSAR']) / df['Close']
df['BB_width']    = (df[bbu_col] - df[bbl_col]) / df['Close']
df['BB_position'] = (df['Close'] - df[bbl_col]) / (df[bbu_col] - df[bbl_col] + 1e-9)
df['ATR_pct']     = df['ATRr_14'] / df['Close']
df['RSI_norm']    = df['RSI_14'] / 100.0
df['MACD_norm']   = df['MACD_12_26_9'] / df['Close']
df['MACDs_norm']  = df['MACDs_12_26_9'] / df['Close']
df['Stoch_K']     = df[stochk_col] / 100.0
df['Stoch_D']     = df[stochd_col] / 100.0
df['ADX_norm']    = df[adx_col] / 100.0
df['CCI_norm']    = df[cci_col] / 200.0   # CCI biasanya -200 to +200

df['swing_high_pct'] = (df['swing_high'] - df['Close']) / df['Close']
df['swing_low_pct']  = (df['Close'] - df['swing_low']) / df['Close']

# PERBAIKAN #3: SMC fitur sebagai biner (aktif/tidak aktif)
df['OB_bull_active'] = (df['OB_bull'] > 0).astype(float)
df['OB_bear_active'] = (df['OB_bear'] > 0).astype(float)
df['FVG_up_active']  = (df['FVG_up']  > 0).astype(float)
df['FVG_down_active']= (df['FVG_down'] > 0).astype(float)

# Fitur price action tambahan
df['candle_body']  = (df['Close'] - df['Open']) / df['ATRr_14'].replace(0, 1e-9)
df['upper_shadow'] = (df['High'] - df[['Open','Close']].max(axis=1)) / df['ATRr_14'].replace(0, 1e-9)
df['lower_shadow'] = (df[['Open','Close']].min(axis=1) - df['Low']) / df['ATRr_14'].replace(0, 1e-9)
df['volume_norm']  = df['Volume'] / (df['Volume'].rolling(20).mean() + 1e-9)

# PERBAIKAN #4: Target sebagai klasifikasi biner (naik/turun)
future_step = 3
df['Target_dir'] = (df['Close'].shift(-future_step) > df['Close']).astype(int)

# Swing fill
df[['swing_high', 'swing_low']] = df[['swing_high', 'swing_low']].ffill().bfill()
df[['swing_high_pct', 'swing_low_pct']] = df[['swing_high_pct', 'swing_low_pct']].ffill().bfill()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# CEK CLASS BALANCE
print("=" * 50)
print("CEK CLASS BALANCE TARGET:")
balance = df['Target_dir'].value_counts(normalize=True)
print(f"  Turun (0): {balance.get(0, 0):.2%}")
print(f"  Naik  (1): {balance.get(1, 0):.2%}")
print("=" * 50)

# ============================================================
# SPLIT DATA
# ============================================================
train_size = int(len(df) * 0.8)
val_size   = int(len(df) * 0.1)

train_df = df[:train_size]
val_df   = df[train_size:train_size + val_size]
test_df  = df[train_size + val_size:]

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def create_dataset(X, y, window=32):
    Xs, ys = [], []
    for i in range(window, len(X)):
        Xs.append(X[i - window:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

window_size = 32

# ============================================================
# EXPERIMENT SETUP
# ============================================================

# 1. OHLC ONLY
fitur_ohlc = ['Open', 'High', 'Low', 'Close']

# 2. OHLC + INDIKATOR (dengan indikator baru)
fitur_indikator = [
    'Open', 'High', 'Low', 'Close',
    'EMA_diff', 'PSAR_diff', 'BB_width', 'BB_position',
    'ATR_pct', 'RSI_norm', 'MACD_norm', 'MACDs_norm',
    'Stoch_K', 'Stoch_D', 'ADX_norm', 'CCI_norm',
    'candle_body', 'upper_shadow', 'lower_shadow', 'volume_norm',
]

# 3. OHLC + INDIKATOR + SMC BINER
fitur_smc = [
    'Open', 'High', 'Low', 'Close',
    'EMA_diff', 'PSAR_diff', 'BB_width', 'BB_position',
    'ATR_pct', 'RSI_norm', 'MACD_norm', 'MACDs_norm',
    'Stoch_K', 'Stoch_D', 'ADX_norm', 'CCI_norm',
    'candle_body', 'upper_shadow', 'lower_shadow', 'volume_norm',
    'swing_high_pct', 'swing_low_pct',
    'OB_bull_active', 'OB_bear_active',
    'FVG_up_active', 'FVG_down_active',
]

# ============================================================
# MODEL BUILDER
# ============================================================

def build_bilstm(input_shape):
    """PERBAIKAN #5a: Bidirectional LSTM"""
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')   # PERBAIKAN #4: sigmoid untuk klasifikasi
    ])
    return model

def build_cnn_lstm(input_shape):
    """PERBAIKAN #5b: CNN-LSTM Hybrid"""
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

def run_experiment(fitur_list, label, model_type='bilstm', epochs=100):
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {label} | Model: {model_type.upper()}")
    print(f"{'='*60}")

    # Scaler — fit HANYA dari train
    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    scaler_x.fit(train_df[fitur_list])

    train_x = scaler_x.transform(train_df[fitur_list])
    val_x   = scaler_x.transform(val_df[fitur_list])
    test_x  = scaler_x.transform(test_df[fitur_list])

    train_y = train_df['Target_dir'].values
    val_y   = val_df['Target_dir'].values
    test_y  = test_df['Target_dir'].values

    X_train, y_train = create_dataset(train_x, train_y, window_size)
    X_val,   y_val   = create_dataset(val_x,   val_y,   window_size)
    X_test,  y_test  = create_dataset(test_x,  test_y,  window_size)

    # PERBAIKAN #2: Class weight untuk atasi imbalance
    classes = np.unique(y_train)
    cw = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = {int(c): w for c, w in zip(classes, cw)}
    print(f"Class weights: {class_weight_dict}")

    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    if model_type == 'bilstm':
        model = build_bilstm(input_shape)
    else:
        model = build_cnn_lstm(input_shape)

    # PERBAIKAN #4: loss binary_crossentropy
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        class_weight=class_weight_dict,   # PERBAIKAN #2
        verbose=1,
        callbacks=callbacks
    )

    # Predict
    y_prob  = model.predict(X_test, verbose=0).flatten()
    y_pred  = (y_prob >= 0.5).astype(int)
    y_true  = y_test

    # Metrics
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    # Harga untuk MAE/RMSE/MAPE
    current_price = test_df['Close'].values[window_size: window_size + len(y_pred)]
    close_future  = test_df['Close'].shift(-future_step).values[window_size: window_size + len(y_pred)]
    y_pred_price  = current_price * (1 + np.where(y_pred == 1, 0.001, -0.001))

    # PERBAIKAN: buang baris yang NaN (karena shift di ujung test_df)
    valid_mask = ~np.isnan(close_future)
    close_future_valid = close_future[valid_mask]
    y_pred_price_valid = y_pred_price[valid_mask]

    mae  = mean_absolute_error(close_future_valid, y_pred_price_valid)
    rmse = np.sqrt(mean_squared_error(close_future_valid, y_pred_price_valid))

    print(f"\nHasil:")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  MAE       : {mae:.4f}")
    print(f"  RMSE      : {rmse:.4f}")

    return {
        'label': label, 'model_type': model_type,
        'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1,
        'mae': mae, 'rmse': rmse,
        'y_true': y_true, 'y_pred': y_pred, 'y_prob': y_prob,
        'history': history, 'cm': cm
    }

# ============================================================
# RUN EXPERIMENTS
# BiLSTM vs CNN-LSTM pada fitur SMC (full feature set)
# ============================================================
res1 = run_experiment(fitur_ohlc,       "OHLC",                   model_type='bilstm',   epochs=100)
res2 = run_experiment(fitur_indikator,  "OHLC + Indikator",       model_type='bilstm',   epochs=100)
res3 = run_experiment(fitur_smc,        "OHLC + Indikator + SMC", model_type='bilstm',   epochs=100)
res4 = run_experiment(fitur_smc,        "OHLC + Indikator + SMC", model_type='cnn_lstm', epochs=100)

results = [res1, res2, res3, res4]

# ============================================================
# TABEL RINGKASAN
# ============================================================
summary_df = pd.DataFrame([{
    'Model'     : r['label'],
    'Arsitektur': r['model_type'].upper(),
    'Accuracy'  : round(r['acc'], 4),
    'Precision' : round(r['prec'], 4),
    'Recall'    : round(r['rec'], 4),
    'F1-Score'  : round(r['f1'], 4),
    'MAE'       : round(r['mae'], 2),
    'RMSE'      : round(r['rmse'], 2),
} for r in results])

print("\n" + "="*80)
print("RINGKASAN HASIL EVALUASI")
print("="*80)
print(summary_df.to_string(index=False))
summary_df.to_csv('hasil_evaluasi.csv', index=False)

# ============================================================
# GRAFIK 1: PERBANDINGAN METRIK
# ============================================================
exp_labels = [f"{r['label']}\n({r['model_type'].upper()})" for r in results]
metrics_to_plot = ['acc', 'prec', 'rec', 'f1']
metric_names    = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']

fig, axes = plt.subplots(1, 4, figsize=(20, 6))
fig.suptitle('Perbandingan Metrik Klasifikasi — XAUUSD H1', fontsize=14, fontweight='bold')

for i, (metric, name) in enumerate(zip(metrics_to_plot, metric_names)):
    vals = [r[metric] for r in results]
    bars = axes[i].bar(exp_labels, vals, color=colors, edgecolor='black', linewidth=0.5)
    axes[i].set_title(name, fontweight='bold')
    axes[i].set_ylim(0, 1.1)
    axes[i].axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Random Baseline')
    axes[i].grid(axis='y', linestyle='--', alpha=0.5)
    axes[i].tick_params(axis='x', labelsize=8)
    for bar, val in zip(bars, vals):
        axes[i].text(bar.get_x() + bar.get_width()/2, val + 0.02,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

axes[0].legend(fontsize=8)
plt.tight_layout()
plt.savefig('comparison_metrics.png', dpi=150, bbox_inches='tight')
print("\nGrafik metrik disimpan ke 'comparison_metrics.png'")

# ============================================================
# GRAFIK 2: CONFUSION MATRIX
# ============================================================
fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5))
fig.suptitle('Confusion Matrix per Model', fontsize=14, fontweight='bold')

for ax, r in zip(axes, results):
    sns.heatmap(r['cm'], annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Turun', 'Naik'], yticklabels=['Turun', 'Naik'])
    ax.set_title(f"{r['label']}\n({r['model_type'].upper()})", fontsize=9)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
print("Confusion matrix disimpan ke 'confusion_matrix.png'")

# ============================================================
# GRAFIK 3: TRAINING HISTORY (LOSS)
# ============================================================
fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
fig.suptitle('Training vs Validation Loss', fontsize=14, fontweight='bold')

for ax, r in zip(axes, results):
    h = r['history'].history
    ax.plot(h['loss'],     label='Train Loss', color='blue')
    ax.plot(h['val_loss'], label='Val Loss',   color='orange')
    ax.set_title(f"{r['label']}\n({r['model_type'].upper()})", fontsize=9)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
print("Training history disimpan ke 'training_history.png'")

# ============================================================
# GRAFIK 4: PROBABILITAS PREDIKSI (model terbaik)
# ============================================================
best_res = max(results, key=lambda r: r['f1'])
print(f"\nModel terbaik berdasarkan F1: {best_res['label']} ({best_res['model_type'].upper()})")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
fig.suptitle(f"Prediksi Model Terbaik: {best_res['label']} ({best_res['model_type'].upper()})", fontweight='bold')

# Plot probabilitas
n_show = min(200, len(best_res['y_prob']))
ax1.plot(best_res['y_prob'][:n_show], color='purple', alpha=0.7, label='Prob Naik')
ax1.axhline(0.5, color='red', linestyle='--', label='Threshold 0.5')
ax1.fill_between(range(n_show), 0.5, best_res['y_prob'][:n_show],
                  where=(best_res['y_prob'][:n_show] > 0.5), alpha=0.2, color='green')
ax1.fill_between(range(n_show), 0.5, best_res['y_prob'][:n_show],
                  where=(best_res['y_prob'][:n_show] <= 0.5), alpha=0.2, color='red')
ax1.set_ylabel('Probabilitas Naik')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot prediksi vs actual
ax2.scatter(range(n_show), best_res['y_true'][:n_show], label='Actual', alpha=0.4, s=10, color='blue')
ax2.scatter(range(n_show), best_res['y_pred'][:n_show], label='Predicted', alpha=0.4, s=10, color='red', marker='x')
ax2.set_ylabel('Arah (0=Turun, 1=Naik)')
ax2.set_xlabel('Sample ke-')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('best_model_predictions.png', dpi=150, bbox_inches='tight')
print("Prediksi model terbaik disimpan ke 'best_model_predictions.png'")

print("\n" + "="*60)
print("EKSPERIMEN SELESAI!")
print(f"File output: hasil_evaluasi.csv, comparison_metrics.png,")
print(f"             confusion_matrix.png, training_history.png,")
print(f"             best_model_predictions.png")
print("="*60)