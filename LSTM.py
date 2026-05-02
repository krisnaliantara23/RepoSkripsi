"""
=============================================================
  XAUUSD H1 - LSTM Price Prediction dengan SMC Features
  Diterjemahkan dari Pine Script SMC ke Python
=============================================================
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             mean_absolute_percentage_error,
                             accuracy_score, precision_score, recall_score)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 1. LOAD DATA
# ============================================================
df = pd.read_csv('XAUUSD_H1_1Tahun.csv', sep='\t')
df['Date'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'])
df.set_index('Date', inplace=True)
df.rename(columns={
    '<OPEN>': 'Open', '<HIGH>': 'High',
    '<LOW>': 'Low', '<CLOSE>': 'Close',
    '<TICKVOL>': 'Volume'
}, inplace=True)
df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
df = df.sort_index()
print(f"[INFO] Data loaded: {len(df)} baris | {df.index[0]} s/d {df.index[-1]}")


# ============================================================
# 2. INDIKATOR TEKNIKAL
# ============================================================
df.ta.ema(length=20, append=True)
df.ta.ema(length=50, append=True)
df.ta.rsi(length=14, append=True)
df.ta.atr(length=14, append=True)

# Deteksi nama kolom ATR yang benar
atr_col = None
for candidate in ['ATRr_14', 'ATR_14', 'ATRR_14']:
    if candidate in df.columns:
        atr_col = candidate
        break
if atr_col is None:
    raise ValueError("Kolom ATR tidak ditemukan! Cek versi pandas_ta kamu.")
print(f"[INFO] ATR column: {atr_col}")

# PSAR - perbaikan deprecated fillna
psar = df.ta.psar()
df['PSAR'] = psar.iloc[:, 0].bfill()


# ============================================================
# 3. SMC FEATURES (diterjemahkan dari Pine Script)
# ============================================================

# ----------------------------------------------------------
# 3a. SWING HIGH / SWING LOW
#     Sama seperti ta.pivothigh / ta.pivotlow di Pine Script
# ----------------------------------------------------------
def compute_swing(df, window=5):
    """
    Deteksi Swing High dan Swing Low.
    Swing High = high[i] adalah maksimum dalam window kiri & kanan.
    Swing Low  = low[i]  adalah minimum dalam window kiri & kanan.
    """
    highs = df['High'].values
    lows  = df['Low'].values
    n     = len(df)

    swing_high = np.zeros(n)
    swing_low  = np.zeros(n)

    for i in range(window, n - window):
        left_h  = highs[i - window:i]
        right_h = highs[i + 1:i + window + 1]
        left_l  = lows [i - window:i]
        right_l = lows [i + 1:i + window + 1]

        if highs[i] >= max(left_h) and highs[i] >= max(right_h):
            swing_high[i] = highs[i]

        if lows[i] <= min(left_l) and lows[i] <= min(right_l):
            swing_low[i] = lows[i]

    df['swing_high'] = swing_high
    df['swing_low']  = swing_low

    # Nilai terakhir swing yang valid (untuk referensi struktur)
    df['last_swing_high'] = df['swing_high'].replace(0, np.nan).ffill().fillna(0)
    df['last_swing_low']  = df['swing_low'].replace(0, np.nan).ffill().fillna(0)

    return df


# ----------------------------------------------------------
# 3b. MARKET STRUCTURE: BOS & CHoCH
#     Logika dari fungsi structure() di Pine Script
# ----------------------------------------------------------
def compute_market_structure(df, window=5):
    """
    Deteksi Break of Structure (BOS) dan Change of Character (CHoCH).
    Trend  1 = bullish, -1 = bearish, 0 = undefined.
    BOS    = konfirmasi kelanjutan trend (close melewati swing sebelumnya searah trend).
    CHoCH  = perubahan struktur (close melewati swing berlawanan arah trend).
    """
    n      = len(df)
    trend  = np.zeros(n, dtype=int)
    bos    = np.zeros(n)
    choch  = np.zeros(n)

    # Pakai swing yang sudah dihitung
    sh = df['swing_high'].values
    sl = df['swing_low'].values

    last_sh = 0.0
    last_sl = 0.0
    curr_trend = 0

    for i in range(window, n):
        c = df['Close'].iloc[i]
        o = df['Open'].iloc[i]

        if sh[i] > 0:
            last_sh = sh[i]
        if sl[i] > 0:
            last_sl = sl[i]

        if last_sh == 0 or last_sl == 0:
            trend[i] = curr_trend
            continue

        if curr_trend == 0:
            # Inisialisasi trend awal
            if c > last_sh:
                curr_trend = 1
            elif c < last_sl:
                curr_trend = -1

        elif curr_trend == 1:
            # Bullish: cari BOS ke atas atau CHoCH ke bawah
            if c > last_sh:
                bos[i] = 1          # Break of Structure bullish
            elif c < last_sl:
                choch[i]   = -1     # Change of Character → bearish
                curr_trend = -1

        elif curr_trend == -1:
            # Bearish: cari BOS ke bawah atau CHoCH ke atas
            if c < last_sl:
                bos[i] = -1         # Break of Structure bearish
            elif c > last_sh:
                choch[i]   = 1      # Change of Character → bullish
                curr_trend = 1

        trend[i] = curr_trend

    df['ms_trend'] = trend
    df['ms_bos']   = bos      #  1 = bullish BOS, -1 = bearish BOS
    df['ms_choch'] = choch    #  1 = bullish CHoCH, -1 = bearish CHoCH

    return df


# ----------------------------------------------------------
# 3c. ORDER BLOCK
#     Logika dari fnOB() dan mitigated() di Pine Script
# ----------------------------------------------------------
def compute_order_block(df, window=5):
    """
    Order Block:
    - Bullish OB: candle bearish sebelum impulse bullish (swing low terakhir).
    - Bearish OB: candle bullish sebelum impulse bearish (swing high terakhir).
    Hasilnya: top, bottom, dan midpoint (avg) dari OB aktif.
    """
    n      = len(df)
    ob_bull_top = np.zeros(n)
    ob_bull_btm = np.zeros(n)
    ob_bull_avg = np.zeros(n)
    ob_bear_top = np.zeros(n)
    ob_bear_btm = np.zeros(n)
    ob_bear_avg = np.zeros(n)
    ob_bull_active = np.zeros(n)
    ob_bear_active = np.zeros(n)

    closes = df['Close'].values
    opens  = df['Open'].values
    highs  = df['High'].values
    lows   = df['Low'].values
    trend  = df['ms_trend'].values

    for i in range(window + 1, n):
        # Bullish OB: trend bullish → cari candle bearish terakhir sebelum swing low
        if trend[i] == 1:
            for j in range(i - 1, max(i - window - 1, 0), -1):
                if closes[j] < opens[j]:  # candle bearish = kandidat bullish OB
                    ob_bull_top[i] = highs[j]
                    ob_bull_btm[i] = lows[j]
                    ob_bull_avg[i] = (highs[j] + lows[j]) / 2
                    # Aktif jika harga belum menembus bawah OB
                    ob_bull_active[i] = 1 if closes[i] > lows[j] else 0
                    break

        # Bearish OB: trend bearish → cari candle bullish terakhir sebelum swing high
        if trend[i] == -1:
            for j in range(i - 1, max(i - window - 1, 0), -1):
                if closes[j] > opens[j]:  # candle bullish = kandidat bearish OB
                    ob_bear_top[i] = highs[j]
                    ob_bear_btm[i] = lows[j]
                    ob_bear_avg[i] = (highs[j] + lows[j]) / 2
                    # Aktif jika harga belum menembus atas OB
                    ob_bear_active[i] = 1 if closes[i] < highs[j] else 0
                    break

    df['ob_bull_top']    = ob_bull_top
    df['ob_bull_btm']    = ob_bull_btm
    df['ob_bull_avg']    = ob_bull_avg
    df['ob_bull_active'] = ob_bull_active
    df['ob_bear_top']    = ob_bear_top
    df['ob_bear_btm']    = ob_bear_btm
    df['ob_bear_avg']    = ob_bear_avg
    df['ob_bear_active'] = ob_bear_active

    return df


# ----------------------------------------------------------
# 3d. FAIR VALUE GAP (FVG)
#     Logika dari dFVG() di Pine Script
# ----------------------------------------------------------
def compute_fvg(df, fvg_thresh=0.0):
    """
    Fair Value Gap:
    - Bullish FVG: low[i] > high[i-2]  (gap ke atas antara candle i-2 dan i)
    - Bearish FVG: high[i] < low[i-2]  (gap ke bawah antara candle i-2 dan i)
    fvg_thresh: filter minimum ukuran gap (dalam persen dari ATR).
    """
    n      = len(df)
    highs  = df['High'].values
    lows   = df['Low'].values
    closes = df['Close'].values
    opens  = df['Open'].values
    atr_vals = df[atr_col].values

    fvg_bull     = np.zeros(n)   # 1 = ada bullish FVG
    fvg_bear     = np.zeros(n)   # 1 = ada bearish FVG
    fvg_bull_top = np.zeros(n)
    fvg_bull_btm = np.zeros(n)
    fvg_bear_top = np.zeros(n)
    fvg_bear_btm = np.zeros(n)
    fvg_bull_mid = np.zeros(n)
    fvg_bear_mid = np.zeros(n)
    fvg_mitigated_bull = np.zeros(n)
    fvg_mitigated_bear = np.zeros(n)

    # Track FVG aktif (simpan top & btm)
    active_bull_fvg = []  # list of (top, btm)
    active_bear_fvg = []

    for i in range(2, n):
        atr_i = atr_vals[i - 1] if not np.isnan(atr_vals[i - 1]) else 0
        thresh = atr_i * fvg_thresh

        # Bullish FVG: gap antara high[i-2] dan low[i]
        if lows[i] > highs[i - 2] and (lows[i] - highs[i - 2]) > thresh:
            top = lows[i]
            btm = highs[i - 2]
            fvg_bull[i]     = 1
            fvg_bull_top[i] = top
            fvg_bull_btm[i] = btm
            fvg_bull_mid[i] = (top + btm) / 2
            active_bull_fvg.append((top, btm))

        # Bearish FVG: gap antara low[i-2] dan high[i]
        if highs[i] < lows[i - 2] and (lows[i - 2] - highs[i]) > thresh:
            top = lows[i - 2]
            btm = highs[i]
            fvg_bear[i]     = 1
            fvg_bear_top[i] = top
            fvg_bear_btm[i] = btm
            fvg_bear_mid[i] = (top + btm) / 2
            active_bear_fvg.append((top, btm))

        # Cek mitigasi: harga masuk ke dalam FVG aktif
        # Bullish FVG dimitigasi jika close < btm FVG
        for fvg in active_bull_fvg[:]:
            if closes[i] < fvg[1]:
                fvg_mitigated_bull[i] = 1
                active_bull_fvg.remove(fvg)

        # Bearish FVG dimitigasi jika close > top FVG
        for fvg in active_bear_fvg[:]:
            if closes[i] > fvg[0]:
                fvg_mitigated_bear[i] = 1
                active_bear_fvg.remove(fvg)

    df['fvg_bull']           = fvg_bull
    df['fvg_bear']           = fvg_bear
    df['fvg_bull_top']       = fvg_bull_top
    df['fvg_bull_btm']       = fvg_bull_btm
    df['fvg_bull_mid']       = fvg_bull_mid
    df['fvg_bear_top']       = fvg_bear_top
    df['fvg_bear_btm']       = fvg_bear_btm
    df['fvg_bear_mid']       = fvg_bear_mid
    df['fvg_mitigated_bull'] = fvg_mitigated_bull
    df['fvg_mitigated_bear'] = fvg_mitigated_bear

    return df


# ----------------------------------------------------------
# 3e. DERIVED FEATURES (fitur turunan SMC)
# ----------------------------------------------------------
def compute_derived(df):
    """
    Fitur turunan dari komponen SMC:
    - Jarak harga ke swing, OB, FVG
    - Posisi harga relatif terhadap struktur
    """
    c = df['Close']

    # Jarak ke swing
    df['dist_to_swing_high'] = df['last_swing_high'] - c
    df['dist_to_swing_low']  = c - df['last_swing_low']

    # Apakah harga di atas / bawah OB
    df['price_above_ob_bull'] = (c > df['ob_bull_avg']).astype(int)
    df['price_below_ob_bear'] = (c < df['ob_bear_avg']).astype(int)

    # Jarak ke midpoint OB
    df['dist_to_ob_bull_avg'] = c - df['ob_bull_avg']
    df['dist_to_ob_bear_avg'] = df['ob_bear_avg'] - c

    # Jarak ke FVG midpoint
    df['dist_to_fvg_bull_mid'] = c - df['fvg_bull_mid']
    df['dist_to_fvg_bear_mid'] = df['fvg_bear_mid'] - c

    # Apakah harga di dalam FVG
    df['in_bull_fvg'] = ((c > df['fvg_bull_btm']) & (c < df['fvg_bull_top'])).astype(int)
    df['in_bear_fvg'] = ((c > df['fvg_bear_btm']) & (c < df['fvg_bear_top'])).astype(int)

    # Momentum struktur
    df['bos_choch_signal'] = df['ms_bos'] + df['ms_choch']   # -1, 0, 1, atau 2

    return df


# ============================================================
# 4. APPLY SEMUA SMC
# ============================================================
print("[INFO] Menghitung SMC features...")
df = compute_swing(df, window=5)
df = compute_market_structure(df, window=5)
df = compute_order_block(df)
df = compute_fvg(df, fvg_thresh=0.0)
df = compute_derived(df)
print("[INFO] SMC features selesai.")


# ============================================================
# 5. HANDLE NaN & TARGET
# ============================================================
df.replace([np.inf, -np.inf], np.nan, inplace=True)
for c in df.columns:
    if df[c].dtype != 'object' and df[c].isnull().any():
        df[c] = df[c].ffill().bfill()

df['Target'] = df['Close'].shift(-1)
df.dropna(inplace=True)
print(f"[INFO] Data setelah clean: {len(df)} baris")


# ============================================================
# 6. DEFINISI FITUR PER MODEL
# ============================================================
# Model 1: OHLC saja
fitur_model1 = ['Open', 'High', 'Low', 'Close', 'Volume']

# Model 2: OHLC + Indikator Teknikal
fitur_model2 = fitur_model1 + [
    'EMA_20', 'EMA_50', 'RSI_14', atr_col, 'PSAR'
]

# Model 3: OHLC + Indikator + Semua SMC
fitur_model3 = fitur_model2 + [
    'swing_high', 'swing_low',
    'last_swing_high', 'last_swing_low',
    'ms_trend', 'ms_bos', 'ms_choch',
    'ob_bull_top', 'ob_bull_btm', 'ob_bull_avg', 'ob_bull_active',
    'ob_bear_top', 'ob_bear_btm', 'ob_bear_avg', 'ob_bear_active',
    'fvg_bull', 'fvg_bear',
    'fvg_bull_top', 'fvg_bull_btm', 'fvg_bull_mid',
    'fvg_bear_top', 'fvg_bear_btm', 'fvg_bear_mid',
    'fvg_mitigated_bull', 'fvg_mitigated_bear',
    'dist_to_swing_high', 'dist_to_swing_low',
    'price_above_ob_bull', 'price_below_ob_bear',
    'dist_to_ob_bull_avg', 'dist_to_ob_bear_avg',
    'dist_to_fvg_bull_mid', 'dist_to_fvg_bear_mid',
    'in_bull_fvg', 'in_bear_fvg',
    'bos_choch_signal'
]

# Verifikasi semua kolom ada
for nama, fitur in [("Model 1", fitur_model1), ("Model 2", fitur_model2), ("Model 3", fitur_model3)]:
    missing = [f for f in fitur if f not in df.columns]
    if missing:
        print(f"[WARNING] {nama} - kolom tidak ditemukan: {missing}")
    else:
        print(f"[INFO] {nama}: {len(fitur)} fitur ✓")


# ============================================================
# 7. SPLIT DATA
# ============================================================
train_size = int(len(df) * 0.8)
val_size   = int(len(df) * 0.1)
train_df   = df.iloc[:train_size].copy()
val_df     = df.iloc[train_size:train_size + val_size].copy()
test_df    = df.iloc[train_size + val_size:].copy()

print(f"[INFO] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")


# ============================================================
# 8. HELPER FUNCTIONS
# ============================================================
def create_dataset(X, y, window=60):
    Xs, ys = [], []
    for i in range(window, len(X)):
        Xs.append(X[i - window:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def build_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model


def evaluate_model(y_true, y_pred, current_price, label="Model"):
    n = min(len(y_true), len(y_pred), len(current_price))
    y_true        = y_true[:n]
    y_pred        = y_pred[:n]
    current_price = current_price[:n]

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    y_true_dir = (y_true > current_price).astype(int)
    y_pred_dir = (y_pred > current_price).astype(int)
    acc  = accuracy_score (y_true_dir, y_pred_dir)
    prec = precision_score(y_true_dir, y_pred_dir, zero_division=0)
    rec  = recall_score   (y_true_dir, y_pred_dir, zero_division=0)

    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  MAE          : {mae:.4f}")
    print(f"  RMSE         : {rmse:.4f}")
    print(f"  MAPE         : {mape:.4f}%")
    print(f"  Akurasi Arah : {acc:.4f}")
    print(f"  Precision    : {prec:.4f}")
    print(f"  Recall       : {rec:.4f}")

    return {
        'label': label, 'mae': mae, 'rmse': rmse,
        'mape': mape, 'acc': acc, 'prec': prec, 'rec': rec,
        'y_true': y_true, 'y_pred': y_pred
    }


# ============================================================
# 9. TRAINING & EVALUASI 3 SKENARIO
# ============================================================
window_size = 24
early_stop  = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
results     = []

skenario = [
    (fitur_model1, "Model 1: OHLC Saja"),
    (fitur_model2, "Model 2: OHLC + Indikator Teknikal"),
    (fitur_model3, "Model 3: OHLC + Indikator + SMC Lengkap"),
]

for fitur, label in skenario:
    print(f"\n[TRAINING] {label}...")

    # Validasi kolom
    missing = [f for f in fitur if f not in df.columns]
    if missing:
        print(f"  [ERROR] Kolom tidak ditemukan: {missing}")
        continue

    # Scaling
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaler_x.fit(train_df[fitur])
    scaler_y.fit(train_df[['Target']])

    train_x = scaler_x.transform(train_df[fitur])
    val_x   = scaler_x.transform(val_df  [fitur])
    test_x  = scaler_x.transform(test_df [fitur])
    train_y = scaler_y.transform(train_df[['Target']])
    val_y   = scaler_y.transform(val_df  [['Target']])
    test_y  = scaler_y.transform(test_df [['Target']])

    # Buat dataset dengan sliding window
    X_train, y_train = create_dataset(train_x, train_y, window_size)
    X_val,   y_val   = create_dataset(val_x,   val_y,   window_size)
    X_test,  y_test  = create_dataset(test_x,  test_y,  window_size)

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        print(f"  [SKIP] Data tidak cukup. Kurangi window_size atau tambah data.")
        continue

    # Build & train
    model = build_model((X_train.shape[1], X_train.shape[2]))
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )

    # Prediksi & inverse transform
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_true        = scaler_y.inverse_transform(y_test).flatten()
    y_pred        = scaler_y.inverse_transform(y_pred_scaled).flatten()

    # Current price: sejajarkan dengan index setelah window
    current_price = test_df['Close'].reset_index(drop=True).iloc[window_size:].values

    results.append(evaluate_model(y_true, y_pred, current_price, label))


# ============================================================
# 10. VISUALISASI
# ============================================================
if results:
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # Plot 1: Perbandingan prediksi vs aktual (100 data terakhir)
    ax1 = axes[0]
    ax1.plot(results[0]['y_true'][-100:], label='Harga Aktual', color='black', linewidth=2)
    colors = ['#2196F3', '#FF9800', '#4CAF50']
    for res, col in zip(results, colors):
        ax1.plot(res['y_pred'][-100:], label=res['label'], linestyle='--', color=col)
    ax1.legend(fontsize=8)
    ax1.set_title("Prediksi vs Harga Aktual XAUUSD (100 Data Terakhir)")
    ax1.set_xlabel("Periode")
    ax1.set_ylabel("Harga (USD)")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Bar chart perbandingan metrik
    ax2 = axes[1]
    labels   = [r['label'].replace("Model ", "M") for r in results]
    mae_vals = [r['mae']  for r in results]
    acc_vals = [r['acc']  for r in results]
    x = np.arange(len(labels))
    w = 0.35
    bar1 = ax2.bar(x - w/2, mae_vals, w, label='MAE',          color='#f44336', alpha=0.8)
    ax2b = ax2.twinx()
    bar2 = ax2b.bar(x + w/2, acc_vals, w, label='Akurasi Arah', color='#4CAF50', alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel('MAE (lebih rendah = lebih baik)', color='#f44336')
    ax2b.set_ylabel('Akurasi Arah (lebih tinggi = lebih baik)', color='#4CAF50')
    ax2.set_title("Perbandingan Metrik Antar Model")
    lines = [bar1, bar2]
    ax2.legend(lines, ['MAE', 'Akurasi Arah'], loc='upper left')

    plt.tight_layout()
    plt.savefig('hasil_perbandingan_model.png', dpi=150)
    plt.show()
    print("\n[INFO] Grafik disimpan ke 'hasil_perbandingan_model.png'")

    # ============================================================
    # 11. RINGKASAN TABEL
    # ============================================================
    print("\n\n" + "=" * 75)
    print("  RINGKASAN PERBANDINGAN MODEL")
    print("=" * 75)
    print(f"{'Model':<42} {'MAE':>8} {'RMSE':>8} {'MAPE%':>8} {'Acc Arah':>10} {'Precision':>10} {'Recall':>8}")
    print("-" * 96)
    for r in results:
        print(f"{r['label']:<42} {r['mae']:>8.4f} {r['rmse']:>8.4f} "
              f"{r['mape']:>8.4f} {r['acc']:>10.4f} {r['prec']:>10.4f} {r['rec']:>8.4f}")
    print("=" * 96)

    # Tentukan model terbaik berdasarkan akurasi arah
    best = max(results, key=lambda x: x['acc'])
    print(f"\n  ★ Model terbaik (akurasi arah): {best['label']}")
    print(f"    Akurasi Arah : {best['acc']:.4f}")
    print(f"    MAE          : {best['mae']:.4f}")
    print(f"    RMSE         : {best['rmse']:.4f}")

else:
    print("\n[WARNING] Tidak ada model yang berhasil dilatih.")
    print("          Periksa: ukuran data, window_size, dan kelengkapan kolom.")