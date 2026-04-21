import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
print("1. Memuat data dan menghitung indikator teknikal...")
df = pd.read_csv('XAUUSD_H1_1Tahun.csv', sep='\t')

df['Date'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'])
df.set_index('Date', inplace=True)
df.rename(columns={'<OPEN>': 'Open', '<HIGH>': 'High', '<LOW>': 'Low', '<CLOSE>': 'Close', '<TICKVOL>': 'Volume'}, inplace=True)

df.ta.ema(length=20, append=True)
df.ta.ema(length=50, append=True)
df.ta.rsi(length=14, append=True)
df.ta.atr(length=14, append=True)

#
psar = df.ta.psar()
df['PSAR'] = psar.iloc[:, 0].fillna(psar.iloc[:, 1]) 

# Representasi Fitur SMC
df['Order_Block_Low'] = df['Low'].rolling(window=24).min() 
df['Jarak_Order_Block'] = df['Close'] - df['Order_Block_Low']

#PEMBUATAN LABEL REGRESI
df['Target'] = df['Close'].shift(-1)
kolom_dipakai = ['Open', 'High', 'Low', 'Close', 'EMA_20', 'EMA_50', 'RSI_14', 'ATRr_14', 'PSAR', 'Jarak_Order_Block']
df.dropna(subset=kolom_dipakai, inplace=True)

# TAHAP 4: DEFINISI SKENARIO
skenario_pengujian = {
    "Model 1 (OHLC)": ['Open', 'High', 'Low', 'Close'],
    "Model 2 (OHLC + Indikator)": ['Open', 'High', 'Low', 'Close', 'EMA_20', 'EMA_50', 'RSI_14', 'ATRr_14', 'PSAR'],
    "Model 3 (OHLC + Indikator + SMC)": ['Open', 'High', 'Low', 'Close', 'EMA_20', 'EMA_50', 'RSI_14', 'ATRr_14', 'PSAR', 'Jarak_Order_Block']
}

hasil_evaluasi = {}
window_size = 60 

for nama_model, fitur_pilihan in skenario_pengujian.items():
    print(f"\nRUNNING: {nama_model}")
    
    # 1. Normalisasi
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    
    data_x = scaler_x.fit_transform(df[fitur_pilihan])
    data_y = scaler_y.fit_transform(df[['Close']]) # KOREKSI: Target langsung menggunakan kolom 'Close' murni

    # 2. Sliding Window (Otomatis mengambil t+1 sebagai y)
    X, y = [], []
    for i in range(window_size, len(data_x)):
        X.append(data_x[i - window_size : i])
        y.append(data_y[i]) # data_y[i] adalah Close di waktu t+1 (tepat setelah window)
    X, y = np.array(X), np.array(y)

    # 3. Split Data 80:10:10
    train_size = int(len(X) * 0.8)
    val_size = int(len(X) * 0.1)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size : train_size+val_size], y[train_size : train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

    # 4. Arsitektur Model Regresi
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.1), # KOREKSI: Diturunkan jadi 0.1 agar model tidak terlalu banyak "lupa"
        LSTM(50, return_sequences=False),
        Dropout(0.1),
        Dense(1, activation='linear') 
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    # KOREKSI: Fitur EarlyStopping agar model bisa belajar lama (epochs besar) 
    # tapi otomatis berhenti jika sudah mencapai titik terbaik (mencegah overfitting).
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 5. Training
    print("Sedang melatih model... (Tunggu sebentar)")
    # KOREKSI: Epochs dinaikkan ke 100 agar model punya cukup waktu mengurangi 'gap' harga
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, callbacks=[early_stop], verbose=0)

    # 6. Prediksi & Inverse Transform
    y_pred_scaled = model.predict(X_test)
    y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))

    # 7. Evaluasi Metrik
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    hasil_evaluasi[nama_model] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'Pred': y_pred, 'True': y_true}
    print(f"Hasil {nama_model} -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
    # TAHAP 5: VISUALISASI PERBANDINGAN (TAMBAHKAN INI DI PALING BAWAH)
print("\n" + "="*50)
print("MEMBUAT VISUALISASI...")
print("="*50)

# Kita ambil 100 sampel terakhir agar grafik tidak terlalu padat dan terlihat jelas gap-nya
n_view = 100 

plt.figure(figsize=(15, 8))

# Plot Harga Aktual
# Kita ambil dari model manapun karena harga aktualnya sama
plt.plot(hasil_evaluasi["Model 1 (OHLC)"]['True'][-n_view:], 
         label='Harga Aktual (True)', color='black', linewidth=2, linestyle='--')

# Plot Prediksi Model 1
plt.plot(hasil_evaluasi["Model 1 (OHLC)"]['Pred'][-n_view:], 
         label=f"Model 1 (MAPE: {hasil_evaluasi['Model 1 (OHLC)']['MAPE']:.2f}%)", alpha=0.7)

# Plot Prediksi Model 2
plt.plot(hasil_evaluasi["Model 2 (OHLC + Indikator)"]['Pred'][-n_view:], 
         label=f"Model 2 (MAPE: {hasil_evaluasi['Model 2 (OHLC + Indikator)']['MAPE']:.2f}%)", alpha=0.7)

# Plot Prediksi Model 3
plt.plot(hasil_evaluasi["Model 3 (OHLC + Indikator + SMC)"]['Pred'][-n_view:], 
         label=f"Model 3 (MAPE: {hasil_evaluasi['Model 3 (OHLC + Indikator + SMC)']['MAPE']:.2f}%)", 
         color='blue', linewidth=2)

plt.title('Perbandingan Prediksi Harga XAUUSD (100 Jam Terakhir)', fontsize=14)
plt.xlabel('Data ke- (Waktu)')
plt.ylabel('Harga (USD)')
plt.legend()
plt.grid(True, alpha=0.3)

# Perintah krusial agar window chart muncul
plt.show()