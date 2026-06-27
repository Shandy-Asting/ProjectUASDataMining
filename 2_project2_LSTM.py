import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings, os
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, BatchNormalization,
                                     Bidirectional, Conv1D, MaxPooling1D,
                                     Input, Flatten)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

#AUTO LOAD DATASET
@st.cache_data(show_spinner="Memuat dataset otomatis...")
def auto_load_pollution():
    from pathlib import Path
    base = Path(__file__).resolve().parent
    
    if (base / "air_pollution_china.csv").exists():
        return pd.read_csv(base / "air_pollution_china.csv", low_memory=False)
    
    return None

if 'df_dl' not in st.session_state:
    df_auto = auto_load_pollution()
    if df_auto is not None:
        st.session_state['df_dl']        = df_auto
        st.session_state['df_source_dl'] = "auto (lokal)"

#Custom CSS Pembaruan Warna & Kotak Navigasi Terpisah
st.markdown("""
    <style>
    /* Mengubah font global dan background aplikasi */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #F8FAFC;
        color: #1E293B;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* STYLE: Kotak Terpisah untuk setiap Menu Navigasi di Sidebar */
    .nav-card {
        background-color: #FFFFFF;
        padding: 12px 16px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
        border: 1px solid #E2E8F0;
        margin-bottom: 10px;
        transition: all 0.2s ease-in-out;
    }
    .nav-card:hover {
        border-color: #4F46E5;
        box-shadow: 0 4px 6px rgba(79, 70, 229, 0.1);
    }
    
    /* Header Seksi Utama */
    .section-container {
        display: flex;
        align-items: center;
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 2px solid #E2E8F0;
    }
    .section-num {
        background-color: #4F46E5; /* Warna Utama: Indigo */
        color: white;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 14px;
        margin-right: 12px;
    }
    .section-title {
        font-size: 20px;
        font-weight: 700;
        color: #0F172A;
    }
    
    /* Info Card / Banner */
    .info-card {
        background-color: #FFFFFF;
        border-left: 4px solid #10B981; /* Warna Aksentuasi: Emerald Green */
        padding: 16px 20px;
        border-radius: 0 12px 12px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 24px;
        border-top: 1px solid #E2E8F0;
        border-right: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
    }
    
    /* Badge Kustom Multi-warna */
    .custom-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 10px;
    }
    .badge-indigo { background-color: #E0E7FF; color: #4338CA; }
    .badge-emerald { background-color: #D1FAE5; color: #065F46; }
    .badge-orange { background-color: #FFEDD5; color: #9A3412; }
    .badge-slate { background-color: #F1F5F9; color: #475569; }
    
    /* Menyembunyikan radio bawaan agar bisa menggunakan visual kustom */
    div[data-testid="stSidebar"] div.stRadio {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Parameter global visualisasi agar serasi dengan background putih
PLOT_BG = '#FFFFFF'
PLOT_TEXT = '#1E293B'
PLOT_BORDER = '#E2E8F0'
COLOR_INDIGO = '#4F46E5'
COLOR_EMERALD = '#10B981'
COLOR_ORANGE = '#F97316'
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_FILE = "air_pollution_china.csv"
DEFAULT_DATA_PATH = os.path.join(PROJECT_DIR, DEFAULT_DATA_FILE)

@st.cache_data(show_spinner=False)
def load_csv_data(source):
    return pd.read_csv(source)

def initialize_default_dataset():
    if 'df_dl' in st.session_state:
        return

    if os.path.exists(DEFAULT_DATA_PATH):
        st.session_state['df_dl'] = load_csv_data(DEFAULT_DATA_PATH)
        st.session_state['df_source_dl'] = DEFAULT_DATA_FILE
    else:
        st.session_state['df_source_dl'] = None

initialize_default_dataset()

#Helper Kustom
def section_header(num, title):
    st.markdown(f"""
    <div class="section-container">
        <span class="section-num">{num}</span>
        <span class="section-title">{title}</span>
    </div>
    """, unsafe_allow_html=True)

#Bagian Atas: Header Utama & Penjelasan Algoritma
st.title("📈 Project 2: Time Series Forecasting & Regression")
st.markdown(f"""
<span class="custom-badge badge-indigo">Deep Learning</span>
<span class="custom-badge badge-emerald">TensorFlow {tf.__version__}</span>
<span class="custom-badge badge-orange">LSTM & CNN-1D</span>
<span class="custom-badge badge-slate">Sequential Model</span>
""", unsafe_allow_html=True)

# Cek Ketersediaan Hardware GPU/CPU di Backend Banner
gpu_devices = tf.config.list_physical_devices('GPU')
gpu_status = f"Terdeteksi ({len(gpu_devices)} GPU)" if gpu_devices else "Tidak Aktif (Menggunakan CPU)"

# Penjelasan Arsitektur Algoritma Deep Learning yang Digunakan
st.markdown(f"""
<div class="info-card">
    <div style="font-weight:700; color:#0F172A; font-size:16px; margin-bottom:8px;">🤖 Arsitektur Deep Learning Sekuensial</div>
    <p style="color:#475569; font-size:14px; margin:0; line-height:1.6;">
        Sistem ini dirancang untuk memproses data deret waktu (<i>time series</i>) atau regresi menggunakan kombinasi arsitektur jaringan saraf tingkat lanjut (Deep Learning) melalui framework TensorFlow/Keras:
    </p>
    <ul style="color:#475569; font-size:14px; margin-top:6px; margin-bottom:6px; padding-left:20px; line-height:1.6;">
        <li><strong>Long Short-Term Memory (LSTM / Bidirectional):</strong> Lapisan khusus RNN yang mampu menangkap pola ketergantungan jangka panjang (<i>long-term dependencies</i>) dan tren sekuensial masa lalu tanpa mengalami kendala <i>vanishing gradient</i>.</li>
        <li><strong>Convolutional 1D (Conv1D & MaxPooling1D):</strong> Digunakan untuk ekstraksi fitur otomatis berskala lokal sepanjang sumbu waktu, mempercepat konvergensi model sebelum diteruskan ke lapisan rekuren.</li>
        <li><strong>Regularisasi (Dropout & Batch Normalization):</strong> Mencegah kondisi <i>overfitting</i> pada data latih serta menstabilkan proses pembaruan bobot sinapsis selama masa komputasi.</li>
    </ul>
    <div style="font-size: 12px; margin-top: 8px; color: #64748B; font-weight: 500;">
        ⚡ Akselerasi Hardware GPU: <span style="color:{'#10B981' if gpu_devices else '#EF4444'}; font-weight:700;">{gpu_status}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Navigasi Sidebar Kiri (Kotak Terpisah Berbasis Terstruktur) ─
with st.sidebar:
    st.markdown("<p style='font-weight:700; color:#0F172A; margin-bottom:15px; font-size:15px;'>📌 Alur Pemrosesan Jaringan</p>", unsafe_allow_html=True)
    
    menu_options = [
        "📂 Load Data", "🔍 EDA", "🩹 Missing Value", "📉 Outlier", 
        "🔤 Encoding", "🔄 Transformasi", "🎯 Split Sequence", "🤖 Modeling DL", 
        "📊 Evaluasi", "💾 Ekspor Model"
    ]
    
    st.markdown('<div style="margin-bottom: -15px;">', unsafe_allow_html=True)
    sub = st.radio("Pilih Tahapan:", menu_options, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

# SUB PAGE: LOAD DATA
if sub == "📂 Load Data":
    section_header("01", "Load Dataset Sejarah Waktu")
    
    st.markdown("""
    <div class="info-card" style="border-left-color: #4F46E5;">
        <div style="font-weight:700; color:#0F172A; font-size:15px; margin-bottom:6px;">📌 Pemuatan Berkas Kontinu</div>
        <p style="color:#475569; font-size:14px; margin:0; line-height:1.6;">
        Unggah berkas runtun waktu Anda (Format .csv) yang memiliki indeks waktu berurutan atau representasi fitur target regresi.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Pilih file CSV Time Series / Regresi", type=["csv"])
    
    if uploaded_file is not None:
        df = load_csv_data(uploaded_file)
        st.session_state['df_dl'] = df
        st.session_state['df_source_dl'] = uploaded_file.name
        st.success(f"✅ Dataset berhasil dimuat: {df.shape[0]:,} Baris | {df.shape[1]} Kolom")
        st.dataframe(df.head(10), use_container_width=True)
    elif 'df_dl' in st.session_state:
        df = st.session_state['df_dl']
        source_name = st.session_state.get('df_source_dl') or "dataset default"
        st.success(f"âœ… Dataset otomatis aktif dari {source_name}: {df.shape[0]:,} Baris | {df.shape[1]} Kolom")
        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.warning(f"Dataset default {DEFAULT_DATA_FILE} tidak ditemukan. Silakan unggah file CSV terlebih dahulu.")

# SUB PAGE: EDA
elif sub == "🔍 EDA":
    section_header("02", "Eksplorasi Karakteristik Tren (EDA)")
    
    if 'df_dl' not in st.session_state:
        st.warning("⚠️ Silakan muat dataset terlebih dahulu di tab Load Data.")
        st.stop()
        
    df = st.session_state['df_dl']
    
    st.markdown("### Ringkasan Struktur Matriks")
    st.dataframe(df.describe().T, use_container_width=True)
    
    # Visualisasi Tren Sederhana jika terdeteksi kolom numerik
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        st.markdown("### Visualisasi Fitur Numerik Kontinu")
        selected_col = st.selectbox("Pilih kolom untuk dianalisis polanya:", num_cols)
        
        fig, ax = plt.subplots(figsize=(12, 4))
        fig.patch.set_facecolor(PLOT_BG)
        ax.set_facecolor(PLOT_BG)
        ax.tick_params(colors=PLOT_TEXT)
        ax.spines[:].set_color(PLOT_BORDER)
        
        ax.plot(df[selected_col].values[:1000], color=COLOR_INDIGO, linewidth=1.5)
        ax.set_title(f"Tren Sinyal Data Sekuensial: {selected_col} (1000 Titik Pertama)", color=PLOT_TEXT, weight='bold')
        ax.grid(alpha=0.2, color=PLOT_BORDER)
        st.pyplot(fig)

# SUB PAGE: MISSING VALUE
elif sub == "🩹 Missing Value":
    section_header("03", "Restorasi Missing Value")
    
    if 'df_dl' not in st.session_state:
        st.warning("⚠️ Data belum diinisialisasi.")
        st.stop()
        
    df = st.session_state['df_dl'].copy()
    null_counts = df.isnull().sum()
    
    if null_counts.sum() == 0:
        st.success("✨ Sempurna! Tidak ditemukan adanya missing value di dalam matriks data.")
    else:
        st.write(null_counts[null_counts > 0])
        method_dl = st.selectbox("Metode Imputasi Runtun Waktu:", ["Linear Interpolation (Rekomendasi Time Series)", "Forward Fill", "Backward Fill", "Isi Nol"])
        
        if st.button("Terapkan Imputasi", type="primary"):
            if "Interpolation" in method_dl:
                df = df.interpolate(method='linear')
            elif "Forward" in method_dl:
                df = df.ffill()
            elif "Backward" in method_dl:
                df = df.bfill()
            else:
                df = df.fillna(0)
                
            st.session_state['df_dl'] = df
            st.success("✅ Seluruh nilai kosong berhasil dipulihkan secara kontinu!")

# SUB PAGE: OUTLIER
elif sub == "📉 Outlier":
    section_header("04", "Manajemen Hambatan Outlier")
    
    if 'df_dl' not in st.session_state:
        st.warning("⚠️ Data belum tersedia.")
        st.stop()
        
    st.info("💡 Catatan: Dalam arsitektur pemodelan runtun waktu / sinyal, nilai ekstrim sering kali merupakan anomali penting. Penskalaan MinMaxScaler pada tahap berikutnya umumnya direkomendasikan dibandingkan membuang data baris secara acak.")

# SUB PAGE: ENCODING
elif sub == "🔤 Encoding":
    section_header("05", "Transformasi Label Kategorikal")
    
    if 'df_dl' not in st.session_state:
        st.warning("⚠️ Data belum tersedia.")
        st.stop()
        
    df = st.session_state['df_dl'].copy()
    obj_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if not obj_cols:
        st.info("Seluruh kolom sudah berbentuk representasi numerik standar (Siap dimasukkan ke arsitektur Jaringan Saraf).")
    else:
        st.write("Kolom teks terdeteksi:", obj_cols)
        if st.button("Konversi Otomatis Menggunakan Label Encoder"):
            le = LabelEncoder()
            for col in obj_cols:
                df[col] = le.fit_transform(df[col].astype(str))
            st.session_state['df_dl'] = df
            st.success("✅ Matriks objek berhasil diubah ke dalam bentuk diskret integer.")

# SUB PAGE: TRANSFORMASI
elif sub == "🔄 Transformasi":
    section_header("06", "Normalisasi Skala MinMax")
    
    if 'df_dl' not in st.session_state:
        st.warning("⚠️ Data tidak ditemukan.")
        st.stop()
        
    df = st.session_state['df_dl'].copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    st.markdown("### Penskalaan Rentang Fitur (Crucial for Neural Networks)")
    selected_scale_cols = st.multiselect("Pilih kolom untuk di-scale ke rentang [0, 1]:", num_cols, default=num_cols)
    
    if st.button("Lakukan Rekayasa Skala", type="primary") and selected_scale_cols:
        scaler = MinMaxScaler()
        df[selected_scale_cols] = scaler.fit_transform(df[selected_scale_cols])
        st.session_state['df_dl'] = df
        st.success("✅ Fitur telah dinormalisasi menggunakan MinMaxScaler agar laju pembelajaran konvergen.")
        st.dataframe(df.head(5), use_container_width=True)

# SUB PAGE: SPLIT SEQUENCE
elif sub == "🎯 Split Sequence":
    section_header("07", "Pembentukan Data Sekuensial (Windowing)")
    
    if 'df_dl' not in st.session_state:
        st.warning("⚠️ Data tidak terdeteksi.")
        st.stop()
        
    df = st.session_state['df_dl']
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    target_col = st.selectbox("Pilih Fitur Target (Y):", num_cols)
    window_size = st.slider("Panjang Jendela Waktu Kebelakang (Lookback Window / Timesteps):", 3, 60, 10)
    
    if st.button("Konstruksi Menjadi Tensor 3-Dimensi"):
        # Logika pembentukan windowing untuk LSTM [Samples, Timesteps, Features]
        data_mat = df[num_cols].values
        target_idx = num_cols.index(target_col)
        
        X_list, y_list = [], []
        for i in range(len(data_mat) - window_size):
            X_list.append(data_mat[i:i+window_size, :])
            y_list.append(data_mat[i+window_size, target_idx])
            
        X_arr, y_arr = np.array(X_list), np.array(y_list)
        
        # Train, Test Split
        X_train, X_test, y_train, y_test = train_test_split(X_arr, y_arr, test_size=0.2, random_state=42, shuffle=False)
        
        st.session_state.update({
            'X_train_dl': X_train, 'X_test_dl': X_test,
            'y_train_dl': y_train, 'y_test_dl': y_test,
            'timesteps': window_size, 'features_dim': X_arr.shape[2]
        })
        
        col1, col2 = st.columns(2)
        col1.metric("Dimensi Matriks Latih (X_train)", str(X_train.shape), delta="Format 3D Tensor")
        col2.metric("Dimensi Matriks Uji (X_test)", str(X_test.shape))
        st.success("✅ Struktur data berhasil diubah ke dalam bentuk sekuensial 3D untuk komputasi Tensor!")

# SUB PAGE: MODELING DL
elif sub == "🤖 Modeling DL":
    section_header("08", "Konfigurasi Arsitektur Deep Learning")
    
    if 'X_train_dl' not in st.session_state:
        st.warning("⚠️ Lakukan transformasi bentuk tensor sekuensial di tab Split Sequence terlebih dahulu.")
        st.stop()
        
    timesteps = st.session_state['timesteps']
    features_dim = st.session_state['features_dim']
    
    X_train = st.session_state['X_train_dl']
    y_train = st.session_state['y_train_dl']
    
    arch_type = st.selectbox("Arsitektur Inti Jaringan:", ["Simple LSTM Network", "Stacked LSTM with Dropout", "CNN-LSTM Hybrid"])
    epochs_num = st.slider("Jumlah Iterasi Pelatihan (Epochs):", 5, 100, 20)
    batch_sz = st.select_slider("Ukuran Batch Kontinu:", options=[16, 32, 64, 128], value=32)
    
    if st.button("Mulai Proses Pelatihan Jaringan Saraf", type="primary"):
        with st.spinner("Memproses propagasi maju dan mundur..."):
            
            # Membangun Model Sekuensial Berdasarkan Pilihan
            model = Sequential()
            model.add(Input(shape=(timesteps, features_dim)))
            
            if arch_type == "Simple LSTM Network":
                model.add(LSTM(64, activation='tanh', return_sequences=False))
                model.add(Dense(1))
            elif arch_type == "Stacked LSTM with Dropout":
                model.add(LSTM(64, activation='tanh', return_sequences=True))
                model.add(Dropout(0.2))
                model.add(LSTM(32, activation='tanh', return_sequences=False))
                model.add(Dropout(0.2))
                model.add(Dense(1))
            else: # CNN-LSTM Hybrid
                model.add(Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))
                model.add(MaxPooling1D(pool_size=2))
                model.add(LSTM(64, activation='tanh', return_sequences=False))
                model.add(Dense(1))
                
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
            
            # Callback Otomatisasi
            history = model.fit(
                X_train, y_train,
                epochs=epochs_num,
                batch_size=batch_sz,
                validation_split=0.15,
                verbose=0
            )
            
            st.session_state['dl_model_instance'] = model
            st.session_state['dl_history'] = history.history
            
        st.success("🎉 Model Jaringan Saraf berhasil dikonvergensi dan dilatih secara optimal!")
        
        # Plot Loss Pelatihan menggunakan warna Emerald baru
        fig, ax = plt.subplots(figsize=(10, 3.5))
        fig.patch.set_facecolor(PLOT_BG)
        ax.set_facecolor(PLOT_BG)
        ax.tick_params(colors=PLOT_TEXT)
        ax.spines[:].set_color(PLOT_BORDER)
        
        ax.plot(history.history['loss'], label='Loss Latih', color=COLOR_INDIGO, linewidth=2)
        if 'val_loss' in history.history:
            ax.plot(history.history['val_loss'], label='Loss Validasi', color=COLOR_EMERALD, linewidth=2)
            
        ax.set_title("Kurva Pembelajaran (Loss Convergence)", color=PLOT_TEXT, weight='bold')
        ax.set_ylabel("Mean Squared Error (MSE)", color=PLOT_TEXT)
        ax.legend(labelcolor=PLOT_TEXT)
        ax.grid(alpha=0.2, color=PLOT_BORDER)
        st.pyplot(fig)

# SUB PAGE: EVALUASI
elif sub == "📊 Evaluasi":
    section_header("09", "Metrik Validasi & Komparasi Nilai")
    
    if 'dl_model_instance' not in st.session_state:
        st.warning("⚠️ Belum ada model aktif yang dilatih.")
        st.stop()
        
    model = st.session_state['dl_model_instance']
    X_test = st.session_state['X_test_dl']
    y_test = st.session_state['y_test_dl']
    
    predictions = model.predict(X_test).flatten()
    
    mae_v = mean_absolute_error(y_test, predictions)
    mse_v = mean_squared_error(y_test, predictions)
    r2_v = r2_score(y_test, predictions)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Absolute Error (MAE)", f"{mae_v:.5f}")
    col2.metric("Mean Squared Error (MSE)", f"{mse_v:.5f}")
    col3.metric("R-Squared Score (R²)", f"{r2_v:.4f}")
    
    # Plot Hasil Komparasi Prediksi Aktual vs Estimasi
    st.markdown("### Perbandingan Grafik Kontinu Aktual vs Estimasi Model")
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor(PLOT_BG)
    ax.set_facecolor(PLOT_BG)
    ax.tick_params(colors=PLOT_TEXT)
    ax.spines[:].set_color(PLOT_BORDER)
    
    ax.plot(y_test[:200], label='Nilai Aktual Realitas', color=COLOR_INDIGO, alpha=0.8, linewidth=1.5)
    ax.plot(predictions[:200], label='Estimasi Jaringan Saraf', color=COLOR_ORANGE, linestyle='--', linewidth=1.5)
    ax.set_title("Grafik Komparasi Sinyal (200 Sampel Pertama Test Set)", color=PLOT_TEXT, weight='bold')
    ax.legend(labelcolor=PLOT_TEXT)
    ax.grid(alpha=0.15, color=PLOT_BORDER)
    st.pyplot(fig)

# SUB PAGE: EKSPOR MODEL
elif sub == "💾 Ekspor Model":
    section_header("10", "Penyimpanan Pembobotan Arsitektur")
    
    if 'dl_model_instance' not in st.session_state:
        st.info("Belum ada arsitektur model komplit yang tersimpan di memori jangka pendek.")
    else:
        st.success("Model Jaringan Saraf saat ini siap diekspor ke dalam bentuk pembobotan terkompresi H5 / Keras Native.")
        # Catatan: Ekspor biner model DL kompleks di Streamlit umumnya diarahkan ke direktori penyimpanan lokal server penyedia.
        st.info("Gunakan instruksi internal `model.save('DL_forecasting_model.h5')` di dalam server lokal Anda untuk mengamankan struktur sinapsis.")
