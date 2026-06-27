# pages/1_project1_flight.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import zipfile
from pathlib import Path
import lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve,
                             classification_report, confusion_matrix)
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Flight Delay Classification",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Mengubah font global dan background aplikasi */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #F8FAFC;
        color: #1E293B;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* STYLE BARU: Kotak Terpisah untuk setiap Menu Navigasi di Sidebar */
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
        background-color: #0284C7; /* Warna 1: Sky Blue */
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
        border-left: 4px solid #4F46E5; /* Warna 3: Indigo */
        padding: 16px 20px;
        border-radius: 0 12px 12px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 24px;
        border-top: 1px solid #E2E8F0;
        border-right: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
    }
    
    /* Badge Klasifikasi Multi-warna */
    .custom-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 10px;
    }
    .badge-blue { background-color: #E0F2FE; color: #0369A1; }
    .badge-indigo { background-color: #E0E7FF; color: #4338CA; }
    .badge-emerald { background-color: #D1FAE5; color: #065F46; } /* Warna 4: Emerald */
    .badge-amber { background-color: #FEF3C7; color: #B45309; }
    
    /* Sembunyikan radio bawaan agar bisa menggunakan visual kustom */
    div[data-testid="stSidebar"] div.stRadio {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
    }
    </style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent
LOCAL_FLIGHTS_CSV = BASE_DIR / "flights.csv"
LOCAL_ARCHIVE_ZIP = BASE_DIR / "archive (3).zip"


def section_header(num, title):
    st.markdown(f"""
    <div class="section-container">
        <span class="section-num">{num}</span>
        <span class="section-title">{title}</span>
    </div>
    """, unsafe_allow_html=True)


PLOT_BG = '#FFFFFF'
PLOT_TEXT = '#1E293B'
PLOT_BORDER = '#E2E8F0'
COLOR_PRIMARY = '#0284C7'    
COLOR_SECONDARY = '#EF4444'  
COLOR_INDIGO = '#4F46E5'     
COLOR_EMERALD = '#10B981'    


st.title("✈️ Project 1: Klasifikasi Flight Delay")
st.markdown("""
<span class="custom-badge badge-blue">Data Tabel</span>
<span class="custom-badge badge-indigo">LightGBM & Tree</span>
<span class="custom-badge badge-emerald">Optimized UI</span>
<span class="custom-badge badge-amber">2015 Dataset</span>
""", unsafe_allow_html=True)


st.markdown("""
<div class="info-card" style="border-left-color: #4F46E5;">
    <div style="font-weight:700; color:#0F172A; font-size:16px; margin-bottom:8px;">🤖 Algoritma Pemodelan yang Digunakan</div>
    <p style="color:#475569; font-size:14px; margin:0; line-height:1.6;">
        Sistem ini mengimplementasikan dua jenis pendekatan algoritma berbasis pohon keputusan untuk melakukan klasifikasi status keterlambatan penerbangan:
    </p>
    <ul style="color:#475569; font-size:14px; margin-top:6px; margin-bottom:0; padding-left:20px; line-height:1.6;">
        <li><strong style="color:#4F46E5;">LightGBM (Algoritma Utama):</strong> Framework <i>gradient boosting</i> yang menggunakan struktur pertumbuhan pohon secara vertikal (<i>leaf-wise</i>). Memiliki kecepatan pelatihan yang sangat tinggi, efisiensi memori yang optimal, serta akurasi yang unggul ketika menangani dataset berskala besar seperti catatan penerbangan domestik ini.</li>
        <li><strong style="color:#0284C7;">Decision Tree (Algoritma Pembanding):</strong> Model pohon keputusan standar yang membagi data berdasarkan nilai fitur dengan metrik impuritas Gini atau Entropy. Digunakan sebagai baseline struktural untuk membandingkan peningkatan performa yang dihasilkan oleh model ensemble boosting.</li>
    </ul>
</div>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("<p style='font-weight:700; color:#0F172A; margin-bottom:15px; font-size:15px;'>📌 Alur Pemrosesan Data</p>", unsafe_allow_html=True)
    
    menu_options = [
        "📂 Load Data", "🔍 EDA", "🩹 Missing Value", "📉 Outlier", 
        "🔤 Encoding", "🔄 Transformasi", "🎯 Split", "🤖 Modeling", 
        "📊 Evaluasi", "💾 Ekspor"
    ]
    
    # Membuat visual kotak terpisah untuk setiap pilihan radio button
    st.markdown('<div style="margin-bottom: -15px;">', unsafe_allow_html=True)
    sub = st.radio("Pilih Tahapan:", menu_options, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

# SUB PAGE: LOAD DATA
if sub == "📂 Load Data":
    section_header("01", "Load Dataset")

    st.markdown("""
    <div class="info-card" style="border-left-color: #0284C7;">
        <div style="font-weight:700; color:#0F172A; font-size:15px; margin-bottom:6px;">📌 Sumber Data: 2015 Flight Delays and Cancellations</div>
        <p style="color:#475569; font-size:14px; margin:0; line-height:1.6;">
        Silakan unggah berkas <strong>flights.csv</strong> dari Kaggle atau gunakan berkas lokal. Dataset mencakup informasi operasional maskapai, rute, waktu, dan rincian keterlambatan domestik.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        sample_size = st.slider(
            "Jumlah baris yang dimuat (sampling)",
            min_value=50_000, max_value=1_000_000,
            value=300_000, step=50_000,
            format="%d baris"
        )
    with col2:
        delay_thresh = st.slider(
            "Threshold delay (menit)",
            min_value=5, max_value=60, value=15,
            help="Penerbangan dianggap delay jika ARRIVAL_DELAY > threshold ini"
        )

    uploaded = st.file_uploader("Upload flights.csv (opsional)", type=['csv'])

    def get_local_flights_source():
        if LOCAL_FLIGHTS_CSV.exists():
            return LOCAL_FLIGHTS_CSV
        if LOCAL_ARCHIVE_ZIP.exists():
            return LOCAL_ARCHIVE_ZIP
        return None

    @st.cache_data(show_spinner="Memuat dataset...")
    def load_flights_from_path(path_str, n):
        path = Path(path_str)
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as zf:
                with zf.open("flights.csv") as file:
                    return pd.read_csv(file, nrows=n, low_memory=False)
        return pd.read_csv(path, nrows=n, low_memory=False)

    @st.cache_data(show_spinner="Memuat dataset upload...")
    def load_flights_from_upload(file, n, seed=42):
        df_full = pd.read_csv(file, low_memory=False)
        if len(df_full) > n:
            return df_full.sample(n=n, random_state=seed).reset_index(drop=True)
        return df_full.reset_index(drop=True)

    local_source = get_local_flights_source()
    if local_source:
        st.caption(f"File lokal terdeteksi: {local_source.name}")
    else:
        st.caption("File lokal flights.csv / archive (3).zip belum ditemukan.")

    should_load = uploaded is not None or local_source is not None
    reload_clicked = st.button("Load / reload dataset", type="primary", disabled=not should_load)
    
    if should_load and ('df' not in st.session_state or reload_clicked):
        if uploaded is not None:
            df = load_flights_from_upload(uploaded, sample_size)
            source_label = uploaded.name
        else:
            df = load_flights_from_path(str(local_source), sample_size)
            source_label = local_source.name

        if 'ARRIVAL_DELAY' not in df.columns:
            st.error("Kolom ARRIVAL_DELAY tidak ditemukan. Pastikan file yang dimuat adalah flights.csv.")
            st.stop()

        df['IS_DELAYED'] = (df['ARRIVAL_DELAY'] > delay_thresh).astype(int)
        st.session_state['df']           = df
        st.session_state['delay_thresh'] = delay_thresh
        st.session_state['data_source']  = source_label

    if 'df' in st.session_state:
        df = st.session_state['df']
        st.success(f"Dataset berhasil dimuat dari {st.session_state.get('data_source', 'file')}: {df.shape[0]:,} baris, {df.shape[1]} kolom")

        # Menggunakan warna Emerald baru untuk indikator angka utama
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Baris",   f"{df.shape[0]:,}")
        col2.metric("Total Kolom",   f"{df.shape[1]}")
        col3.metric(f"Delay (>{delay_thresh} mnt)", f"{df['IS_DELAYED'].sum():,}")
        col4.metric("Memori Objek",  f"{df.memory_usage(deep=True).sum()/1e6:.1f} MB")

        st.dataframe(df.head(8), use_container_width=True)
    elif not should_load:
        st.info("Letakkan flights.csv atau archive (3).zip di folder project, atau upload flights.csv untuk memulai.")

# SUB PAGE: EDA
elif sub == "🔍 EDA":
    section_header("02", "Eksplorasi Data (EDA)")

    if 'df' not in st.session_state:
        st.warning("⚠️ Load dataset terlebih dahulu di tab Load Data.")
        st.stop()

    df = st.session_state['df']

    tab1, tab2, tab3, tab4 = st.tabs([
        "Info Kolom", "Statistik", "Distribusi Delay", "Pola Waktu"
    ])

    with tab1:
        dtype_df = pd.DataFrame({
            "Kolom":     df.dtypes.index,
            "Tipe":      df.dtypes.values.astype(str),
            "Non-Null":  df.notnull().sum().values,
            "% Missing": (df.isnull().mean()*100).round(2).values,
            "Unique":    df.nunique().values
        })
        st.dataframe(dtype_df, use_container_width=True)

    with tab2:
        st.dataframe(df.describe().T, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(5, 5))
            fig.patch.set_facecolor(PLOT_BG)
            ax.set_facecolor(PLOT_BG)
            counts = df['IS_DELAYED'].value_counts()
            ax.pie(counts, labels=['Tidak Delay','Delay'],
                   autopct='%1.1f%%',
                   colors=[COLOR_PRIMARY, COLOR_SECONDARY], startangle=90,
                   textprops={'color': PLOT_TEXT})
            ax.set_title("Proporsi Target", color=PLOT_TEXT, weight='bold')
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(5, 5))
            fig.patch.set_facecolor(PLOT_BG)
            ax.set_facecolor(PLOT_BG)
            ax.tick_params(colors=PLOT_TEXT)
            ax.spines[:].set_color(PLOT_BORDER)
            df['ARRIVAL_DELAY'].clip(-60, 300).hist(
                bins=80, ax=ax, color=COLOR_PRIMARY, edgecolor='none')
            ax.axvline(st.session_state.get('delay_thresh', 15),
                       color=COLOR_SECONDARY, linestyle='--', linewidth=1.5,
                       label='Threshold')
            ax.set_title("Distribusi Arrival Delay", color=PLOT_TEXT, weight='bold')
            ax.legend(labelcolor=PLOT_TEXT)
            st.pyplot(fig)

        st.subheader("Delay Rate per Maskapai")
        fig, ax = plt.subplots(figsize=(12, 4))
        fig.patch.set_facecolor(PLOT_BG)
        ax.set_facecolor(PLOT_BG)
        ax.tick_params(colors=PLOT_TEXT)
        ax.spines[:].set_color(PLOT_BORDER)
        delay_by = df.groupby('AIRLINE')['IS_DELAYED'].mean().sort_values(ascending=False)
        delay_by.plot(kind='bar', ax=ax, color=COLOR_SECONDARY, edgecolor='none')
        ax.set_title("Delay Rate per Maskapai", color=PLOT_TEXT, weight='bold')
        ax.set_ylabel("Proporsi", color=PLOT_TEXT)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    with tab4:
        col_viz = st.selectbox("Pilih dimensi waktu", ['MONTH','DAY_OF_WEEK'])

        fig, ax = plt.subplots(figsize=(12, 4))
        fig.patch.set_facecolor(PLOT_BG)
        ax.set_facecolor(PLOT_BG)
        ax.tick_params(colors=PLOT_TEXT)
        ax.spines[:].set_color(PLOT_BORDER)
        df.groupby(col_viz)['IS_DELAYED'].mean().plot(
            ax=ax, marker='o', color=COLOR_INDIGO, linewidth=2)
        ax.set_title(f"Delay Rate per {col_viz}", color=PLOT_TEXT, weight='bold')
        ax.set_ylabel("Proporsi Delay", color=PLOT_TEXT)
        ax.grid(alpha=0.2, color=PLOT_BORDER)
        st.pyplot(fig)

# SUB PAGE: MISSING VALUE
elif sub == "🩹 Missing Value":
    section_header("03", "Handling Missing Value")

    if 'df' not in st.session_state:
        st.warning("⚠️ Load dataset dulu.")
        st.stop()

    df = st.session_state['df'].copy()

    missing     = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df  = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_pct
    }).sort_values('Missing %', ascending=False)
    missing_df  = missing_df[missing_df['Missing Count'] > 0]

    st.markdown(f"**Total kolom dengan missing value: {len(missing_df)}**")
    st.dataframe(missing_df, use_container_width=True)

    if len(missing_df) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor(PLOT_BG)
        ax.set_facecolor(PLOT_BG)
        ax.tick_params(colors=PLOT_TEXT)
        ax.spines[:].set_color(PLOT_BORDER)
        missing_pct[missing_pct > 0].plot(
            kind='barh', ax=ax, color=COLOR_SECONDARY, edgecolor='none')
        ax.set_title("Kolom dengan Missing Value (%)", color=PLOT_TEXT, weight='bold')
        plt.tight_layout()
        st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        threshold_drop = st.slider("Drop kolom jika missing > (%)", 0, 100, 80)
    with col2:
        num_fill = st.selectbox("Isi missing numerik dengan", ["median","mean","0"])

    if st.button("🚀 Terapkan Penanganan Missing Value"):
        drop_cols = missing_pct[missing_pct > threshold_drop].index.tolist()
        df.drop(columns=drop_cols, inplace=True)

        delay_cause = ['AIR_SYSTEM_DELAY','SECURITY_DELAY','AIRLINE_DELAY',
                       'LATE_AIRCRAFT_DELAY','WEATHER_DELAY']
        for c in delay_cause:
            if c in df.columns and df[c].isnull().any():
                df[c] = df[c].fillna(0)

        if 'CANCELLATION_REASON' in df.columns:
            df['CANCELLATION_REASON'] = df['CANCELLATION_REASON'].fillna('Not Cancelled')

        num_cols = df.select_dtypes(include=np.number).columns
        for c in num_cols:
            if df[c].isnull().any():
                val = (df[c].median() if num_fill == "median"
                       else df[c].mean() if num_fill == "mean" else 0)
                df[c] = df[c].fillna(val)

        df.dropna(inplace=True)
        st.session_state['df'] = df
        st.success(f"✅ Selesai! Sisa missing: {df.isnull().sum().sum()} | Shape: {df.shape}")


# SUB PAGE: OUTLIER
elif sub == "📉 Outlier":
    section_header("04", "Handling Outlier")

    if 'df' not in st.session_state:
        st.warning("⚠️ Load dataset dulu.")
        st.stop()

    df = st.session_state['df'].copy()

    outlier_cols = [c for c in ['DEPARTURE_DELAY','ARRIVAL_DELAY','TAXI_OUT',
                                 'TAXI_IN','AIR_TIME','DISTANCE','ELAPSED_TIME']
                    if c in df.columns]

    col_sel = st.selectbox("Pilih kolom untuk inspeksi", outlier_cols)

    def iqr_stats(series):
        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
        n_out = ((series < lower) | (series > upper)).sum()
        return lower, upper, n_out

    lower, upper, n_out = iqr_stats(df[col_sel])

    col1, col2, col3 = st.columns(3)
    col1.metric("Batas Bawah IQR", f"{lower:.1f}")
    col2.metric("Batas Atas IQR",  f"{upper:.1f}")
    col3.metric("Jumlah Outlier",  f"{n_out:,}")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor(PLOT_BG)
        ax.set_facecolor(PLOT_BG)
        ax.tick_params(colors=PLOT_TEXT)
        ax.spines[:].set_color(PLOT_BORDER)
        bp = ax.boxplot(df[col_sel].dropna(), patch_artist=True,
                   boxprops=dict(facecolor='#4F46E522', color=COLOR_INDIGO), # Kombinasi Indigo
                   medianprops=dict(color=COLOR_SECONDARY))
        ax.set_title(f"Boxplot: {col_sel}", color=PLOT_TEXT, weight='bold')
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor(PLOT_BG)
        ax.set_facecolor(PLOT_BG)
        ax.tick_params(colors=PLOT_TEXT)
        ax.spines[:].set_color(PLOT_BORDER)
        df[col_sel].clip(
            df[col_sel].quantile(0.01),
            df[col_sel].quantile(0.99)
        ).hist(bins=60, ax=ax, color=COLOR_INDIGO, edgecolor='none') # Kombinasi Indigo
        ax.set_title(f"Histogram: {col_sel}", color=PLOT_TEXT, weight='bold')
        st.pyplot(fig)

    method = st.selectbox("Metode penanganan", ["Winsorizing (IQR Clip)", "Drop baris outlier"])
    cols_handle = st.multiselect("Kolom yang ditangani", outlier_cols, default=outlier_cols[:4])

    if st.button("🚀 Terapkan Handling Outlier"):
        for col in cols_handle:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
            if method == "Winsorizing (IQR Clip)":
                df[col] = df[col].clip(lo, hi)
            else:
                df = df[(df[col] >= lo) & (df[col] <= hi)]

        st.session_state['df'] = df
        st.success(f"✅ Outlier ditangani! Shape sekarang: {df.shape}")

# SUB PAGE: ENCODING
elif sub == "🔤 Encoding":
    section_header("05", "Encoding Fitur Kategorikal")

    if 'df' not in st.session_state:
        st.warning("⚠️ Load dataset dulu.")
        st.stop()

    df = st.session_state['df'].copy()

    cat_cols = df.select_dtypes(include='object').columns.tolist()
    st.markdown(f"Kolom kategorikal terdeteksi: **{len(cat_cols)}** kolom")
    st.write(cat_cols)

    encoding_method = st.radio("Metode encoding",
        ["Label Encoding", "Frequency Encoding", "Gabungan (Auto)"],
        help="Gabungan: Label untuk kardinalitas rendah, Frequency untuk tinggi (>20 unik)")

    cols_to_encode = st.multiselect("Pilih kolom untuk di-encode", cat_cols, default=cat_cols)

    if st.button("🚀 Terapkan Encoding"):
        le = LabelEncoder()
        enc_report = []

        for col in cols_to_encode:
            n_unique = df[col].nunique()

            if encoding_method == "Label Encoding":
                df[col] = le.fit_transform(df[col].astype(str))
                enc_report.append({'Kolom': col, 'Metode': 'Label', 'Kategori': n_unique})

            elif encoding_method == "Frequency Encoding":
                freq = df[col].value_counts(normalize=True).to_dict()
                df[f'{col}_FREQ'] = df[col].map(freq)
                df.drop(columns=[col], inplace=True)
                enc_report.append({'Kolom': col, 'Metode': 'Frequency', 'Kategori': n_unique})

            else:  # Gabungan Auto
                if n_unique <= 20:
                    df[col] = le.fit_transform(df[col].astype(str))
                    enc_report.append({'Kolom': col, 'Metode': 'Label (auto)', 'Kategori': n_unique})
                else:
                    freq = df[col].value_counts(normalize=True).to_dict()
                    df[f'{col}_FREQ'] = df[col].map(freq)
                    df.drop(columns=[col], inplace=True)
                    enc_report.append({'Kolom': col, 'Metode': 'Frequency (auto)', 'Kategori': n_unique})

        st.session_state['df'] = df
        st.success(f"✅ Encoding selesai! Shape: {df.shape}")
        st.dataframe(pd.DataFrame(enc_report), use_container_width=True, hide_index=True)

# SUB PAGE: TRANSFORMASI
elif sub == "🔄 Transformasi":
    section_header("06", "Transformasi & Feature Engineering")

    if 'df' not in st.session_state:
        st.warning("⚠️ Load dataset dulu.")
        st.stop()

    df = st.session_state['df'].copy()

    st.subheader("6a. Fitur Waktu & Domain")
    if st.button("⏱️ Buat fitur waktu & domain"):
        if 'DAY_OF_WEEK' in df.columns:
            df['IS_WEEKEND']      = df['DAY_OF_WEEK'].isin([6,7]).astype(int)
        if 'SCHEDULED_DEPARTURE' in df.columns:
            df['DEP_HOUR']        = (df['SCHEDULED_DEPARTURE'] // 100).astype('Int64')
            df['IS_RUSH_HOUR']    = df['DEP_HOUR'].isin([7,8,9,16,17,18,19]).astype(int)
            df['IS_NIGHT_FLIGHT'] = df['DEP_HOUR'].isin([22,23,0,1,2,3,4]).astype(int)
            df['DEP_HOUR_SIN']    = np.sin(2*np.pi*df['DEP_HOUR']/24)
            df['DEP_HOUR_COS']    = np.cos(2*np.pi*df['DEP_HOUR']/24)
        if 'MONTH' in df.columns:
            df['IS_PEAK_MONTH']   = df['MONTH'].isin([6,7,8,11,12]).astype(int)
            df['MONTH_SIN']       = np.sin(2*np.pi*df['MONTH']/12)
            df['MONTH_COS']       = np.cos(2*np.pi*df['MONTH']/12)

        st.session_state['df'] = df
        st.success("✅ Fitur waktu berhasil dikonstruksi!")

    st.subheader("6b. Fitur Operasional")
    if st.button("✈️ Buat fitur operasional"):
        if all(c in df.columns for c in ['TAXI_OUT','TAXI_IN']):
            df['TOTAL_TAXI_TIME'] = df['TAXI_OUT'] + df['TAXI_IN']
        if all(c in df.columns for c in ['ELAPSED_TIME','SCHEDULED_TIME']):
            df['EFFICIENCY_RATIO'] = (df['ELAPSED_TIME'] / (df['SCHEDULED_TIME'] + 1)).round(4)
        delay_src = ['AIR_SYSTEM_DELAY','SECURITY_DELAY','AIRLINE_DELAY',
                     'LATE_AIRCRAFT_DELAY','WEATHER_DELAY']
        avail = [c for c in delay_src if c in df.columns]
        if avail:
            df['TOTAL_CAUSE_DELAY'] = df[avail].sum(axis=1)
            df['N_DELAY_CAUSES']    = (df[avail] > 0).sum(axis=1)
        if 'DISTANCE' in df.columns:
            df['DISTANCE_CAT'] = pd.cut(
                df['DISTANCE'], bins=[0,500,1000,2000,9999],
                labels=[0,1,2,3]).astype(int)

        st.session_state['df'] = df
        st.success("✅ Fitur operasional berhasil dikonstruksi!")

    st.subheader("6c. Scaling Numerik")
    num_cols_sc = df.select_dtypes(include=np.number).columns.tolist()
    if 'IS_DELAYED' in num_cols_sc:
        num_cols_sc.remove('IS_DELAYED')

    scale_method = st.selectbox("Metode scaling", ["Tidak di-scale (LightGBM tidak perlu)", "StandardScaler", "MinMaxScaler"])
    cols_scale   = st.multiselect("Pilih kolom", num_cols_sc, default=[])

    if st.button("🚀 Terapkan Scaling") and scale_method != "Tidak di-scale (LightGBM tidak perlu)" and cols_scale:
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        scaler = StandardScaler() if "Standard" in scale_method else MinMaxScaler()
        df[cols_scale] = scaler.fit_transform(df[cols_scale])
        st.session_state['df'] = df
        st.success(f"✅ Scaling selesai pada {len(cols_scale)} kolom")

    st.info("💡 LightGBM tidak memerlukan scaling — scaling opsional hanya jika dipakai sebagai perbandingan algoritma linear.")
    st.markdown(f"**Shape saat ini:** `{df.shape}` — Total {df.shape[1]} kolom")

# SUB PAGE: SPLIT
elif sub == "🎯 Split":
    section_header("07", "Seleksi Fitur & Split Data")

    if 'df' not in st.session_state:
        st.warning("⚠️ Load dataset dulu.")
        st.stop()

    df = st.session_state['df'].copy()
    TARGET = 'IS_DELAYED'

    if TARGET not in df.columns:
        st.error("Kolom IS_DELAYED tidak ditemukan. Pastikan sudah di-load dengan benar.")
        st.stop()

    leakage = ['DEPARTURE_DELAY','AIR_SYSTEM_DELAY','SECURITY_DELAY',
                'AIRLINE_DELAY','LATE_AIRCRAFT_DELAY','WEATHER_DELAY',
                'CANCELLED','ARRIVAL_DELAY']
    leak_exist = [c for c in leakage if c in df.columns]
    if leak_exist:
        st.warning(f"⚠️ Kolom berikut di-drop otomatis (data leakage): {leak_exist}")
        df.drop(columns=leak_exist, inplace=True)

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    feature_cols = [c for c in num_cols if c != TARGET]

    selected_features = st.multiselect("Pilih fitur (X)", feature_cols, default=feature_cols)
    test_size = st.slider("Ukuran test set (%)", 10, 30, 20)

    if st.button("✂️ Split Data"):
        X = df[selected_features]
        y = df[TARGET]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42, stratify=y_train)

        st.session_state.update({
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
            'feature_cols': selected_features
        })

        col1, col2, col3 = st.columns(3)
        col1.metric("Train Set", f"{X_train.shape[0]:,}")
        col2.metric("Val Set",   f"{X_val.shape[0]:,}")
        col3.metric("Test Set",  f"{X_test.shape[0]:,}")

        st.success(f"✅ Data berhasil dibagi! Fitur aktif: {len(selected_features)} kolom")


# SUB PAGE: MODELING
elif sub == "🤖 Modeling":
    section_header("08", "Modeling")

    keys = ['X_train','X_val','X_test','y_train','y_val','y_test']
    if not all(k in st.session_state for k in keys):
        st.warning("⚠️ Lakukan split data dulu di tab Split.")
        st.stop()

    X_train = st.session_state['X_train']
    X_val   = st.session_state['X_val']
    X_test  = st.session_state['X_test']
    y_train = st.session_state['y_train']
    y_val   = st.session_state['y_val']
    y_test  = st.session_state['y_test']

    model_choice = st.selectbox("Pilih Algoritma", ["LightGBM (Utama)", "Decision Tree (Pembanding)"])

    col1, col2 = st.columns(2)
    with col1:
        if "LightGBM" in model_choice:
            n_est    = st.slider("n_estimators", 50, 500, 300)
            lr       = st.select_slider("learning_rate", options=[0.01, 0.03, 0.05, 0.1, 0.2], value=0.05)
            n_leaves = st.slider("num_leaves", 15, 127, 63)
        else:
            max_d    = st.slider("max_depth", 3, 20, 10)

    if st.button("🚀 Latih Model"):
        with st.spinner("Melatih model..."):
            n_neg = (y_train == 0).sum()
            n_pos = (y_train == 1).sum()

            if "LightGBM" in model_choice:
                model = lgb.LGBMClassifier(
                    n_estimators     = n_est,
                    learning_rate    = lr,
                    num_leaves       = n_leaves,
                    scale_pos_weight = n_neg / n_pos,
                    random_state     = 42,
                    n_jobs           = -1,
                    verbose          = -1
                )
                model.fit(X_train, y_train,
                          eval_set=[(X_val, y_val)],
                          callbacks=[lgb.early_stopping(30, verbose=False)])
            else:
                model = DecisionTreeClassifier(
                    max_depth    = max_d,
                    class_weight = 'balanced',
                    random_state = 42
                )
                model.fit(X_train, y_train)

            y_pred      = model.predict(X_test)
            y_pred_prob = model.predict_proba(X_test)[:, 1]
            acc  = accuracy_score(y_test, y_pred)
            auc  = roc_auc_score(y_test, y_pred_prob)

        st.session_state.update({
            'model': model, 'y_pred': y_pred,
            'y_pred_prob': y_pred_prob,
            'acc': acc, 'auc': auc,
            'model_name': model_choice
        })

        col1, col2 = st.columns(2)
        col1.metric("Accuracy Score", f"{acc:.4f}")
        col2.metric("ROC-AUC Score",  f"{auc:.4f}")
        st.success(f"✅ Model {model_choice} berhasil dilatih dan disimpan!")

# SUB PAGE: EVALUASI
elif sub == "📊 Evaluasi":
    section_header("09", "Evaluasi Model")

    if 'model' not in st.session_state:
        st.warning("⚠️ Latih model dulu di tab Modeling.")
        st.stop()

    model        = st.session_state['model']
    y_test       = st.session_state['y_test']
    y_pred       = st.session_state['y_pred']
    y_pred_prob  = st.session_state['y_pred_prob']
    acc          = st.session_state['acc']
    auc          = st.session_state['auc']
    feature_cols = st.session_state.get('feature_cols', [])

    col1, col2 = st.columns(2)
    col1.metric("Accuracy", f"{acc:.4f} ({acc*100:.2f}%)")
    col2.metric("ROC-AUC",  f"{auc:.4f}")

    tab1, tab2, tab3 = st.tabs(["Classification Report", "Confusion Matrix & ROC", "Feature Importance"])

    with tab1:
        report = classification_report(y_test, y_pred, target_names=['Tidak Delay','Delay'], output_dict=True)
        st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

    with tab2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor(PLOT_BG)
        for ax in axes:
            ax.set_facecolor(PLOT_BG)
            ax.tick_params(colors=PLOT_TEXT)
            ax.spines[:].set_color(PLOT_BORDER)

        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                    xticklabels=['Tidak Delay','Delay'],
                    yticklabels=['Tidak Delay','Delay'],
                    cbar=False, annot_kws={'weight':'bold'})
        axes[0].set_title("Confusion Matrix", color=PLOT_TEXT, weight='bold')

        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        axes[1].plot(fpr, tpr, color=COLOR_PRIMARY, linewidth=2, label=f'AUC = {auc:.4f}')
        axes[1].plot([0,1],[0,1],'--', color='#8B949E', linewidth=1)
        axes[1].set_title("ROC Curve", color=PLOT_TEXT, weight='bold')
        axes[1].set_xlabel("FPR", color=PLOT_TEXT)
        axes[1].set_ylabel("TPR", color=PLOT_TEXT)
        axes[1].legend(labelcolor=PLOT_TEXT)
        axes[1].grid(alpha=0.2, color=PLOT_BORDER)

        plt.tight_layout()
        st.pyplot(fig)

    with tab3:
        if hasattr(model, 'feature_importances_') and feature_cols:
            fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)

            fig, ax = plt.subplots(figsize=(12, 6))
            fig.patch.set_facecolor(PLOT_BG)
            ax.set_facecolor(PLOT_BG)
            ax.tick_params(colors=PLOT_TEXT)
            ax.spines[:].set_color(PLOT_BORDER)
            fi.head(15).plot(kind='barh', ax=ax, color=COLOR_EMERALD, edgecolor='none') # Menggunakan warna Emerald baru
            ax.set_title("Top 15 Feature Importance", color=PLOT_TEXT, weight='bold')
            ax.invert_yaxis()
            ax.grid(alpha=0.2, axis='x', color=PLOT_BORDER)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Feature importance tidak tersedia untuk tipe arsitektur model ini.")

# SUB PAGE: EKSPOR
elif sub == "💾 Ekspor":
    section_header("10", "Ekspor Dataset & Model")

    if 'df' in st.session_state:
        df_export = st.session_state['df']
        csv = df_export.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Download dataset bersih (CSV)",
            data=csv,
            file_name="flights_clean.csv",
            mime="text/csv"
        )
        st.success(f"Dataset siap diekspor: {df_export.shape[0]:,} baris, {df_export.shape[1]} kolom")
    else:
        st.info("Belum ada dataset aktif yang dimuat.")

    if 'model' in st.session_state:
        import io, pickle
        model_bytes = pickle.dumps(st.session_state['model'])
        st.download_button(
            "⬇️ Download model (.pkl)",
            data=model_bytes,
            file_name="flight_model.pkl",
            mime="application/octet-stream"
        )