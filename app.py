# app.py (root)
import streamlit as st

st.set_page_config(
    page_title="Project UAS — Data Mining 3B",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #F8FAFC;
    color: #1E293B;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E2E8F0 !important;
}
[data-testid="stSidebarNav"] { display: none; }
.section-container {
    display: flex;
    align-items: center;
    margin-bottom: 20px;
    padding-bottom: 10px;
    border-bottom: 2px solid #E2E8F0;
}
.section-num {
    background-color: #0284C7;
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
.info-card {
    background-color: #FFFFFF;
    border-left: 4px solid #4F46E5;
    padding: 16px 20px;
    border-radius: 0 12px 12px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    margin-bottom: 24px;
    border-top: 1px solid #E2E8F0;
    border-right: 1px solid #E2E8F0;
    border-bottom: 1px solid #E2E8F0;
}
.custom-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    margin-right: 6px;
    margin-bottom: 10px;
}
.badge-blue    { background-color: #E0F2FE; color: #0369A1; }
.badge-indigo  { background-color: #E0E7FF; color: #4338CA; }
.badge-emerald { background-color: #D1FAE5; color: #065F46; }
.badge-amber   { background-color: #FEF3C7; color: #B45309; }
.badge-orange  { background-color: #FFEDD5; color: #9A3412; }
.badge-slate   { background-color: #F1F5F9; color: #475569; }
.nav-card {
    background-color: #FFFFFF;
    padding: 12px 16px;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    border: 1px solid #E2E8F0;
    margin-bottom: 10px;
    transition: all 0.2s ease-in-out;
}
.nav-card:hover {
    border-color: #4F46E5;
    box-shadow: 0 4px 6px rgba(79,70,229,0.1);
}
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 16px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
[data-testid="stMetricValue"] {
    color: #0284C7 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.stButton > button {
    background-color: #0284C7 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    padding: 8px 20px !important;
    transition: background 0.2s !important;
}
.stButton > button:hover { background-color: #0369A1 !important; }
.stTabs [data-baseweb="tab-list"] {
    background: #F1F5F9 !important;
    border-radius: 8px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid #E2E8F0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #64748B !important;
    border-radius: 6px !important;
    padding: 6px 16px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background: #FFFFFF !important;
    color: #0284C7 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
}
[data-testid="stDataFrame"] {
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
    overflow: hidden;
}
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    color: #1E293B !important;
}
.stSlider > div > div > div { background: #0284C7 !important; }
div[data-testid="stSidebar"] div.stRadio {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

#SIDEBAR
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 20px;">
        <div style="font-size:22px; font-weight:700; color:#0F172A; letter-spacing:-0.5px;">
            🎓 Project UAS
        </div>
        <div style="font-size:12px; color:#64748B; margin-top:4px;">
            Data Mining · Kelas 3B · 2026
        </div>
    </div>
    <div style="height:1px; background:#E2E8F0; margin-bottom:20px;"></div>
    <div style="font-size:11px; color:#94A3B8; font-weight:600;
                letter-spacing:1px; margin-bottom:12px;">
        NAVIGASI PROJECT
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✈️ Project 1", use_container_width=True):
            st.switch_page("Pages/1_project1_flight.py")
    with col2:
        if st.button("📈 Project 2", use_container_width=True):
            st.switch_page("Pages/2_project2_LSTM.py")

    st.markdown("""
    <div style="height:1px; background:#E2E8F0; margin:20px 0;"></div>
    <div style="font-size:11px; color:#94A3B8; font-weight:600;
                letter-spacing:1px; margin-bottom:12px;">
        INFO DATASET
    </div>
    <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px;
                padding:12px; margin-bottom:10px;">
        <div style="font-size:12px; font-weight:600; color:#0284C7; margin-bottom:4px;">
            ✈️ Project 1
        </div>
        <div style="font-size:11px; color:#64748B; line-height:1.5;">
            2015 Flight Delays<br>LightGBM + Decision Tree
        </div>
    </div>
    <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px;
                padding:12px; margin-bottom:10px;">
        <div style="font-size:12px; font-weight:600; color:#4F46E5; margin-bottom:4px;">
            📈 Project 2
        </div>
        <div style="font-size:11px; color:#64748B; line-height:1.5;">
            Air Pollution China<br>LSTM + CNN-LSTM
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="position:fixed; bottom:0; left:0; width:244px;
                background:#FFFFFF; border-top:1px solid #E2E8F0;
                padding:14px 16px; font-size:11px; color:#94A3B8;">
        <div style="font-weight:600; color:#1E293B; margin-bottom:2px;">
            Kelompok · Kelas 3B
        </div>
        <div>Mata Kuliah Data Mining</div>
    </div>
    """, unsafe_allow_html=True)

#BERANDA
st.markdown("""
<div style="padding: 40px 0 32px;">
    <div style="font-size:13px; color:#0284C7; font-weight:600;
                letter-spacing:2px; margin-bottom:12px;">
        DATA MINING · UAS 2026 · KELAS 3B
    </div>
    <h1 style="font-size:38px; font-weight:700; color:#0F172A;
               line-height:1.2; letter-spacing:-1px; margin-bottom:16px;">
        Sistem Analisis &amp; Prediksi<br>
        <span style="color:#0284C7;">Data Mining Terpadu</span>
    </h1>
    <p style="font-size:15px; color:#64748B; max-width:560px; line-height:1.7; margin-bottom:0;">
        Platform analisis data terintegrasi yang menggabungkan klasifikasi
        data tabel dengan deep learning untuk forecasting deret waktu.
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Project", "2 Project")
col2.metric("Algoritma",     "3 Model")
col3.metric("Dataset",       "2 Dataset")
col4.metric("Deep Learning", "LSTM + CNN")

st.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:14px;
                padding:28px; box-shadow:0 1px 3px rgba(0,0,0,0.05);
                border-top:4px solid #0284C7;">
        <div style="font-size:32px; margin-bottom:12px;">✈️</div>
        <div style="font-size:18px; font-weight:700; color:#0F172A; margin-bottom:8px;">
            Project 1: Klasifikasi Delay
        </div>
        <div style="font-size:13px; color:#64748B; line-height:1.7; margin-bottom:16px;">
            Klasifikasi keterlambatan penerbangan domestik AS menggunakan
            pipeline preprocessing lengkap dan algoritma LightGBM.
        </div>
        <div style="display:flex; flex-wrap:wrap; gap:6px; margin-bottom:16px;">
            <span class="custom-badge badge-blue">Flight Delays 2015</span>
            <span class="custom-badge badge-indigo">LightGBM</span>
            <span class="custom-badge badge-amber">Decision Tree</span>
        </div>
        <div style="font-size:12px; color:#94A3B8;">
            5.8 juta baris · 31 kolom · Klasifikasi Biner
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Buka Project 1 →", use_container_width=True, key="btn_p1"):
        st.switch_page("Pages/1_project1_flight.py")

with col2:
    st.markdown("""
    <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:14px;
                padding:28px; box-shadow:0 1px 3px rgba(0,0,0,0.05);
                border-top:4px solid #4F46E5;">
        <div style="font-size:32px; margin-bottom:12px;">📈</div>
        <div style="font-size:18px; font-weight:700; color:#0F172A; margin-bottom:8px;">
            Project 2: Forecasting LSTM
        </div>
        <div style="font-size:13px; color:#64748B; line-height:1.7; margin-bottom:16px;">
            Peramalan konsentrasi PM2.5 polusi udara Beijing menggunakan
            arsitektur deep learning LSTM dan CNN-LSTM hybrid.
        </div>
        <div style="display:flex; flex-wrap:wrap; gap:6px; margin-bottom:16px;">
            <span class="custom-badge badge-emerald">Air Pollution China</span>
            <span class="custom-badge badge-indigo">LSTM</span>
            <span class="custom-badge badge-orange">CNN-LSTM</span>
        </div>
        <div style="font-size:12px; color:#94A3B8;">
            43.800 baris · Time Series · Regresi
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Buka Project 2 →", use_container_width=True, key="btn_p2"):
        st.switch_page("Pages/2_project2_LSTM.py")

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div style="font-size:11px; color:#94A3B8; font-weight:600;
            letter-spacing:1px; margin-bottom:16px;">
    PIPELINE PREPROCESSING (BERLAKU UNTUK KEDUA PROJECT)
</div>
""", unsafe_allow_html=True)

steps = [
    ("📂", "Load Dataset",  "Sampling & validasi format"),
    ("🔍", "EDA",           "Eksplorasi & distribusi"),
    ("🩹", "Missing Value", "Imputasi & drop"),
    ("📉", "Outlier",       "IQR & winsorizing"),
    ("🔤", "Encoding",      "Label & frequency"),
    ("🔄", "Transformasi",  "Feature engineering"),
    ("🎯", "Split Data",    "Train / val / test"),
    ("🤖", "Modeling",      "Training & tuning"),
]

cols = st.columns(4)
for i, (icon, title, desc) in enumerate(steps):
    with cols[i % 4]:
        st.markdown(f"""
        <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px;
                    padding:16px; margin-bottom:12px; min-height:100px;">
            <div style="font-size:20px; margin-bottom:6px;">{icon}</div>
            <div style="font-weight:600; font-size:13px; color:#0F172A; margin-bottom:4px;">{title}</div>
            <div style="font-size:11px; color:#94A3B8;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)