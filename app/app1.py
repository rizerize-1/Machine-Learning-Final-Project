import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np
import base64
import plotly.graph_objects as go
from feature_engineering import create_all_features 
from datetime import timedelta 
import os
import requests
import io

# --- FILE DOWNLOAD URL ---
# !!! REPLACE WITH YOUR HUGGING FACE REPO URL !!!
BASE_URL = "https://huggingface.co/ingresp/my-weather-models/resolve/main/"
# ------------------------

# --- HELPER FUNCTIONS ---

@st.cache_data
def download_file(file_name):
    local_path = file_name
    if not os.path.exists(local_path):
        url = BASE_URL + file_name
        try:
            with st.spinner(f"Downloading {file_name}..."):
                response = requests.get(url)
                response.raise_for_status()
                with open(local_path, 'wb') as f:
                    f.write(response.content)
        except requests.exceptions.RequestException as e:
            st.error(f"Error downloading {file_name}: {e}")
            st.stop()
    return local_path

@st.cache_data
def get_base64_of_bin_file(bin_file):
    try:
        download_file(bin_file)
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception as e:
        return None

def set_bg_and_css(file_path):
    file_extension = file_path.split('.')[-1].lower()
    if file_extension in ['jpg', 'jpeg']:
        mime_type = 'image/jpeg'
    elif file_extension == 'png':
        mime_type = 'image/png'
    else:
        mime_type = f'image/{file_extension}'

    img_base64 = get_base64_of_bin_file(file_path)
    
    if img_base64 is None:
        st.markdown("""<style></style>""", unsafe_allow_html=True)
        return 

    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:{mime_type};base64,{img_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    [data-testid="stAppViewContainer"]::before {{
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(0, 0, 0, 0.4); 
        backdrop-filter: blur(3px); 
        z-index: -1;
    }}
    h1, h2, h3, h4, p, [data-testid="stSidebar"] > * {{
        color: white !important;
    }}
    [data-testid="stMarkdown"] p, [data-testid="stMarkdown"] li {{
        color: white !important;
    }}
    [data-testid="stSidebar"] {{
        background-color: rgba(0, 0, 0, 0.3) !important;
    }}
    div[data-testid="stMetricValue"] {{
        font-size: 5rem !important; 
        color: white !important;
        font-weight: bold;
        text-shadow: 3px 3px 6px #000000;
    }}
    div[data-testid="stMetricLabel"] p {{
        font-size: 1.2rem !important;
        font-weight: 500;
        color: #E0E0E0 !important;
    }}
    [data-testid="stHorizontalBlock"] [data-testid="stMetric"] div[data-testid="stMetricValue"] {{
        font-size: 2.5rem !important; 
    }}
    [data-testid="stHorizontalBlock"] [data-testid="stMetric"] div[data-testid="stMetricLabel"] p {{
        font-size: 1rem !important; 
    }}
    [data-testid="stMetric"] {{
        background-color: rgba(255, 255, 255, 0.15); 
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }}
    .stDateInput input {{ color: black !important; }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# --- LOAD DATA AND MODEL FUNCTIONS ---

@st.cache_data
def load_models_and_features():
    models = {}
    horizons = ["t1", "t2", "t3", "t4", "t5"]
    try:
        for h in horizons:
            model_path = download_file(f'model_{h}.joblib')
            models[h] = joblib.load(model_path)
        features_path = download_file('selected_features.json')
        with open(features_path, 'r') as f:
            features = json.load(f)
        return models, features
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

@st.cache_data
def load_and_process_data(data_path='hcm.xlsx'):
    try:
        data_file_path = download_file(data_path)
        df_raw = pd.read_excel(data_file_path)
        download_file("hcm.jpg")
        
        # Keep actual data for comparison
        df_original = df_raw[['datetime', 'temp', 'feelslike', 'humidity']].copy() 
        df_original['datetime'] = pd.to_datetime(df_original['datetime'])
        df_original['date_col'] = df_original['datetime'].dt.date
        
        df_engineered = create_all_features(df_raw)
        if 'date_col' not in df_engineered.columns:
            df_engineered['date_col'] = pd.to_datetime(df_engineered['datetime']).dt.date
        
        return df_engineered, df_original
    except Exception as e:
        st.error(f"Error processing data: {e}")
        return pd.DataFrame(), pd.DataFrame()

# ---- Initialize App ----
st.set_page_config(layout="wide", page_title="Weather Forecast")

try:
    models, features = load_models_and_features() 
    df_engineered, df_original = load_and_process_data() 
    set_bg_and_css("hcm.jpg") 
    
    df_engineered = df_engineered.ffill().bfill()

    if models is None or df_engineered.empty:
        st.stop()

    st.title("☀️ Ho Chi Minh City Weather") 
    st.markdown("---") 

    # ---- Sidebar ----
    st.sidebar.header("Controls 🗓️") 
    first_valid_date = df_engineered.iloc[0]['date_col']
    last_valid_date = df_engineered.iloc[-5]['date_col'] # Ensure space for predictions
    
    target_default = pd.Timestamp("2025-11-14").date()
    default_val = target_default if first_valid_date <= target_default <= last_valid_date else last_valid_date

    selected_date = st.sidebar.date_input("Select Day", value=default_val, min_value=first_valid_date, max_value=last_valid_date)

    # ---- LOGIC DỰ BÁO MỚI (SHIFT) ----
    
    # 1. DỰ BÁO CHO NGÀY T (Dùng dữ liệu T-1)
    pred_day_t = None
    date_prev = selected_date - timedelta(days=1)
    data_prev = df_engineered[df_engineered['date_col'] == date_prev]
    
    if not data_prev.empty:
        # Dùng model T1 cho dữ liệu T-1 để ra T
        X_prev = data_prev.iloc[0].to_frame().T
        valid_cols_t1 = [c for c in features['t1'] if c in X_prev.columns]
        X_prev = X_prev[valid_cols_t1].astype(float)
        pred_day_t = models['t1'].predict(X_prev)[0]

    # 2. DỰ BÁO TƯƠNG LAI (T+1 -> T+4) (Dùng dữ liệu T)
    future_predictions = []
    data_curr = df_engineered[df_engineered['date_col'] == selected_date]
    
    if not data_curr.empty:
        X_curr_base = data_curr.iloc[0]
        
        # Dự báo 4 ngày tới (T+1, T+2, T+3, T+4)
        # Dùng lần lượt model t1, t2, t3, t4
        for i, h in enumerate(['t1', 't2', 't3', 't4'], start=1):
            valid_cols = [c for c in features[h] if c in X_curr_base.index]
            X_in = X_curr_base[valid_cols].to_frame().T.astype(float)
            
            pred_val = models[h].predict(X_in)[0]
            future_date = selected_date + timedelta(days=i)
            
            future_predictions.append({
                "Date": future_date,
                "Temperature": pred_val
            })

    # ---- HIỂN THỊ KẾT QUẢ ----
    
    st.subheader(f"**{selected_date.strftime('%A, %B %d, %Y')}**") 

    # --- 1. Metric T (DỰ BÁO TỪ QUÁ KHỨ) ---
    # Lấy thêm dữ liệu thực tế để so sánh (nếu muốn)
    actual_row = df_original[df_original['date_col'] == selected_date]
    actual_temp = actual_row['temp'].values[0] if not actual_row.empty else "N/A"
    
    col_main, col_sub = st.columns([2, 1])
    
    with col_main:
        if pred_day_t is not None:
            st.metric(
                f"🌡️ Predicted Temp (Day T)", 
                f"{pred_day_t:.1f} °C",
                delta=f"{pred_day_t - actual_temp:.1f} °C vs Actual" if isinstance(actual_temp, (int, float)) else None,
                delta_color="off", # Màu xám trung tính
                help=f"This value is PREDICTED by the model using data from yesterday ({date_prev}). Actual was {actual_temp}"
            )
        else:
            st.metric("Predicted Temp (Day T)", "No Data (T-1 missing)")

    with col_sub:
        # Hiển thị thông tin phụ (Thực tế)
        if not actual_row.empty:
            st.markdown(f"**Actual:** {actual_temp:.1f} °C")
            st.markdown(f"**Feels:** {actual_row['feelslike'].values[0]:.1f} °C")
            st.markdown(f"**Humid:** {actual_row['humidity'].values[0]:.0f}%")

    st.markdown("---") 

    # --- 2. 4-Day Forecast Metrics (T+1 -> T+4) ---
    st.subheader("Forecast (Next 4 Days)")
    if len(future_predictions) >= 4:
        cols = st.columns(4)
        for idx, col in enumerate(cols):
            item = future_predictions[idx]
            with col:
                st.metric(
                    f"{item['Date'].strftime('%a, %d-%m')}",
                    f"{item['Temperature']:.1f} °C"
                )

    st.markdown("---") 

    # --- 3. Chart (Kết hợp T dự báo và T+1..4) ---
    # Tạo dữ liệu cho biểu đồ: Gồm Day T (dự báo) + 4 ngày tới
    chart_data = []
    if pred_day_t is not None:
        chart_data.append({"Date": pd.to_datetime(selected_date), "Temperature": pred_day_t, "Type": "Predicted (T)"})
    
    for item in future_predictions:
        chart_data.append({"Date": pd.to_datetime(item['Date']), "Temperature": item['Temperature'], "Type": "Forecast"})
        
    df_chart = pd.DataFrame(chart_data)

    if not df_chart.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_chart['Date'], 
            y=df_chart['Temperature'],
            mode='lines+markers+text',
            text=df_chart['Temperature'].round(1),
            textposition="top center",
            line=dict(color='cyan', width=3),
            marker=dict(color='cyan', size=8),
            name="Prediction"
        ))
        fig.update_layout(
            title="Temperature Prediction Trend (T to T+4)",
            xaxis_title="Date",
            yaxis_title="Temperature (°C)",
            plot_bgcolor='rgba(0,0,0,0.2)', 
            paper_bgcolor='rgba(0,0,0,0)', 
            font=dict(color='white'),
            xaxis=dict(gridcolor='rgba(255, 255, 255, 0.2)'),
            yaxis=dict(gridcolor='rgba(255, 255, 255, 0.2)')
        )
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"An unexpected error occurred:")
    st.exception(e)

st.sidebar.markdown("---")
st.sidebar.caption("© 2025 AI Weather Forecast")
