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

# --- HELPER FUNCTIONS (CSS & BACKGROUND) 🎨 ---
# (Giữ nguyên 136 dòng CSS và helper functions... )

@st.cache_data
def get_base64_of_bin_file(bin_file):
    """Read file and base64 encode"""
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception as e:
        return None

def set_bg_and_css(file_path):
    """
    Improved CSS: Background image, transparency, blur, and
    COMPLETE FIX for Date Picker visibility including day names.
    """
    
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
    /* 1. NỀN ẢNH VÀ LỚP PHỦ MỜ (Giữ nguyên) */
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
    
    /* 2. MÀU CHỮ TRẮNG */
    h1, h2, h3, h4, p, [data-testid="stSidebar"] > * {{
        color: white !important;
    }}
    
    [data-testid="stMarkdown"] p, [data-testid="stMarkdown"] li {{
        color: white !important;
    }}

    [data-testid="stSidebar"] {{
        background-color: rgba(0, 0, 0, 0.3) !important;
    }}
    
    [data-testid="stVerticalBlock"], 
    [data-testid="stHorizontalBlock"] {{
        background-color: transparent !important; 
    }}
    
    /* 3. ĐỊNH DẠNG CHO METRIC (LỚN) */
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
    
    /* CSS CHO CÁC Ô METRIC NHỎ (T+1 đến T+4) */
    [data-testid="stHorizontalBlock"] [data-testid="stMetric"] div[data-testid="stMetricValue"] {{
        font-size: 2.5rem !important; /* Kích thước chữ nhỏ hơn */
    }}
    
    [data-testid="stHorizontalBlock"] [data-testid="stMetric"] div[data-testid="stMetricLabel"] p {{
        font-size: 1rem !important; /* Kích thước label nhỏ hơn */
    }}
    
    [data-testid="stMetric"] {{
        background-color: rgba(255, 255, 255, 0.15); 
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }}
    
    /* 4. SỬA LỖI HIỂN THỊ LỊCH */
    [data-testid="stSidebar"] [data-testid="stDateInput"] [data-testid="stWidgetLabel"] {{
        color: black !important;
    }}
    .stDateInput input, .stDateInput [data-testid="baseInput-date"], .stDateInput [data-testid="StyledDatePickerIcon"] {{
        color: black !important;
    }}
    .react-datepicker-popper,
    .react-datepicker-popper * {{
        color: black !important;
    }}
    .react-datepicker-popper .react-datepicker__day--selected,
    .react-datepicker-popper .react-datepicker__day--keyboard-selected {{
        background-color: #ff4b4b !important;
        color: white !important; 
    }}
    .react-datepicker-popper .react-datepicker__day--outside-month {{
        color: #ccc !important; 
    }}
    .react-datepicker-popper .react-datepicker__day--today {{
        border: 1px solid #ccc !important;
        border-radius: 50%;
        font-weight: bold;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# --- LOAD DATA AND MODEL FUNCTIONS ---

@st.cache_data
def load_models_and_features():
    """Load trained model and feature list."""
    models = {}
    horizons = ["t1", "t2", "t3", "t4", "t5"]
    try:
        for h in horizons:
            models[h] = joblib.load(f'model_{h}.joblib')
        with open('selected_features.json', 'r') as f:
            features = json.load(f)
        return models, features
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình hoặc features: {e}")
        return None, None

@st.cache_data
def load_and_process_data(data_path='hcm.xlsx'):
    """Load raw data and run the entire feature engineering pipeline."""
    try:
        df_raw = pd.read_excel(data_path)
        
        # *** SỬA LỖI: Tải lại cột 'temp' để hiển thị nhiệt độ ngày T ***
        df_original = df_raw[['datetime', 'temp']].copy() 
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
    # --- SET BACKGROUND ---
    set_bg_and_css("hcm.jpg") 
    # ----------------------

    # Load resources
    models, features = load_models_and_features() 
    df_engineered, df_original = load_and_process_data() 

    df_engineered = df_engineered.ffill().bfill()

    if models is None or df_engineered.empty or df_original.empty:
        st.stop()

    st.title("☀️ Ho Chi Minh City Weather") 
    st.markdown("---") 

    # ---- Sidebar Interface ----
    st.sidebar.header("Controls 🗓️") 
    st.sidebar.markdown("Select a day to see the **Temperature Forecast**.")

    MAX_ROLLING_WINDOW = 56 
    FORECAST_HORIZON = 5
    
    first_valid_date = df_engineered.iloc[0]['date_col']
    last_valid_date = df_engineered.iloc[-FORECAST_HORIZON - 1]['date_col']

    selected_date = st.sidebar.date_input(
        "Select Day",
        value=first_valid_date,
        min_value=first_valid_date,
        max_value=last_valid_date
    )
    selected_datetime = pd.to_datetime(selected_date)

    # ---- Data Retrieval and Prediction Logic ----
    data_t = df_engineered[df_engineered['date_col'] == selected_date]

    if data_t.empty:
        st.error(f"Không tìm thấy dữ liệu đã xử lý cho ngày {selected_date}.")
        st.stop()

    X_predict_base = data_t.iloc[0]
    
    predictions = []
    horizons = ["t1", "t2", "t3", "t4", "t5"] # Vẫn dự báo 5 ngày cho biểu đồ

    for i, h in enumerate(horizons, 1):
            model = models[h]
            feature_list = features[h] 
            
            valid_features = [f for f in feature_list if f in X_predict_base.index]
            
            X_pred_h = X_predict_base[valid_features].to_frame().T
            X_pred_h = X_pred_h.astype(float)
            
            prediction_temp = model.predict(X_pred_h)[0]
            forecast_date = selected_date + timedelta(days=i)
            
            predictions.append({
                "Ngày dự báo": forecast_date,
                "Nhiệt độ dự báo (°C)": prediction_temp
            })

    # ---- DISPLAY RESULTS (New Layout) ----
    
    st.subheader(f"**{selected_date.strftime('%A, %B %d, %Y')}**") 
    
    df_forecast = pd.DataFrame(predictions)
    
    # Chuyển đổi ngày tháng để định dạng
    df_forecast['Ngày dự báo'] = pd.to_datetime(df_forecast['Ngày dự báo'])

    # --- 1. Hiển thị Metric T (Cái lớn) ---
    actual_data_t = df_original[df_original['date_col'] == selected_date]
    
    if not actual_data_t.empty:
        actual_temp_t = actual_data_t['temp'].values[0]
        st.metric(
            f"🌡️ Temperatue {selected_date.strftime('%A, %d-%m')}", 
            f"{actual_temp_t:.1f} °C",
            help="Đây là nhiệt độ thực tế của ngày T (ngày bạn đã chọn)."
        )
    else:
        st.metric(f"🌡️ Temperatue  {selected_date.strftime('%d-%m')}", "N/A")
    
    st.markdown("---") 

    # --- 2. Hiển thị 4 Metrics (T+1 đến T+4) ---
    if len(df_forecast) >= 4:
        # Lấy 4 hàng đầu tiên (T+1, T+2, T+3, T+4)
        forecast_t1 = df_forecast.iloc[0]
        forecast_t2 = df_forecast.iloc[1]
        forecast_t3 = df_forecast.iloc[2]
        forecast_t4 = df_forecast.iloc[3]

        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                f"{forecast_t1['Ngày dự báo'].strftime('%A, %d-%m')}",
                f"{forecast_t1['Nhiệt độ dự báo (°C)']:.1f} °C"
            )
        with col2:
            st.metric(
                f"{forecast_t2['Ngày dự báo'].strftime('%A, %d-%m')}",
                f"{forecast_t2['Nhiệt độ dự báo (°C)']:.1f} °C"
            )
        with col3:
            st.metric(
                f"{forecast_t3['Ngày dự báo'].strftime('%A, %d-%m')}",
                f"{forecast_t3['Nhiệt độ dự báo (°C)']:.1f} °C"
            )
        with col4:
            st.metric(
                f"{forecast_t4['Ngày dự báo'].strftime('%A, %d-%m')}",
                f"{forecast_t4['Nhiệt độ dự báo (°C)']:.1f} °C"
            )

    st.markdown("---") 

    # --- 3. Hiển thị Biểu đồ (Vẫn hiển thị 5 ngày) ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_forecast['Ngày dự báo'], # Dùng df_forecast gốc (với datetime)
        y=df_forecast['Nhiệt độ dự báo (°C)'],
        mode='lines+markers+text',
        text=df_forecast['Nhiệt độ dự báo (°C)'].round(1),
        textposition="top center",
        line=dict(color='cyan', width=3),
        marker=dict(color='cyan', size=8),
        name="Dự báo"
    ))
    fig.update_layout(
        title="Temperature trend for the next 5 days",
        xaxis_title="Day",
        yaxis_title="Temp (°C)",
        plot_bgcolor='rgba(0,0,0,0.2)', 
        paper_bgcolor='rgba(0,0,0,0)', 
        font=dict(color='white'),
        xaxis=dict(gridcolor='rgba(255, 255, 255, 0.2)'),
        yaxis=dict(gridcolor='rgba(255, 255, 255, 0.2)')
    )
    st.plotly_chart(fig, use_container_width=True)


except FileNotFoundError as e:
    st.error(f"Lỗi: File không tìm thấy. Vui lòng kiểm tra tên file: {e.filename if hasattr(e, 'filename') else e}")
    st.warning("Ghi chú: Đảm bảo các file Model, Features, và Data ở cùng thư mục.")
except Exception as e:
    st.error(f"Đã xảy ra lỗi không mong muốn khi chạy ứng dụng:")
    st.exception(e)

st.sidebar.markdown("---")
st.sidebar.caption("© 2025 AI Weather Forecast")