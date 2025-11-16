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
import requests # <- Thêm thư viện
import io # <- Thêm thư viện

# --- URL ĐỂ TẢI FILE ---
# !!! THAY THẾ BẰNG LINK REPO HUGGING FACE CỦA BẠN !!!
# Thêm "/resolve/main/" vào cuối link.
BASE_URL = "https://huggingface.co/ingresp/my-weather-models/resolve/main/"
# ------------------------


# --- HELPER FUNCTIONS ---

@st.cache_data
def download_file(file_name):
    """Tải file từ Hugging Face nếu nó chưa tồn tại."""
    local_path = file_name
    
    # Nếu file chưa có, tải nó về
    if not os.path.exists(local_path):
        url = BASE_URL + file_name
        try:
            with st.spinner(f"Đang tải {file_name}..."):
                response = requests.get(url)
                response.raise_for_status() # Báo lỗi nếu tải thất bại
                
                with open(local_path, 'wb') as f:
                    f.write(response.content)
        except requests.exceptions.RequestException as e:
            st.error(f"Lỗi khi tải file {file_name}: {e}")
            st.stop()
            
    return local_path

@st.cache_data
def get_base64_of_bin_file(bin_file):
    """Read file and base64 encode"""
    try:
        # Đảm bảo file ảnh nền đã được tải về
        download_file(bin_file)
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception as e:
        return None

def set_bg_and_css(file_path):
    # (Code CSS của bạn không thay đổi...)
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
    /* ... (Toàn bộ 100+ dòng CSS của bạn ở đây) ... */
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
    [data-testid="stVerticalBlock"], 
    [data-testid="stHorizontalBlock"] {{
        background-color: transparent !important; 
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

# --- LOAD DATA AND MODEL FUNCTIONS (ĐÃ CẬP NHẬT) ---

@st.cache_data
def load_models_and_features():
    """Tải model và features từ file đã download."""
    models = {}
    horizons = ["t1", "t2", "t3", "t4", "t5"]
    try:
        # Tải 5 mô hình
        for h in horizons:
            model_path = download_file(f'model_{h}.joblib')
            models[h] = joblib.load(model_path)
        
        # Tải 1 file JSON
        features_path = download_file('selected_features.json')
        with open(features_path, 'r') as f:
            features = json.load(f)
            
        return models, features
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình hoặc features: {e}")
        return None, None

@st.cache_data
def load_and_process_data(data_path='hcm.xlsx'):
    """Tải data và chạy feature engineering."""
    try:
        # Tải file data
        data_file_path = download_file(data_path)
        df_raw = pd.read_excel(data_file_path)
        
        # Tải trước file ảnh nền để nó sẵn sàng
        download_file("hcm.jpg")
        
        df_original = df_raw[['datetime', 'temp', 'feelslike', 'humidity']].copy() 
        df_original['datetime'] = pd.to_datetime(df_original['datetime'])
        df_original['date_col'] = df_original['datetime'].dt.date
        
        df_engineered = create_all_features(df_raw)
        
        if 'date_col' not in df_engineered.columns:
            df_engineered['date_col'] = pd.to_datetime(df_engineered['datetime']).dt.date
        
        return df_engineered, df_original
    except KeyError as e:
        st.error(f"Lỗi: Không tìm thấy cột {e} trong file Excel.")
        return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        st.error(f"Error processing data: {e}")
        return pd.DataFrame(), pd.DataFrame()

# ---- Initialize App ----
st.set_page_config(layout="wide", page_title="Weather Forecast")

try:
    # Load resources (Hàm này giờ sẽ tự động download)
    models, features = load_models_and_features() 
    df_engineered, df_original = load_and_process_data() 

    # --- SET BACKGROUND ---
    # Phải gọi sau khi load_and_process_data đã tải file hcm.jpg
    set_bg_and_css("hcm.jpg") 
    # ----------------------

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

    # ---- Data Retrieval and Prediction Logic ----
    data_t = df_engineered[df_engineered['date_col'] == selected_date]

    if data_t.empty:
        st.error(f"Không tìm thấy dữ liệu đã xử lý cho ngày {selected_date}.")
        st.stop()

    X_predict_base = data_t.iloc[0]
    
    predictions = []
    horizons = ["t1", "t2", "t3", "t4", "t5"] 

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
    df_forecast['Ngày dự báo'] = pd.to_datetime(df_forecast['Ngày dự báo'])

    # --- 1. Hiển thị Metric T (Cái lớn) ---
    actual_data_t = df_original[df_original['date_col'] == selected_date]
    
    if not actual_data_t.empty:
        actual_temp_t = actual_data_t['temp'].values[0]
        st.metric(
            f"🌡️ Nhiệt độ (Ngày T: {selected_date.strftime('%A, %d-%m')})", 
            f"{actual_temp_t:.1f} °C",
            help="Đây là nhiệt độ thực tế của ngày T (ngày bạn đã chọn)."
        )
        
        actual_feelslike_t = actual_data_t['feelslike'].values[0]
        actual_humidity_t = actual_data_t['humidity'].values[0]
        
        col_feels, col_humid = st.columns(2)
        with col_feels:
            st.metric(
                f"🥵 Cảm giác như", 
                f"{actual_feelslike_t:.1f} °C",
            )
        with col_humid:
            st.metric(
                f"💧 Độ ẩm",
                f"{actual_humidity_t:.0f} %"
            )
    else:
        st.metric(f"🌡️ Nhiệt độ (Ngày T: {selected_date.strftime('%d-%m')})", "N/A")
    
    st.markdown("---") 

    # --- 2. Hiển thị 4 Metrics (T+1 đến T+4) ---
    st.subheader("Dự báo 4 ngày tới")
    if len(df_forecast) >= 4:
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
        x=df_forecast['Ngày dự báo'], 
        y=df_forecast['Nhiệt độ dự báo (°C)'],
        mode='lines+markers+text',
        text=df_forecast['Nhiệt độ dự báo (°C)'].round(1),
        textposition="top center",
        line=dict(color='cyan', width=3),
        marker=dict(color='cyan', size=8),
        name="Dự báo"
    ))
    fig.update_layout(
        title="Xu hướng nhiệt độ 5 ngày tới (T+1 đến T+5)",
        xaxis_title="Ngày",
        yaxis_title="Nhiệt độ (°C)",
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
