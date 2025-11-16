import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings("ignore")

def create_all_features(df):
    # --- Basic cleaning and encoding ---
    df = df.sort_values('datetime').reset_index(drop=True)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['sunrise'] = pd.to_datetime(df['sunrise'])
    df['sunset'] = pd.to_datetime(df['sunset'])
    
    df.drop(columns=['name', 'snow', 'snowdepth', 'conditions', 'description', 'icon', 'stations', 'severerisk'], inplace=True)
    
    df['has_rain'] = (
        df['preciptype']
        .fillna('')
        .str.lower()
        .str.contains('rain')
        .astype(int)
    )
    df.drop(columns=['preciptype'], inplace=True)
    
    df['precip'] = df['precip'] * 25.4
    pct_cols = ['humidity', 'cloudcover', 'precipprob', 'precipcover']
    df[pct_cols] = df[pct_cols] / 100.0
    wind_cols = ['windspeed', 'windgust']
    df[wind_cols] = df[wind_cols] / 2.237
    df['visibility'] = df['visibility'] * 1.609
    df['solarenergy_wm2eq'] = df['solarenergy'] * 11.6
    
    df['day_length'] = (
        (df['sunset'] - df['sunrise'])
        .dt.total_seconds()
        .div(3600)
        .clip(lower=0, upper=24)
    )
    df.drop(columns=['sunrise', 'sunset'], inplace=True)

    # --- 4. Physics-based feature engineering (Cell 244) ---
    df['radiation_efficiency'] = (df['solarradiation'] + 0.1 * df['solarenergy']) / (df['cloudcover'].clip(lower=1) + 1e-3)
    df['rad_per_hour'] = df['solarenergy'] / (df['day_length'] + 1e-3)
    df['dew_humidity_ratio'] = df['dew'] / df['humidity'].replace(0, np.nan)
    df['precip_intensity'] = (df['precip'] / (df['precipcover'].replace(0, np.nan) + 1e-3)).clip(upper=100)
    
    df['wind_u'] = df['windspeed'] * np.cos(np.deg2rad(df['winddir']))
    df['wind_v'] = df['windspeed'] * np.sin(np.deg2rad(df['winddir']))
    
    df['storminess'] = (df['windspeed'] ** 2) * (df['precipprob'] / 100.0)
    df['dayofyear'] = df['datetime'].dt.dayofyear
    df['doy_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25)

    # --- 6. Features engineering (Rolling) (Cell 246) ---
    windows = [3, 7, 14, 28, 56]
    
    for w in [7, 28]:
        df[f'wind_u_mean_w{w}'] = df['wind_u'].rolling(w, 1).mean()
        df[f'wind_v_mean_w{w}'] = df['wind_v'].rolling(w, 1).mean()
        df[f'wind_u_var_w{w}'] = df['wind_u'].rolling(w, 1).var()
        df[f'wind_v_var_w{w}'] = df['wind_v'].rolling(w, 1).var()

    df['pressure_change_1d'] = df['sealevelpressure'].diff(1)
    df['humidity_change_1d'] = df['humidity'].diff(1)
    df['dew_change_1d'] = df['dew'].diff(1)

    for w in [7, 28]:
        df[f'pressure_mean_w{w}'] = df['sealevelpressure'].rolling(w, 1).mean()
        df[f'pressure_var_w{w}'] = df['sealevelpressure'].rolling(w, 1).var()
        df[f'humidity_mean_w{w}'] = df['humidity'].rolling(w, 1).mean()
        df[f'humidity_var_w{w}'] = df['humidity'].rolling(w, 1).var()
    
    m = df['datetime'].dt.month
    df['is_wet_season']   = m.isin([5,6,7,8,9,10]).astype(int)
    df['season_progress'] = np.where(m < 5, 0, np.where(m > 10, 1, (m-5)/5.0))

    wet14 = df['precip'].rolling(14,1).sum()
    df['soil_wetness_index'] = 1 - np.exp(-0.05 * wet14)

    df['sw_monsoon_flag'] = (df['wind_v'] > 0).astype(int)
    df['ne_monsoon_flag'] = (df['wind_v'] < 0).astype(int)
    df['wind_monsoon_index']     = 0.7*df['wind_u'] + 0.7*df['wind_v']
    df['wind_monsoon_weighted']  = df['wind_monsoon_index'] * df.get('doy_sin',1.0)

    sea_sector = (df['winddir'] >= 90) & (df['winddir'] <= 210)
    df['is_onshore_flow']  = sea_sector.astype(int)
    df['is_offshore_flow'] = (~sea_sector).astype(int)

    cloud7 = df['cloudcover'].rolling(7,1).mean()
    df['cloudy_spell_flag'] = (cloud7 > 0.7).astype(int)
    df['clear_spell_flag']  = (cloud7 < 0.3).astype(int)

    df['precip_intensity'] = df['precip'] / (df['precipcover'] + 1e-3)
    df.loc[df['precip']==0, 'precip_intensity'] = 0
    med_intensity = df['precip_intensity'].median()
    med_gust      = df['windgust'].median()
    df['is_convective_rain'] = ((df['precip_intensity'] > med_intensity) &
                                (df['windgust'] > med_gust)).astype(int)
    df['is_stratiform_rain'] = ((df['precip'] > 0) &
                                (df['is_convective_rain']==0)).astype(int)
    df['convective_yesterday'] = df['is_convective_rain'].shift(1).fillna(0)

    hum_thr = df['humidity'].median()
    rad_thr = df['solarradiation'].median()
    df['regime_hot_humid']    = ((df['humidity'] > hum_thr) & (df['solarradiation'] > rad_thr)).astype(int)
    df['regime_hot_drier']    = ((df['humidity'] < hum_thr) & (df['solarradiation'] > rad_thr)).astype(int)
    df['regime_cloudy_humid'] = ((df['humidity'] > hum_thr) & (df['solarradiation'] < rad_thr)).astype(int)

    u, v = df['wind_u'], df['wind_v']
    df['wind_dir_consistency_w7']  = np.sqrt(u.rolling(7,1).mean()**2 + v.rolling(7,1).mean()**2) / (
                                    df['windspeed'].rolling(7,1).mean() + 1e-3)
    df['wind_dir_consistency_w28'] = np.sqrt(u.rolling(28,1).mean()**2 + v.rolling(28,1).mean()**2) / (
                                    df['windspeed'].rolling(28,1).mean() + 1e-3)

    df['wind_pressure_coupling'] = df['windspeed'] * (
        df['sealevelpressure'].rolling(3,1).mean() - df['sealevelpressure'].rolling(7,1).mean()
    )

    df['humid_radiation_balance'] = df['humidity'].rolling(3,1).mean() * df['solarradiation'].rolling(3,1).mean()
    df['humid_radiation_ratio']   = df['humidity'] / (df['solarradiation'] + 1e-3)

    if 'rain_risk_combo_w7' in df.columns:
        df['rain_risk_trend_3d'] = df['rain_risk_combo_w7'] - df['rain_risk_combo_w7'].shift(3)
        df['rain_risk_yesterday'] = df['rain_risk_combo_w7'].shift(1)
    else:
        df['rain_risk_trend_3d'] = 0
        df['rain_risk_yesterday'] = 0

    pairs = [('humidity','solarradiation'),
            ('wind_u','wind_v'),
            ('humidity','precip'),
            ('sealevelpressure','windspeed')]
    for a,b in pairs:
        df[f'{a}_{b}_cov7'] = df[a].rolling(7,1).cov(df[b])

    df['mean_wind_dir_w7']  = np.rad2deg(np.arctan2(df['wind_v_mean_w7'],  df['wind_u_mean_w7']))  % 360
    df['mean_wind_dir_w28'] = np.rad2deg(np.arctan2(df['wind_v_mean_w28'], df['wind_u_mean_w28'])) % 360
    df['mean_wind_dir_w7_rad']  = np.deg2rad(df['mean_wind_dir_w7'])
    df['mean_wind_dir_w28_rad'] = np.deg2rad(df['mean_wind_dir_w28'])
    df['mean_wind_dir_w7_sin']  = np.sin(df['mean_wind_dir_w7_rad'])
    df['mean_wind_dir_w7_cos']  = np.cos(df['mean_wind_dir_w7_rad'])
    df['mean_wind_dir_w28_sin'] = np.sin(df['mean_wind_dir_w28_rad'])
    df['mean_wind_dir_w28_cos'] = np.cos(df['mean_wind_dir_w28_rad'])

    bins = np.arange(-22.5, 382.5, 45)
    labels = ['N','NE','E','SE','S','SW','W','NW']
    df['wind_sector_w7_cat']  = pd.cut(df['mean_wind_dir_w7'],  bins=bins, labels=labels, ordered=False)
    df['wind_sector_w28_cat'] = pd.cut(df['mean_wind_dir_w28'], bins=bins, labels=labels, ordered=False)

    df['precip_efficiency'] = df['precip'].rolling(3,1).sum() / (df['cloudcover'].rolling(3,1).mean() + 1e-3)
    df['wind_energy']      = 0.5 * (df['windspeed'] ** 2)
    df['wind_energy_anom'] = df['wind_energy'] - df['wind_energy'].rolling(56,1).mean()
    df['convective_potential_index'] = df['humidity'].rolling(7,1).mean() * df['solarradiation'].rolling(7,1).mean()
    

    # --- EWMA features ---
    ewma_features = ['humidity', 'dew', 'solarradiation', 'sealevelpressure', 'windspeed', 'precip']
    ewma_configs = {'3d': 0.5, '7d': 0.3, '14d': 0.15}
    
    for col in ewma_features:
        for tag, alpha in ewma_configs.items():
            df[f'{col}_ewma_{tag}'] = df[col].ewm(alpha=alpha, adjust=False).mean()

    cat_cols = [c for c in df.columns if c.endswith('_cat')]
    df = df.drop(columns=cat_cols, errors='ignore')
    
    df = df.replace(['', ' ', 'NA', 'N/A', 'NaN', 'nan', '#DIV/0!', 'inf', '-inf'], np.nan)


    date_cols_to_preserve = ['datetime'] 
    for col in df.columns:
        if col not in date_cols_to_preserve:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df_processed = df.iloc[max(windows):].reset_index(drop=True)
    
    return df_processed