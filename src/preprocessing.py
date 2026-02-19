import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    return pd.read_csv(filepath)

def extract_primary_genre(genre_str):
    if pd.isna(genre_str):
        return np.nan
    return str(genre_str).split(',')[0].strip()

def clean_data(df):
    df = df.copy()
    df['primary_genre'] = df['genre'].apply(extract_primary_genre)
    
    numeric_cols = ['popularity', 'danceability', 'energy', 'tempo']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    subset = df[['primary_genre', 'popularity', 'danceability', 'energy', 'tempo']]
    df_clean = subset.dropna()
    
    return df_clean

def scale_features(df, feature_cols):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[feature_cols])
    return pd.DataFrame(scaled, columns=feature_cols, index=df.index), scaler
