import pandas as pd
import numpy as np

class MusicRecommender:
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
    
    def recommend_by_genre(self, song_idx, n_recommendations=5):
        song = self.df.loc[song_idx]
        same_genre = self.df[
            (self.df['primary_genre'] == song['primary_genre']) & 
            (self.df.index != song_idx)
        ]
        recommendations = same_genre.nlargest(n_recommendations, 'popularity')
        return recommendations
    
    def recommend_by_similarity(self, song_idx, n_recommendations=5):
        song = self.df.loc[song_idx]
        feature_cols = ['danceability', 'energy', 'tempo', 'popularity']
        
        diff = self.df[feature_cols].values - song[feature_cols].values
        distances = np.asarray(((diff**2).sum(axis=1))**0.5, dtype=float)
        
        self.df['similarity_dist'] = distances
        recommendations = self.df[self.df.index != song_idx].nsmallest(
            n_recommendations, 'similarity_dist'
        )
        recommendations = recommendations.drop('similarity_dist', axis=1)
        self.df = self.df.drop('similarity_dist', axis=1)
        
        return recommendations
    
    def recommend_by_audio_features(self, song_idx, n_recommendations=5, weights=None):
        song = self.df.loc[song_idx]
        feature_cols = ['danceability', 'energy', 'tempo']
        
        if weights is None:
            weights = {col: 1.0 for col in feature_cols}
        
        weighted_dist = np.zeros(len(self.df))
        for col in feature_cols:
            weighted_dist += weights.get(col, 1.0) * (self.df[col].values - song[col])**2
        
        self.df['weighted_dist'] = weighted_dist**0.5
        recommendations = self.df[self.df.index != song_idx].nsmallest(
            n_recommendations, 'weighted_dist'
        )
        recommendations = recommendations.drop('weighted_dist', axis=1)
        self.df = self.df.drop('weighted_dist', axis=1)
        
        return recommendations
