from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

class PopularityPredictor:
    def __init__(self, model_type='linear'):
        self.model_type = model_type
        self.model = None
        self.feature_cols = None
        
    def select_model(self):
        if self.model_type == 'linear':
            return LinearRegression()
        elif self.model_type == 'tree':
            return DecisionTreeRegressor(random_state=42, max_depth=10)
        elif self.model_type == 'forest':
            return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            return LinearRegression()
    
    def train(self, X, y, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        self.model = self.select_model()
        self.model.fit(X_train, y_train)
        self.feature_cols = X.columns.tolist()
        
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        return {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse
        }
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)
