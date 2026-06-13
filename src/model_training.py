from sklearn.ensemble import RandomForestRegressor
import joblib

def train_rf(X_train, y_train, save_path="models/car_price_rf_model.pkl"):
    """Train Random Forest and save the model."""
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, save_path)
    return rf