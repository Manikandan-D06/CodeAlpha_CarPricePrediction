import joblib

def load_model(path="models/car_price_rf_model.pkl"):
    """Load a saved model from file."""
    return joblib.load(path)