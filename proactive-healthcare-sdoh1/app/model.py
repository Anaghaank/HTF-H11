import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Path for saving the trained model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'rf_model.joblib')

def train_model():
    """Train the RandomForest model on the dataset and save it for later use"""
    try:
        # Step 1: Load and sort the data
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'sdoh_karnataka_dataset.csv')
        df = pd.read_csv(data_path)
        df.columns = df.columns.str.strip()  # Fix column name spacing
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')

        # Step 2: Define features and target
        X = df.drop(columns=['Date', 'HospitalAdmissions'])  # Features
        y = df['HospitalAdmissions']  # Target

        # Step 3: Train/Validation/Test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

        # Step 4: Train model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        # Step 5: Accuracy check
        val_preds = rf_model.predict(X_val)
        test_preds = rf_model.predict(X_test)

        val_mse = mean_squared_error(y_val, val_preds)
        val_r2 = r2_score(y_val, val_preds)
        test_mse = mean_squared_error(y_test, test_preds)
        test_r2 = r2_score(y_test, test_preds)

        logging.info("🔹 Validation Set Results")
        logging.info(f"MSE: {val_mse}")
        logging.info(f"R² Score: {val_r2}")

        logging.info("🔹 Test Set Results")
        logging.info(f"MSE: {test_mse}")
        logging.info(f"R² Score: {test_r2}")

        # Save the model
        joblib.dump(rf_model, MODEL_PATH)
        logging.info(f"Model saved to {MODEL_PATH}")
        
        return rf_model, {
            'val_mse': val_mse,
            'val_r2': val_r2,
            'test_mse': test_mse,
            'test_r2': test_r2
        }
    except Exception as e:
        logging.error(f"Error training model: {str(e)}")
        raise

def get_model():
    """
    Load the trained model from disk or train a new one if not available
    """
    import os
    import pickle
    import logging
    from sklearn.ensemble import RandomForestRegressor
    
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'hospital_admissions_model.pkl')
    
    try:
        # Try to load the model
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print("Loaded existing model from disk")
            return model
        else:
            # Train a new model if one doesn't exist
            print("No existing model found, training a new one")
            model, _ = train_model()
            return model
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        # Fallback to a simple model if loading fails
        print("Error loading model, creating a simple fallback model")
        return RandomForestRegressor(n_estimators=10)

def predict_from_csv_data(records):
    """
    Make predictions using the trained model on uploaded CSV data
    
    Args:
        records: List of dictionaries containing the uploaded data
        
    Returns:
        List of dictionaries with original data and predictions
    """
    try:
        # Get the trained model
        model = get_model()
        
        # Convert records to DataFrame
        df = pd.DataFrame(records)
        
        # Ensure all required columns are present
        required_columns = ['Pincode', 'UnemploymentRate', 'SchoolAttendance', 
                            'EvictionRecords', 'IncomeVariability', 'SeasonalIllnessIndex']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Required columns missing: {', '.join(missing_columns)}")
        
        # Prepare features for prediction
        X_predict = df[required_columns]
        
        # Make predictions
        predictions = model.predict(X_predict)
        
        # Add predictions to the original records
        for i, record in enumerate(records):
            record['PredictedHospitalAdmissions'] = round(float(predictions[i]), 2)
        
        logging.info(f"Generated predictions for {len(records)} records")
        return records
    except Exception as e:
        logging.error(f"Error making predictions: {str(e)}")
        raise