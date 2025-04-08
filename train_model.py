import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset
df = pd.read_csv("sdoh_karnataka_dataset.csv")
df.columns = df.columns.str.strip()  # Clean column names
df = df.dropna(subset=["HospitalAdmissions"])
df['Date'] = pd.to_datetime(df['Date'])

# Prepare features and target
X = df.drop(columns=["Date", "HospitalAdmissions"])
y = df["HospitalAdmissions"]

# Train-test split and model
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "hospital_admission_model.pkl")
print("✅ Model saved as hospital_admission_model.pkl")
