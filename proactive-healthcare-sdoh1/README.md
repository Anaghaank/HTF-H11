I'll update the README.md file with more detailed instructions for MongoDB setup and model training. Here's the updated version:

```markdown:c:\Users\Palguna\Desktop\proactive_healthcare1\proactive_healthcare\proactive-healthcare-sdoh1\README.md
# Proactive Healthcare Dashboard

A Flask-based web application for healthcare data management and prediction visualization. This application allows users to upload healthcare data (either individually or via CSV), store it in MongoDB, and visualize hospital admission predictions based on social determinants of health (SDOH).

## Project Structure

```
proactive-healthcare-sdoh1/
├── app/
│   ├── static/
│   │   ├── images/
│   │   │   └── doctor.jpg
│   │   └── style.css
│   ├── templates/
│   │   ├── index.html
│   │   ├── index_upload.html
│   │   ├── index_view_predictions.html
│   │   ├── preview_form.html
│   │   ├── result.html
│   │   ├── upload.html
│   │   ├── upload_csv.html
│   │   └── upload_success.html
│   ├── app.py
│   ├── dashboard.py
│   ├── model.py
│   └── utils.py
├── data/
│   ├── sdoh_karnataka_dataset.csv
│   └── uploads/
└── model/
    └── rf_model.joblib
```

## Prerequisites

- Python 3.6 or higher
- MongoDB installed and running on localhost:27017
- Required Python packages (see installation section)

## Installation

1. Clone the repository or download the source code

2. Install the required Python packages:

```bash
pip install flask pymongo pandas scikit-learn matplotlib joblib
```

3. Set up MongoDB:
   - Download and install MongoDB Community Edition from [MongoDB website](https://www.mongodb.com/try/download/community)
   - Start the MongoDB service:
     ```bash
     net start MongoDB
     ```
   - Verify MongoDB is running by connecting to it:
     ```bash
     mongosh
     ```
   - Create the healthcare database:
     ```
     use healthcare_db
     ```
   - Create a collection for records:
     ```
     db.createCollection("records")
     ```
   - Exit the MongoDB shell:
     ```
     exit
     ```

## Running the Application

1. Navigate to the project directory:

```bash
cd c:\Users\Palguna\Desktop\proactive_healthcare1\proactive_healthcare\proactive-healthcare-sdoh1
```

2. Train the machine learning model (optional, will be done automatically if not done):

```bash
python app/model.py
```

3. Run the Flask application:

```bash
python app/app.py
```

4. Open your web browser and go to: http://127.0.0.1:5000/

## Features

- **Upload Single Record**: Add individual healthcare records through a form
- **Upload CSV Data**: Bulk upload healthcare records via CSV file
- **View Predictions**: Visualize hospital admissions data by pincode
- **Data Preview**: Review and edit CSV data before submitting to the database
- **MongoDB Integration**: All data is stored in a MongoDB database for persistence
- **Machine Learning Model**: Predicts hospital admissions based on social determinants of health

## Data Format

The application expects the following data fields:

- Pincode: Postal code (integer)
- Date: Date in YYYY-MM-DD format
- UnemploymentRate: Percentage (float)
- SchoolAttendance: Percentage (float)
- EvictionRecords: Count (integer)
- IncomeVariability: Index value (float)
- SeasonalIllnessIndex: Index value (float)
- HospitalAdmissions: Count (float)

## Model Training

The application uses a Random Forest Regressor model to predict hospital admissions based on social determinants of health. The model is trained on the dataset provided in `data/sdoh_karnataka_dataset.csv`.

To manually train the model:
1. Run the model.py script:
   ```bash
   python app/model.py
   ```
2. The trained model will be saved to `data/rf_model.joblib`
3. Model performance metrics will be displayed in the console

The model will also be automatically trained if it doesn't exist when the application needs to make predictions.

## Workflow

1. **Upload Data**: 
   - Use the single record form or CSV upload to add data
   - For CSV uploads, you'll see a preview of the data before it's submitted
   - The model will automatically generate predictions for the uploaded data

2. **View Predictions**:
   - Navigate to the "View Predictions" page to see hospital admissions by pincode
   - The chart visualizes the total admissions for each pincode

3. **Database Management**:
   - All data is stored in MongoDB for persistence
   - You can view and manage the data using MongoDB tools

## Troubleshooting

- If you encounter connection issues with MongoDB:
  - Ensure the MongoDB service is running: `net start MongoDB`
  - Check MongoDB connection string in utils.py
  - Verify you can connect manually: `mongosh`

- For any file upload issues:
  - Check that the data directory has proper write permissions
  - Ensure the CSV file format matches the expected fields

- If the model fails to load:
  - Run the model training script manually: `python app/model.py`
  - Ensure the model file is saved to `data/rf_model.joblib`
  - Check for any error messages during model training
```