from flask import Flask, render_template, request, redirect, url_for
from datetime import datetime
import os
import json
from werkzeug.utils import secure_filename
from utils import parse_csv, get_db, insert_records_to_mongo
# Import the prediction function
from model import predict_from_csv_data, get_model, train_model
import logging

app = Flask(__name__)

# Add tojson filter to Jinja2 environment
@app.template_filter('tojson')
def to_json_filter(value):
    return json.dumps(value)

# MongoDB is configured in utils.py

def init_db():
    # MongoDB collections are created automatically when data is inserted
    # No need to create tables/collections in advance
    pass

# File Upload Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_form():
    if request.method == 'POST':
        try:
            db = get_db()
            
            # Create the record from form data
            record = {
                'Pincode': int(request.form['Pincode']),
                'Date': request.form['Date'],
                'UnemploymentRate': float(request.form['UnemploymentRate']),
                'SchoolAttendance': float(request.form['SchoolAttendance']),
                'EvictionRecords': int(request.form['EvictionRecords']),
                'IncomeVariability': float(request.form['IncomeVariability']),
                'SeasonalIllnessIndex': float(request.form['SeasonalIllnessIndex']),
                'HospitalAdmissions': float(request.form['HospitalAdmissions']),
                'PredictedHospitalAdmissions': None
            }
            
            # Generate prediction for this record
            try:
                # Get the model
                model = get_model()
                
                # Prepare data for prediction
                features = ['Pincode', 'UnemploymentRate', 'SchoolAttendance', 
                            'EvictionRecords', 'IncomeVariability', 'SeasonalIllnessIndex']
                X = [[float(record[feature]) for feature in features]]
                
                # Make prediction
                prediction = model.predict(X)[0]
                
                # Add prediction to record
                record['PredictedHospitalAdmissions'] = float(prediction)
                print(f"Generated prediction for single record: {prediction}")
            except Exception as e:
                print(f"Error generating prediction: {str(e)}")
                logging.error(f"Prediction error: {str(e)}")
            
            # Insert record with prediction to MongoDB
            try:
                result = db.records.insert_one(record)
                print(f"Inserted record with ID: {result.inserted_id}, Record: {record}")
                
                # Redirect to view-prediction page to show the comparison immediately
                return redirect(url_for('view_prediction_prompt'))
            except Exception as e:
                print(f"MongoDB insertion error: {str(e)}")
                return render_template('error.html', error=f"Database error: {str(e)}")
                
        except Exception as e:
            print(f"Error in upload_form: {str(e)}")
            logging.error(f"Upload form error: {str(e)}")
            return render_template('error.html', error=f"Error: {str(e)}")

    return render_template('upload.html')

@app.route('/view-prediction')
def view_prediction_prompt():
    db = get_db()
    # Ensure we're retrieving all records, including those with predictions
    records = list(db.records.find({}, {'_id': 0}))
    
    # Add debug information
    print(f"Retrieved {len(records)} records from MongoDB for visualization")
    if len(records) > 0:
        print(f"Sample record fields: {list(records[0].keys())}")
    
    return render_template('index_view_predictions.html', predictions=records, count=len(records))

@app.route('/upload-csv', methods=['GET', 'POST'])
def upload_csv():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Parse CSV and preview data
            records = parse_csv(filepath)
            
            # Generate predictions for the uploaded data
            try:
                records_with_predictions = predict_from_csv_data(records)
                return render_template('preview_form.html', records=records_with_predictions)
            except Exception as e:
                logging.error(f"Prediction error: {str(e)}")
                return render_template('error.html', error=str(e))
    
    return render_template('upload_csv.html')

@app.route('/train-model', methods=['GET'])
def train_model_route():
    """Route to manually trigger model training"""
    try:
        _, metrics = train_model()
        return render_template('model_trained.html', metrics=metrics)
    except Exception as e:
        logging.error(f"Model training error: {str(e)}")
        return render_template('error.html', error=str(e))

@app.route('/submit-csv', methods=['POST'])
def submit_csv():
    # Process the form data from preview_form
    form_data = request.form
    
    # Extract headers
    headers = []
    for key in form_data:
        if key.startswith('header_'):
            headers.append(form_data[key])
    
    # Extract records
    records = []
    
    # Determine how many rows we have
    max_row_index = -1
    for key in form_data:
        if key.startswith('field_'):
            parts = key.split('_')
            if len(parts) >= 2:
                row_index = int(parts[1])
                max_row_index = max(max_row_index, row_index)
    
    # Create records
    for row_index in range(max_row_index + 1):
        record = {}
        for col_index, header in enumerate(headers):
            field_name = f"field_{row_index}_{col_index}"
            if field_name in form_data:
                value = form_data[field_name]
                if value:  # Only add non-empty values
                    record[header] = value
        
        if record:  # Only add non-empty records
            # Ensure all required fields have proper types
            try:
                processed_record = {
                    'Pincode': int(float(record.get('Pincode', 0))),
                    'Date': record.get('Date', ''),
                    'UnemploymentRate': float(record.get('UnemploymentRate', 0)),
                    'SchoolAttendance': float(record.get('SchoolAttendance', 0)),
                    'EvictionRecords': int(float(record.get('EvictionRecords', 0))),
                    'IncomeVariability': float(record.get('IncomeVariability', 0)),
                    'SeasonalIllnessIndex': float(record.get('SeasonalIllnessIndex', 0)),
                }
                
                # Include HospitalAdmissions if it exists
                if 'HospitalAdmissions' in record and record['HospitalAdmissions']:
                    processed_record['HospitalAdmissions'] = float(record.get('HospitalAdmissions', 0))
                
                # Include PredictedHospitalAdmissions if it exists
                if 'PredictedHospitalAdmissions' in record and record['PredictedHospitalAdmissions']:
                    processed_record['PredictedHospitalAdmissions'] = float(record.get('PredictedHospitalAdmissions', 0))
                
                records.append(processed_record)
            except (ValueError, TypeError) as e:
                print(f"Error processing record {record}: {str(e)}")
                continue
    
    # Insert records to MongoDB
    if records:
        try:
            print(f"Processing {len(records)} records from form submission")
            db = get_db()
            result = db.records.insert_many(records)
            print(f"Inserted {len(result.inserted_ids)} records to MongoDB")
            return render_template('upload_success.html')
        except Exception as e:
            print(f"Error in submit_csv: {str(e)}")
            return render_template('error.html', error=f"Error: {str(e)}")
    else:
        return render_template('error.html', error="No valid records found in the form submission")

if __name__ == '__main__':
    # Initialize the database
    init_db()
    app.run(debug=True)
