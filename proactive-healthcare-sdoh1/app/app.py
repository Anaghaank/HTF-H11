from flask import Flask, render_template, request, redirect, url_for
from pymongo import MongoClient
from datetime import datetime

app = Flask(__name__)

# MongoDB Configuration
client = MongoClient("mongodb://localhost:27017/")
db = client["healthcare_py"]
collection = db["records"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_form():
    if request.method == 'POST':
        data = {
            "Pincode": int(request.form['Pincode']),
            "Date": datetime.strptime(request.form['Date'], "%Y-%m-%d"),
            "UnemploymentRate": float(request.form['UnemploymentRate']),
            "SchoolAttendance": float(request.form['SchoolAttendance']),
            "EvictionRecords": int(request.form['EvictionRecords']),
            "IncomeVariability": float(request.form['IncomeVariability']),
            "SeasonalIllnessIndex": float(request.form['SeasonalIllnessIndex']),
            "HospitalAdmissions": float(request.form['HospitalAdmissions']),
        }

        # Insert to MongoDB
        collection.insert_one(data)
        return render_template('upload_success.html')

    return render_template('upload.html')

@app.route('/view-prediction')
def view_prediction_prompt():
    records = list(collection.find({}, {"_id": 0}))
    return render_template('index_view_predictions.html', predictions=records, count=len(records))

if __name__ == '__main__':
    app.run(debug=True)
