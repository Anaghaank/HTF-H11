from pymongo import MongoClient
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_db():
    try:
        # Connect to MongoDB with a timeout to ensure we don't hang indefinitely
        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
        # Verify connection is working by sending a ping
        client.admin.command('ping')
        print("Successfully connected to MongoDB")
        db = client["healthcare_db"]
        return db
    except Exception as e:
        print(f"Failed to connect to MongoDB: {str(e)}")
        raise

def insert_records_to_mongo(data):
    try:
        if not data:
            print("No data to insert into MongoDB")
            return False
        
        # Clean and validate the data before insertion
        cleaned_data = []
        for record in data:
            # Skip empty records
            if not record or all(not val for val in record.values()):
                continue
                
            # Ensure all required fields are present and have proper types
            try:
                cleaned_record = {
                    'Pincode': int(float(record.get('Pincode', 0))),
                    'Date': record.get('Date', ''),
                    'UnemploymentRate': float(record.get('UnemploymentRate', 0)),
                    'SchoolAttendance': float(record.get('SchoolAttendance', 0)),
                    'EvictionRecords': int(float(record.get('EvictionRecords', 0))),
                    'IncomeVariability': float(record.get('IncomeVariability', 0)),
                    'SeasonalIllnessIndex': float(record.get('SeasonalIllnessIndex', 0)),
                    'HospitalAdmissions': float(record.get('HospitalAdmissions', 0))
                }
                
                # Include PredictedHospitalAdmissions if it exists
                if 'PredictedHospitalAdmissions' in record:
                    cleaned_record['PredictedHospitalAdmissions'] = float(record.get('PredictedHospitalAdmissions', 0))
                
                cleaned_data.append(cleaned_record)
            except (ValueError, TypeError) as e:
                print(f"Error processing record {record}: {str(e)}")
                continue
        
        if not cleaned_data:
            print("No valid records to insert after cleaning")
            return False
            
        db = get_db()
        print(f"Attempting to insert {len(cleaned_data)} records into MongoDB")
        result = db.records.insert_many(cleaned_data)
        print(f"Successfully inserted {len(result.inserted_ids)} records into MongoDB")
        return True
    except Exception as e:
        print(f"Error inserting records to MongoDB: {str(e)}")
        raise

def parse_csv(file_path):
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Clean the dataframe - drop rows with all NaN values
        df = df.dropna(how='all')
        
        # Ensure all required columns exist
        required_columns = ['Pincode', 'Date', 'UnemploymentRate', 'SchoolAttendance', 
                           'EvictionRecords', 'IncomeVariability', 'SeasonalIllnessIndex', 
                           'HospitalAdmissions']
        
        for col in required_columns:
            if col not in df.columns:
                print(f"Missing required column: {col}")
                raise ValueError(f"CSV file is missing required column: {col}")
        
        # Convert DataFrame to list of dictionaries
        records = df.to_dict(orient='records')
        print(f"Successfully parsed CSV with {len(records)} records")
        return records
    except Exception as e:
        print(f"Error parsing CSV file: {str(e)}")
        raise
