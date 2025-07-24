import pandas as pd
import requests  # Send HTTP POST requests to Flask API
from datetime import datetime  # timestamp generation in filenames
import os  # existence check, writing, etc
import traceback  # error messages
import time  # Measure time for performance logging

X_all = pd.read_csv("D:/dissertation/fraudTrain_v12.csv") # Load processed feature data (for API input) prediction
batch_size = 50
X_all = X_all.iloc[:batch_size]  # Limit to first 50 rows
total_rows = len(X_all)  # Total rows to process

# Load original raw data for mapping (for final display info)
mapping_df = pd.read_csv(r"C:\Users\User\Desktop\fraud_detection_project_cloud\fraudTrain.csv")  # Raw data for lookup
mapping_fields = ['trans_num', 'cc_num', 'merchant', 'amt', 'trans_date_trans_time']  # Fields to map back

now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"results_batch_{now}.csv"  # CSV file to write batch predictions
debug_log = "debug_batch.log"  # Log file
write_header = True  # Whether to write CSV header

# Resume check
written_rows = 0  # Start from 0 unless file exists
if os.path.exists(log_filename):  # Check if output file already exists
    written_rows = sum(1 for _ in open(log_filename)) - 1  # Count already written rows
    print(f"Resuming from row {written_rows}")  # Print resume info
    with open(debug_log, "a") as log:  # Log to debug file
        log.write(f"[{datetime.now()}] Resuming from row {written_rows}\n")
    write_header = False  # If resuming, don't write header again

# Main batch loop
for start in range(written_rows, total_rows, batch_size):  # Iterate by batch
    batch_start_time = time.time()  # Record time per batch
    end = min(start + batch_size, total_rows)  # End index of batch
    batch_data = X_all.iloc[start:end].to_dict(orient="records")  # Convert batch to list of dicts

    print(f"\n Batch {start}–{end}: Starting — reading {end - start} rows")  # Batch info
    print(" Step 1: Sending to Flask API → /predict_batch")  # Send to API

    try:
        res = requests.post("http://127.0.0.1:5000/predict_batch", json={"features_list": batch_data})  # Call local API
        res.raise_for_status()  # Raise error if HTTP status not 200
        result_batch = res.json()["results"]  # Extract results from response
        print(" Step 2: Received predictions from API")  # Success
    except Exception as e:
        print(f" Error at rows {start} to {end}: {e}")  # API call failed
        continue  # Skip to next batch

    print(" Step 3: Processing predictions and mapping results")  # Map and format

    enriched_results = []  # Store enriched prediction rows
    fraud_count = 0  # Count of fraud rows

    for i, result in enumerate(result_batch):  # Iterate over results
        if result["prediction"] == 1:  # If fraud
            fraud_count += 1  # Count fraud
            mapping_index = start + i  # Get index in original
            timestamp = mapping_df.at[mapping_index, 'trans_date_trans_time']  # Lookup timestamp
            cc_num = str(mapping_df.at[mapping_index, 'cc_num'])  # Lookup card number
            merchant = mapping_df.at[mapping_index, 'merchant']  # Lookup merchant
            amt = mapping_df.at[mapping_index, 'amt']  # Lookup amount
            print(f"🚨 Row {mapping_index + 1}: Fraud @ {timestamp} | {cc_num[:4]}-{cc_num[4:6]}XX-XXXX-{cc_num[-4:]} @ {merchant} | ${amt:.2f}")  # Print masked info

        # Enrich every result (not enrich fraud ok )
        mapping_index = start + i  # Get index
        for field in mapping_fields:  # Add back display fields
            result[field] = mapping_df.at[mapping_index, field]  # Enrich field
        enriched_results.append(result)  # Add to output

    print(f" Batch {start}–{end}: Fraudulent rows: {fraud_count} / {end - start}")  # Print fraud summary

try:
    df_result = pd.DataFrame(enriched_results)  # Convert to DataFrame

    df_result["cc_num"] = df_result["cc_num"].apply(lambda x: str(int(float(x))).zfill(16))  # Format credit card to 16 digits

    desired_order = [  # Desired output column order
        "trans_date_trans_time", "trans_num", "cc_num", "amt", "merchant",
        "fraud_probability", "prediction"
    ]
    extra_cols = [col for col in df_result.columns if col not in desired_order]  # Preserve unexpected fields
    df_result = df_result[desired_order + extra_cols]  # Reorder columns

    df_result.to_csv(log_filename, mode="a", index=False, header=write_header, encoding="utf-8-sig")  # Write CSV file
    write_header = False  # Turn off header after first write

    elapsed_min = round((time.time() - batch_start_time) / 60, 2)  # Time used
    print(f" Batch {start}–{end}: preprocess() → scaler.transform() → predict_proba() → return → write → DONE in {elapsed_min} minutes")  # Done info

    with open(debug_log, "a") as log:  # Append debug log
        log.write(f"[{datetime.now()}] Wrote rows {start} to {end} in {elapsed_min} minutes\n")

except Exception as e:
    error_trace = traceback.format_exc()  # Get full stack trace
    with open(debug_log, "a") as log:  # Write error to debug log
        log.write(f"[{datetime.now()}] Error at rows {start} to {end}\n")
        log.write(error_trace + "\n")
    print(f" Error while writing at rows {start} to {end}: {e}")  # Print error