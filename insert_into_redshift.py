import pandas as pd
import os
import datetime

input_file = "C:/Users/User/Desktop/fraud_detection_project_cloud/results_batch_2025-07-14_14-24-18.csv"
output_dir = "C:/Users/User/Desktop/fraud_detection_project_cloud"
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_csv = os.path.join(output_dir, f"results_for_redshift_{timestamp}.csv")
output_excel = os.path.join(output_dir, f"results_for_redshift_{timestamp}.xlsx")

df = pd.read_csv(input_file).copy()

df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"], errors='coerce') \
                                 .dt.strftime("%Y-%m-%d %H:%M:%S")#  timestamp format（ YYYY-MM-DD HH:MM:SS）

df["cc_num"] = df["cc_num"].astype(str)# string

df["amt"] = df["amt"].astype(float)# avoid，into float then string

if "merchant" in df.columns:# space
    df["merchant"] = df["merchant"].astype(str)

df["trans_date"] = pd.to_datetime(df["trans_date_trans_time"], errors='coerce').dt.date# trans_date only date no time

df_clean = df.dropna()# remove any NaN column avoid copy issue

df_clean.to_csv(output_csv, index=False)# S3 + Excel
df_clean.to_excel(output_excel, index=False)

print(f"CSV saved to: {output_csv}")
print(f"Excel saved to: {output_excel}")

print(" Column names and dtypes:")# double check the sequence
print(df_clean.dtypes)
print("\n🔎 Column order:")
print(df_clean.columns.tolist())

print("\n First 5 rows:")
print(df_clean.head())

print("\n Max string lengths per column:")#length max check
for col in df_clean.columns:
    if df_clean[col].dtype == 'object':
        max_len = df_clean[col].astype(str).map(len).max()
        print(f" - {col}: max length = {max_len}")
