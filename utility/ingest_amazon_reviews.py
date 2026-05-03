import os
import subprocess
import zipfile
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm
import math

# Load environment vars from .env file
load_dotenv()

def main():
    # 1. Download zip into the utility folder
    downloads_dir = Path(__file__).parent
    zip_path = downloads_dir / "amazon-product-reviews.zip"

    print(f"Downloading dataset to {zip_path}...")
    # Download using curl
    curl_cmd = [
        "curl", "-L", "-o", str(zip_path), 
        "https://www.kaggle.com/api/v1/datasets/download/arhamrumi/amazon-product-reviews"
    ]
    subprocess.run(curl_cmd, check=True)
    
    # 2. Check if the downloaded file is actually a zip file 
    if not zipfile.is_zipfile(zip_path):
        raise ValueError("The downloaded file is not a valid zip file. Please ensure you are authenticated with Kaggle or have proper access (e.g. valid Kaggle cookies/config).")

    # 3. Unzip file
    extract_dir = downloads_dir / "amazon-product-reviews-extract"
    print(f"Extracting to {extract_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
        
    # Find the csv file inside the extracted folder
    csv_files = list(extract_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV file found in the downloaded zip.")
    csv_path = csv_files[0]
    
    # 4. Read dataset
    print(f"Reading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    # 5. Rename "Text" to "reviewText" for consistency
    if 'Text' in df.columns:
        df.rename(columns={'Text': 'reviewText'}, inplace=True)

    print("Parsing and typing columns...")
    # 6. Parsing other fields
    if 'Id' in df.columns:
        df['Id'] = pd.to_numeric(df['Id'], errors='coerce').astype('Int64')
    if 'ProductId' in df.columns:
        df['ProductId'] = df['ProductId'].astype(str)
    if 'UserId' in df.columns:
        df['UserId'] = df['UserId'].astype(str)
    if 'ProfileName' in df.columns:
        df['ProfileName'] = df['ProfileName'].astype(str)
    if 'HelpfulnessNumerator' in df.columns:
        df['HelpfulnessNumerator'] = pd.to_numeric(df['HelpfulnessNumerator'], errors='coerce').astype('Int64')
    if 'HelpfulnessDenominator' in df.columns:
        df['HelpfulnessDenominator'] = pd.to_numeric(df['HelpfulnessDenominator'], errors='coerce').astype('Int64')
    if 'Score' in df.columns:
        df['Score'] = pd.to_numeric(df['Score'], errors='coerce').astype(float)
    if 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'], unit='s', errors='coerce')
    if 'Summary' in df.columns:
        df['Summary'] = df['Summary'].astype(str)
    if 'reviewText' in df.columns:
        df['reviewText'] = df['reviewText'].astype(str)
    
    # 7. Remove duplicates (same user + product), keeping the most recent review
    required_cols = ['UserId', 'ProductId']
    if all(col in df.columns for col in required_cols) and 'Time' in df.columns:
        initial_count = len(df)
        # Sort descending by Time so the most recent review comes first
        df.sort_values(by='Time', ascending=False, inplace=True)
        # Drop duplicates, keeping only the first (most recent) occurrence
        df.drop_duplicates(subset=required_cols, keep='first', inplace=True)
        final_count = len(df)
        print(f"Removed {initial_count - final_count} duplicates (kept most recent review per user/product).")
    else:
        print("Warning: Could not find 'UserId', 'ProductId' or 'Time' columns to deduplicate.")
    # 8. Save cleaned CSV locally
    clean_csv_path = downloads_dir / "amazon-product-reviews-clean.csv"
    print(f"Saving cleaned CSV locally: {clean_csv_path}")
    df.to_csv(clean_csv_path, index=False)
        
    # 9. Upload to PostgreSQL using the centralized db module
    from utility.db import engine

    table_name = "amazon_reviews"
    print(f"Writing data to table '{table_name}'...")

    chunksize = 10000
    total_chunks = math.ceil(len(df) / chunksize)

    # Upload in chunks so we can show a progress bar
    for i in tqdm(range(total_chunks), desc="Uploading to Postgres", unit="chunk"):
        chunk = df.iloc[i * chunksize : (i + 1) * chunksize]
        # Replace the table on the first chunk, then append subsequent chunks
        chunk_action = 'replace' if i == 0 else 'append'
        chunk.to_sql(table_name, engine, if_exists=chunk_action, index=False)

    print("Data successfully loaded into PostgreSQL!")

if __name__ == "__main__":
    main()
