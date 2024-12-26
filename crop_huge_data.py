import pandas as pd

def crop_csv(input_file, output_file, n_rows=5000):
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Take only the first 5000 rows
    df_cropped = df.head(n_rows)
    
    # Save to new CSV file
    df_cropped.to_csv(output_file, index=False)
    
    print(f"Successfully cropped {input_file} to {n_rows} rows and saved as {output_file}")

input_file = "./data/world_real_estate_data(147k).csv"
output_file = "./data/real_estate.csv"
crop_csv(input_file, output_file)