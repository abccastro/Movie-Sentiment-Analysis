import pandas as pd

def split_and_save_csv(input_file, output_prefix, chunk_size):
    # Read the CSV file
    df = pd.read_csv(input_file)

    # Split the DataFrame into chunks
    chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

    # Save each chunk to a separate CSV file
    for i, chunk in enumerate(chunks):
        output_file = f"{output_prefix}_{i + 1}.csv"
        chunk.to_csv(output_file, index=False)
        print(f"Saved {len(chunk)} records to {output_file}")

# Example usage
input_csv = r'C:\Users\aurad\OneDrive - Lambton College\Documents\Lambton College\Class\Repository\Projects\Movie-Sentiment-Analysis\dataset\reviews.csv'  # Replace with your actual CSV file
output_prefix = r'C:\Users\aurad\OneDrive - Lambton College\Documents\Lambton College\Class\Repository\Projects\Movie-Sentiment-Analysis\dataset\reviews_splitt'      # Prefix for output files
chunk_size = 100000                     # Number of records per chunk

split_and_save_csv(input_csv, output_prefix, chunk_size)
