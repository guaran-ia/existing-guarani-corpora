import os
import pandas as pd


def process_parquet_files(dir_path):
    for root, _, files in os.walk(dir_path):
        for filename in files:
            if filename.endswith('.parquet'):
                file_path = os.path.join(root, filename)
                output_path = os.path.splitext(file_path)[0] + '.csv'
                if not os.path.exists(output_path):
                    print(f'Processing parquet file {file_path}')
                    pd.read_parquet(file_path, engine='pyarrow').to_csv(output_path, index=False)
                    print(f'Processed file saved to {output_path}')
