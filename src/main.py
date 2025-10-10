import os

from downloader import download_corpora
from processor import process_parquet_files, process_corpora, verify_processed_corpora


if __name__ == '__main__':
    project_dir = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
    # download corpora
    download_corpora(project_dir)
    corpora_dir = os.path.join(project_dir, 'data', 'raw')
    # extract parquet file
    process_parquet_files(corpora_dir)
    processed_corpora_dir = os.path.join(project_dir, 'data', 'processed')
    # process raw corpora
    process_corpora(corpora_dir, processed_corpora_dir)
    # verify processed corpora
    verify_processed_corpora(corpora_dir, processed_corpora_dir)
    