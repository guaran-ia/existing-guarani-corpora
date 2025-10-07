import os

from downloader import download_corpora
from processor import process_parquet_files, process_corpora


if __name__ == '__main__':
    project_dir = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
    download_corpora(project_dir)
    corpora_dir = os.path.join(project_dir, 'data', 'raw')
    process_parquet_files(corpora_dir)
    process_corpora(corpora_dir)