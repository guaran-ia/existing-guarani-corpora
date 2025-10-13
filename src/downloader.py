import json
import os
import requests

from dotenv import load_dotenv


def do_download(url, corpus_path, save_mode='w', auth_token=None):
    headers = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
        response = requests.get(url, stream=True, headers=headers)  # stream=True for large files
    else:
        response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(corpus_path, save_mode) as f:
            if 'b' in save_mode:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            else:
                f.write(response.text)
    else:
        print(f'Download failed: {response.status_code}')


def download_corpora(project_dir):
    load_dotenv(os.path.join(project_dir, 'src', '.env'))
    data_dir = os.path.join(project_dir, 'data')
    raw_data_dir = os.path.join(data_dir, 'raw')
    os.makedirs(raw_data_dir, exist_ok=True)
    with open(os.path.join(data_dir, 'gn_corpora.json'), 'r') as f:
        gn_corpora = json.load(f)
    hf_token = None
    for corpus in gn_corpora:
        name = corpus['name']
        urls = corpus['download_urls']
        format = corpus['format']
        corpus_dir = os.path.join(raw_data_dir, name) 
        os.makedirs(corpus_dir, exist_ok=True)
        for url in urls:
            url_file_name = '_'.join(url.split('/')[-2:])
            corpus_file_path = os.path.join(corpus_dir, url_file_name)
            if not os.path.exists(corpus_file_path):
                print(f'Downloading {name} from {url}')
                if 'huggingface.co' in url:
                    hf_token = os.getenv('HF_ACCESS_TOKEN')
                    if not hf_token:
                        raise Exception('Hugging Face token not found in environment variables.')
                if format in ['parquet', 'bin', 'zip', 'tar.gz', 'bz2']:
                    do_download(url, corpus_file_path, 'wb', hf_token)
                else:
                    do_download(url, corpus_file_path, 'w', hf_token)
    print(f'All downloads completed, files can be found in {raw_data_dir}')
