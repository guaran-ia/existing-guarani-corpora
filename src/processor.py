import os
import pandas as pd

from utils import create_jsonl, word_count_spacy, word_count_split, \
    identify_language, save_to_json


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


def process_csv_corpus(file_path, output_dir_path, text_col_name, source_col_name, 
                       url_col_name, corpus_name, lang_code='grn', lang_script='Latn'):
    print(f'Processing CSV corpus: {"/".join(file_path.split("/")[-2:])}')
    df = pd.read_csv(file_path)
    data = []
    report_dict = {
        'num_docs': 0,
        'num_words_split': 0,
        'num_words_punct_spacy': 0,
        'num_words_no_punct_spacy': 0,
        'num_chars': 0,
        'avg_words_split': 0,
        'avg_words_punct_spacy': 0,
        'avg_words_no_punct_spacy': 0,
        'avg_chars': 0,
        'avg_language_score': 0
    }
    sum_lang_score = 0
    for _, row in df.iterrows():
        text = row[text_col_name]
        num_words_split = word_count_split(text)
        num_words_punct_spacy = word_count_spacy(text, include_punct=True)
        num_words_no_punct_spacy = word_count_spacy(text, include_punct=False)
        ident_result = identify_language(text.replace('\n', ' ').replace('\r', ' '))
        if ident_result:
            lang_score = ident_result['score']
            lang_code = ident_result['lang']
            lang_score_source = ident_result['source_score']
            lang_ident_method = ident_result['voting_method']
        data.append(
            {
                'text': text,
                'corpus': corpus_name,
                'source': row[source_col_name] if source_col_name in row else 'unknown',
                'url': row[url_col_name] if url_col_name in row else 'unknown',
                'language': lang_code,
                'language_score': lang_score,
                'language_script': lang_script,
                'language_score_source': lang_score_source,
                'language_identification_method': lang_ident_method,
                'num_words_split': num_words_split,
                'num_words_punct_spacy': num_words_punct_spacy,
                'num_words_no_punct_spacy': num_words_no_punct_spacy,
                'num_chars': len(text)
            }
        )
        report_dict['num_docs'] += 1
        report_dict['num_words_split'] += num_words_split
        report_dict['num_words_punct_spacy'] += num_words_punct_spacy
        report_dict['num_words_no_punct_spacy'] += num_words_no_punct_spacy
        report_dict['num_chars'] += len(text)
        sum_lang_score += lang_score
    output_file_path = os.path.join(output_dir_path, f'{corpus_name}.jsonl')
    create_jsonl(data, output_file_path)
    if report_dict['num_docs'] > 0:
        report_dict['avg_words_split'] = report_dict['num_words_split'] / report_dict['num_docs']
        report_dict['avg_words_punct_spacy'] = report_dict['num_words_punct_spacy'] / report_dict['num_docs']
        report_dict['avg_words_no_punct_spacy'] = report_dict['num_words_no_punct_spacy'] / report_dict['num_docs']
        report_dict['avg_chars'] = report_dict['num_chars'] / report_dict['num_docs']
        report_dict['avg_language_score'] = sum_lang_score / report_dict['num_docs']
    else:
        print(f'No documents found in {file_path}, skipping report generation.')
    report_file_path = os.path.join(output_dir_path, f'{corpus_name}_report.json')
    save_to_json(report_dict, report_file_path)


def process_corpora(dir_path):
    for root, _, files in os.walk(dir_path):
        for filename in files:
            if filename.endswith('.csv'):
                file_path = os.path.join(root, filename)
                if 'jojajovai' in root:
                    text_col_name = 'gn'
                    source_col_name = 'source'
                    url_col_name = None
                    corpus_name = 'jojajovai'
                elif 'culturalx' in root:
                    text_col_name = 'text'
                    source_col_name = 'source'
                    url_col_name = 'url'
                    corpus_name = 'culturalx'
                else:
                    raise Exception(f'Unknown corpus in path {root}')
                corpus_file_path = os.path.join(root, f'{corpus_name}.jsonl')
                if not os.path.exists(corpus_file_path):
                    process_csv_corpus(file_path, root, text_col_name, 
                                       source_col_name, url_col_name, 
                                       corpus_name)
                else:
                    print(f'Corpus file {corpus_file_path} already exists, skipping...')
