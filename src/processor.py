import json
import os
import pandas as pd
import xml.etree.ElementTree as ET

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


def process_text(text, corpus_name, source, url, lang_code, lang_script):
    num_words_split = word_count_split(text)
    num_words_punct_spacy = word_count_spacy(text, include_punct=True)
    num_words_no_punct_spacy = word_count_spacy(text, include_punct=False)
    ident_result = identify_language(text.replace('\n', ' ').replace('\r', ' '))
    lang_score, lang_code, lang_score_source, lang_ident_method = 0.0, lang_code, None, None
    if ident_result:
        lang_score = ident_result['score']
        lang_code = ident_result['lang']
        lang_score_source = ident_result['source_score']
        lang_ident_method = ident_result['voting_method']
    text_dict = {
        'text': text,
        'corpus': corpus_name,
        'source': source,
        'url': url,
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
    return text_dict, num_words_split, num_words_punct_spacy, num_words_no_punct_spacy, lang_score


def save_report(report_dict, report_file_path):
    if os.path.exists(report_file_path):
        with open(report_file_path, 'r') as f:
            e_report_dict = json.load(f)
        # update dictionary if a previous report exist
        report_dict['num_docs'] += e_report_dict['num_docs']
        report_dict['num_words_split'] += e_report_dict['num_words_split']
        report_dict['num_words_punct_spacy'] += e_report_dict['num_words_punct_spacy']
        report_dict['num_words_no_punct_spacy'] += e_report_dict['num_words_no_punct_spacy']
        report_dict['num_chars'] += e_report_dict['num_words_no_punct_spacy']
        report_dict['sum_lang_score'] += e_report_dict['sum_lang_score']
    
    if report_dict['num_docs'] > 0:
        report_dict['avg_words_split'] = report_dict['num_words_split'] / report_dict['num_docs']
        report_dict['avg_words_punct_spacy'] = report_dict['num_words_punct_spacy'] / report_dict['num_docs']
        report_dict['avg_words_no_punct_spacy'] = report_dict['num_words_no_punct_spacy'] / report_dict['num_docs']
        report_dict['avg_chars'] = report_dict['num_chars'] / report_dict['num_docs']
        report_dict['avg_language_score'] = report_dict['sum_lang_score'] / report_dict['num_docs']
    else:
        print(f'No documents found in file, skipping report generation.')
    
    save_to_json(report_dict, report_file_path)


def save_processing(output_dir_path, corpus_name, data, writing_mode, report_dict):
    output_dir_path = os.path.join(output_dir_path, corpus_name)
    os.makedirs(output_dir_path, exist_ok=True)
    output_file_path = os.path.join(output_dir_path, f'{corpus_name}.jsonl')
    create_jsonl(data, output_file_path, writing_mode)
    report_file_path = os.path.join(output_dir_path, f'{corpus_name}_report.json')
    save_report(report_dict, report_file_path)
    print('\n')


def get_report_dict():
    return {
        'num_docs': 0,
        'num_words_split': 0,
        'num_words_punct_spacy': 0,
        'num_words_no_punct_spacy': 0,
        'num_chars': 0,
        'sum_lang_score': 0,
        'avg_words_split': 0,
        'avg_words_punct_spacy': 0,
        'avg_words_no_punct_spacy': 0,
        'avg_chars': 0,
        'avg_language_score': 0
    }


def process_csv_corpus(file_path, output_dir_path, corpus_name, text_col_name, 
                       source_col_name='', url_col_name='', lang_code='grn', 
                       lang_script='Latn', writing_mode='a', sep=',', names=None):
    print(f'Processing CSV corpus: {"/".join(file_path.split("/")[-2:])}')
    df = pd.read_csv(file_path, sep=sep, encoding='utf-8', names=names)
    report_dict = get_report_dict()
    data = []
    for _, row in df.iterrows():
        text = row[text_col_name]
        if isinstance(text, str):
            source = row[source_col_name] if source_col_name in row else 'unknown'
            url = row[url_col_name] if url_col_name in row else 'unknown'
            text_dict, num_words_split, num_words_punct_spacy, num_words_no_punct_spacy, lang_score = \
                process_text(text, corpus_name, source, url, lang_code, lang_script)
            data.append(text_dict)
            report_dict['num_docs'] += 1
            report_dict['num_words_split'] += num_words_split
            report_dict['num_words_punct_spacy'] += num_words_punct_spacy
            report_dict['num_words_no_punct_spacy'] += num_words_no_punct_spacy
            report_dict['num_chars'] += len(text)
            report_dict['sum_lang_score'] += lang_score
    save_processing(output_dir_path, corpus_name, data, writing_mode, report_dict)
    

def process_txt_corpus(file_path, output_dir_path, corpus_name, lang_code='grn', 
                       lang_script='Latn', writing_mode='a', separator=None, 
                       line_prefix=''):
    print(f'Processing CSV corpus: {"/".join(file_path.split("/")[-2:])}')
    with open(file_path, 'r', encoding='utf-8') as f:
        f_lines = f.readlines()
    data = []
    report_dict = get_report_dict()
    for f_line in f_lines:
        text = f_line.strip()
        if text and isinstance(text, str):
            if line_prefix and not text.startswith(line_prefix):
                continue
            if separator:
                text = text.split(separator['str'])[separator['idx']]
            text_dict, num_words_split, num_words_punct_spacy, num_words_no_punct_spacy, lang_score = \
                process_text(text, corpus_name, 'unknown', 'unknown', lang_code, lang_script)
            data.append(text_dict)
            report_dict['num_docs'] += 1
            report_dict['num_words_split'] += num_words_split
            report_dict['num_words_punct_spacy'] += num_words_punct_spacy
            report_dict['num_words_no_punct_spacy'] += num_words_no_punct_spacy
            report_dict['num_chars'] += len(text)
            report_dict['sum_lang_score'] += lang_score
    save_processing(output_dir_path, corpus_name, data, writing_mode, report_dict)


def process_xml_corpus(file_path, output_dir_path, corpus_name, lang_code='grn', 
                       lang_script='Latn', writing_mode='a'):
    tree = ET.parse(file_path)
    root = tree.getroot()  # Get the root element

    print(f'Processing CSV corpus: {"/".join(file_path.split("/")[-2:])}')
    
    data = []
    report_dict = get_report_dict()
    # Iterate through elements
    for element in root.findall('.//s'):  # Find all 's' elements
        text = element.text
        if isinstance(text, str):
            text_dict, num_words_split, num_words_punct_spacy, num_words_no_punct_spacy, lang_score = \
                    process_text(text, corpus_name, 'unknown', 'unknown', lang_code, lang_script)
            data.append(text_dict)
            report_dict['num_docs'] += 1
            report_dict['num_words_split'] += num_words_split
            report_dict['num_words_punct_spacy'] += num_words_punct_spacy
            report_dict['num_words_no_punct_spacy'] += num_words_no_punct_spacy
            report_dict['num_chars'] += len(text)
            report_dict['sum_lang_score'] += lang_score
    save_processing(output_dir_path, corpus_name, data, writing_mode, report_dict)


def process_json_corpus(file_path, output_dir_path, corpus_name, lang_code='grn', 
                       lang_script='Latn', writing_mode='a'):
    with open(file_path, 'r', encoding='utf-8') as f:
        f_data = []
        for line in f:
            json_object = json.loads(line.strip())
            f_data.append(json_object)
    data = []
    report_dict = get_report_dict()
    for line in f_data:
        for text in [line['flores_passage'], line['question']]:
            if isinstance(text, str):
                text_dict, num_words_split, num_words_punct_spacy, num_words_no_punct_spacy, lang_score = \
                    process_text(text, corpus_name, 'unknown', 'unknown', lang_code, lang_script)
                data.append(text_dict)
                report_dict['num_docs'] += 1
                report_dict['num_words_split'] += num_words_split
                report_dict['num_words_punct_spacy'] += num_words_punct_spacy
                report_dict['num_words_no_punct_spacy'] += num_words_no_punct_spacy
                report_dict['num_chars'] += len(text)
                report_dict['sum_lang_score'] += lang_score
    save_processing(output_dir_path, corpus_name, data, writing_mode, report_dict)


def get_corpus_file_names(corpus_dir_name, corpus_dir_path):
    corpus_file_names = os.listdir(corpus_dir_path)
    processed_file_name = f'{corpus_dir_name}.jsonl'
    report_processed_file_name = f'{corpus_dir_name}_report.json'
    if processed_file_name in corpus_file_names:
        os.remove(os.path.join(corpus_dir_path, processed_file_name))
        corpus_file_names.remove(processed_file_name)
    if report_processed_file_name in corpus_file_names:
        os.remove(os.path.join(corpus_dir_path, report_processed_file_name))
        corpus_file_names.remove(report_processed_file_name)
    return corpus_file_names


def prepare_processing_cvs_corpus(corpus_dir_path, corpus_dir_name, filename, 
                                  processed_dir):
    file_path = os.path.join(corpus_dir_path, filename)
    if 'jojajovai' in corpus_dir_name:
        text_col_name = 'gn'
        source_col_name = 'source'
        url_col_name = None
        corpus_name = 'jojajovai'
    elif 'culturalx' in corpus_dir_name:
        text_col_name = 'text'
        source_col_name = 'source'
        url_col_name = 'url'
        corpus_name = 'culturalx'
    else:
        raise Exception(f'Unknown corpus in path {corpus_dir_path}')
    process_csv_corpus(file_path, processed_dir, corpus_name, text_col_name, 
                       source_col_name, url_col_name)


def prepare_processing_txt_corpus(corpus_dir_path, corpus_dir_name, filename,
                                  processed_dir):
    file_path = os.path.join(corpus_dir_path, filename)
    separator, line_prefix = None, ''
    if corpus_dir_name in ['joemo', 'joff+', 'jofun', 'josa']:
        separator = {'str': ' ||| ', 'idx': 0}
    elif corpus_dir_name == 'gua_spa':
        separator = {'str': ': ', 'idx': 1}
        line_prefix = '#'
    process_txt_corpus(file_path, processed_dir, corpus_dir_name, 
                       separator=separator, line_prefix=line_prefix)


def prepare_processing_tsv_corpus(corpus_dir_path, corpus_dir_name, filename, 
                                  processed_dir):
    file_path = os.path.join(corpus_dir_path, filename)
    names = None
    if corpus_dir_name == 'americasnlp2022':
        text_col_name = 'source_processed'
    elif corpus_dir_name in ['americasnlp2024', 'tatoeba']:
        text_col_name = 'col3'
        names = ['col1', 'col2', 'col3']
    else:
        raise Exception(f'Unknown corpus in path {corpus_dir_path}')
    process_csv_corpus(file_path, processed_dir, corpus_dir_name, text_col_name, 
                       sep='\t', names=names)


def prepare_processing_xml_corpus(corpus_dir_path, corpus_dir_name, filename, 
                                  processed_dir):
    file_path = os.path.join(corpus_dir_path, filename)
    process_xml_corpus(file_path, processed_dir, corpus_dir_name)


def prepare_processing_json_corpus(corpus_dir_path, corpus_dir_name, filename, 
                                   processed_dir):
    file_path = os.path.join(corpus_dir_path, filename)
    process_json_corpus(file_path, processed_dir, corpus_dir_name)


def process_corpora(dir_path, processed_corpora_dir):
    os.makedirs(processed_corpora_dir, exist_ok=True)
    for corpus_dir_name in os.listdir(dir_path):
        corpus_path = os.path.join(dir_path, corpus_dir_name)
        corpus_file_names = get_corpus_file_names(corpus_dir_name, corpus_path)
        for filename in corpus_file_names:
            if filename.endswith('.csv'):
                prepare_processing_cvs_corpus(
                    corpus_path, corpus_dir_name, filename, processed_corpora_dir
                )
            elif filename.endswith('.gn') or filename.endswith('.txt'):
                prepare_processing_txt_corpus(
                    corpus_path, corpus_dir_name, filename, processed_corpora_dir
                )
            elif filename.endswith('tsv'):
                prepare_processing_tsv_corpus(
                    corpus_path, corpus_dir_name, filename, processed_corpora_dir
                )
            elif filename.endswith('xml'):
                prepare_processing_xml_corpus(
                    corpus_path, corpus_dir_name, filename, processed_corpora_dir
                )
            elif filename.endswith('jsonl'):
                prepare_processing_json_corpus(
                    corpus_path, corpus_dir_name, filename, processed_corpora_dir
                )
            else:
                print(f'Extension of the file {filename} is not supported')
            
