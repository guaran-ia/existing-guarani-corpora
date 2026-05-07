import json
import os
import pandas as pd
import re
import xml.etree.ElementTree as ET
import ast
import ijson

from urllib.parse import urlparse
from utils import create_jsonl, word_count_spacy, word_count_split, \
    identify_language, save_to_json, sanitize_tsv_corpus

MIN_LANG_SCORE = 0.70


def process_parquet_files(dir_path):
    """
    Convert all `.parquet` files in a directory (recursively) to `.csv` format.
    """
    for root, _, files in os.walk(dir_path):
        for filename in files:
            if filename.endswith('.parquet'):
                file_path = os.path.join(root, filename)
                output_path = os.path.splitext(file_path)[0] + '.csv'
                if not os.path.exists(output_path):
                    print(f'Processing parquet file {file_path}')
                    pd.read_parquet(file_path, engine='pyarrow').to_csv(output_path, index=False)
                    print(f'Processed file saved to {output_path}')


def process_text(text, corpus_name, corpus_file_name, source, url, lang_code, lang_script):
    """
    Process and annotate a text sample with linguistic metadata and word counts.
    """
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
        'corpus_file': corpus_file_name,
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
    """
    Generate and save a JSON report summarizing corpus statistics.
    """
    if os.path.exists(report_file_path):
        with open(report_file_path, 'r') as f:
            e_report_dict = json.load(f)

        report_dict['num_docs'] += e_report_dict['num_docs']
        report_dict['num_words_split'] += e_report_dict['num_words_split']
        report_dict['num_words_punct_spacy'] += e_report_dict['num_words_punct_spacy']
        report_dict['num_words_no_punct_spacy'] += e_report_dict['num_words_no_punct_spacy']
        report_dict['num_chars'] += e_report_dict['num_chars']
        report_dict['sum_lang_score'] += e_report_dict['sum_lang_score']

    if report_dict['num_docs'] > 0:
        report_dict['avg_words_split'] = report_dict['num_words_split'] / report_dict['num_docs']
        report_dict['avg_words_punct_spacy'] = report_dict['num_words_punct_spacy'] / report_dict['num_docs']
        report_dict['avg_words_no_punct_spacy'] = report_dict['num_words_no_punct_spacy'] / report_dict['num_docs']
        report_dict['avg_chars'] = report_dict['num_chars'] / report_dict['num_docs']
        report_dict['avg_language_score'] = report_dict['sum_lang_score'] / report_dict['num_docs']
    else:
        print('No documents found in file, skipping report generation.')

    save_to_json(report_dict, report_file_path)


def save_processing(output_dir_path, corpus_name, data, writing_mode, report_dict):
    """
    Save processed corpus data and associated report to disk.
    """
    if data and report_dict:
        output_dir_path = os.path.join(output_dir_path, corpus_name)
        os.makedirs(output_dir_path, exist_ok=True)

        output_file_path = os.path.join(output_dir_path, f'{corpus_name}.jsonl')
        create_jsonl(data, output_file_path, writing_mode)

        report_file_path = os.path.join(output_dir_path, f'{corpus_name}_report.json')
        save_report(report_dict, report_file_path)

        print('\n')


def get_report_dict():
    """
    Initialize an empty report dictionary for corpus statistics tracking.
    """
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


def get_domain(url):
    try:
        return urlparse(url).netloc
    except Exception:
        return None


def read_csv_corpus(file_path, sep=',', names=None, ignore_bad_lines=True,
                    drop_incomplete_records=True):
    """
    Read and clean a corpus from a CSV or TSV file.
    """
    if ignore_bad_lines:
        df = pd.read_csv(
            file_path,
            sep=sep,
            encoding='utf-8',
            names=names,
            on_bad_lines='skip'
        )
    else:
        df = pd.read_csv(file_path, sep=sep, encoding='utf-8', names=names)

    if drop_incomplete_records:
        df = df.dropna()

    if 'fineweb-2' in file_path and 'removed' in file_path:
        df['domain'] = df['url'].apply(get_domain)

        filtered_df = pd.DataFrame()
        filtered_df = pd.concat(
            [
                filtered_df,
                df.loc[
                    (df.domain.str.contains('gn.wikipedia.org')) &
                    (~df.filter_reason.str.contains('duplicated_', regex=False)) &
                    (df.filter_reason != 'char_dup_ratio')
                ]
            ]
        )

        filtered_df = pd.concat(
            [
                filtered_df,
                df.loc[
                    (df.domain.str.contains('gn.wikipedia.org')) &
                    (~df.filter_reason.str.contains('duplicated_', regex=False)) &
                    (df.filter_reason != 'char_dup_ratio')
                ]
            ]
        )

        filtered_df = pd.concat(
            [
                filtered_df,
                df.loc[
                    (df.domain.str.contains('gn.wikipedia.org')) &
                    (~df.filter_reason.str.contains('duplicated_', regex=False)) &
                    (df.filter_reason != 'char_dup_ratio')
                ]
            ]
        )

        df = filtered_df.copy()

    if 'commonvoice' in file_path:
        if 'reported' not in file_path:
            reported_file_path = os.path.join(os.path.dirname(file_path), 'reported.tsv')
            reported_sentences_df = pd.read_csv(reported_file_path, sep='\t')
            df = pd.read_csv(file_path, sep='\t')
            df = df.loc[~df.sentence_id.isin(reported_sentences_df.sentence_id.unique())]
        else:
            df = pd.DataFrame()

    return df


def process_csv_corpus(file_path, output_dir_path, corpus_name, text_col_name,
                       source_col_name='', url_col_name='',
                       lang_code='grn', lang_script='Latn', writing_mode='a',
                       sep=',', names=None):
    """
    Process and export a text corpus stored in CSV/TSV format.
    """
    gn_corpora = (
        'Alpaca-gn-gpt4',
        'Alpaca-gn-gpt3.5',
        'gn-multi-affective-alpaca',
        'mala-monolingual-split',
        'cc100_gn',
        'FinePDF',
    )

    twitter_corpora = (
        'twitter_giossa_october',
        'twitter_covid_es',
        'twitter_covid_py',
    )

    instruction_based_corpora = (
        'Alpaca-gn-gpt4',
        'Alpaca-gn-gpt3.5',
        'gn-multi-affective-alpaca',
    )

    if corpus_name in gn_corpora or corpus_name in twitter_corpora:
        df = read_csv_corpus(file_path, sep, names, drop_incomplete_records=False)
    else:
        df = read_csv_corpus(file_path, sep, names)

    if df.shape[0] > 0:
        print(f'Processing corpus: {"/".join(file_path.split("/")[-2:])}')
        report_dict = get_report_dict()
        corpus_file_name = file_path.split('/')[-1]
        data = []

        if corpus_name == 'multi-wiki-qa':
            for _, row in df.iterrows():
                context = row[text_col_name[0]]
                question = row[text_col_name[1]]
                answer = row[text_col_name[2]]

                text = ""

                if isinstance(context, str):
                    text += context

                if isinstance(question, str):
                    text += f"\n\n{question}"

                if isinstance(answer, str):
                    try:
                        answer_dict = ast.literal_eval(answer)
                        text += f"\n\n{' '.join(answer_dict['text'])}"
                    except Exception:
                        pass

                if not text:
                    continue

                source = row[source_col_name] if source_col_name in row else 'unknown'
                url = row[url_col_name] if url_col_name in row else 'unknown'

                text_dict, num_words_split, num_words_punct_spacy, \
                    num_words_no_punct_spacy, lang_score = process_text(
                        text,
                        corpus_name,
                        corpus_file_name,
                        source,
                        url,
                        lang_code,
                        lang_script
                    )

                if text_dict['language'] != lang_code:
                    continue
                if lang_score < MIN_LANG_SCORE:
                    continue

                data.append(text_dict)
                report_dict['num_docs'] += 1
                report_dict['num_words_split'] += num_words_split
                report_dict['num_words_punct_spacy'] += num_words_punct_spacy
                report_dict['num_words_no_punct_spacy'] += num_words_no_punct_spacy
                report_dict['num_chars'] += len(text)
                report_dict['sum_lang_score'] += lang_score

        elif corpus_name == 'moscar':
            for index, row in df.iterrows():
                try:
                    text_list = ast.literal_eval(
                        row[text_col_name].replace('}\n', '},').replace('...\n ', '')
                    )
                    metadata_list = ast.literal_eval(row[url_col_name])
                except Exception as e:
                    print(f"Failed to convert line {index} from string: {e}")
                    continue

                if isinstance(text_list, list):
                    text = '\n'.join([t['text'] for t in text_list])
                else:
                    continue

                url = 'unknown'
                if isinstance(metadata_list, dict):
                    url = metadata_list.get('url', 'unknown')

                source = 'unknown'

                text_dict, num_words_split, num_words_punct_spacy, \
                    num_words_no_punct_spacy, lang_score = process_text(
                        text,
                        corpus_name,
                        corpus_file_name,
                        source,
                        url,
                        lang_code,
                        lang_script
                    )

                if text_dict['language'] != lang_code:
                    continue
                if lang_score < MIN_LANG_SCORE:
                    continue

                data.append(text_dict)
                report_dict['num_docs'] += 1
                report_dict['num_words_split'] += num_words_split
                report_dict['num_words_punct_spacy'] += num_words_punct_spacy
                report_dict['num_words_no_punct_spacy'] += num_words_no_punct_spacy
                report_dict['num_chars'] += len(text)
                report_dict['sum_lang_score'] += lang_score

        else:
            for _, row in df.iterrows():
                if corpus_name in instruction_based_corpora:
                    parts = [
                        str(row["instruction"]),
                        str(row["input"]),
                        str(row["output"])
                    ]
                    text = " ".join(parts)
                else:
                    text = row[text_col_name]

                if isinstance(text, str) and text.strip():
                    source = row[source_col_name] if source_col_name in row else 'unknown'
                    url = row[url_col_name] if url_col_name in row else 'unknown'

                    text_dict, num_words_split, num_words_punct_spacy, \
                        num_words_no_punct_spacy, lang_score = process_text(
                            text,
                            corpus_name,
                            corpus_file_name,
                            source,
                            url,
                            lang_code,
                            lang_script
                        )

                    if text_dict['language'] != lang_code:
                        continue
                    if lang_score < MIN_LANG_SCORE:
                        continue

                    data.append(text_dict)
                    report_dict['num_docs'] += 1
                    report_dict['num_words_split'] += num_words_split
                    report_dict['num_words_punct_spacy'] += num_words_punct_spacy
                    report_dict['num_words_no_punct_spacy'] += num_words_no_punct_spacy
                    report_dict['num_chars'] += len(text)
                    report_dict['sum_lang_score'] += lang_score
                else:
                    print(f'Text {text} not an instance of string, excluding...')

            if corpus_name == 'americasnli':
                text_collection = df['premise'].unique().tolist()
                data.extend(
                    process_text_collection(
                        text_collection,
                        report_dict,
                        corpus_name,
                        corpus_file_name,
                        lang_code,
                        lang_script
                    )
                )

        print(
            f'Finished processing {corpus_file_name}. '
            f'From {df.shape[0]} lines, {report_dict["num_docs"]} were included'
        )
        save_processing(output_dir_path, corpus_name, data, writing_mode, report_dict)


def read_txt_corpus(file_path):
    """
    Read a plain-text corpus file as a list of lines.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        f_lines = f.readlines()
    return f_lines


def process_text_collection(content_collection, report_dict, corpus_name,
                            corpus_file_name, lang_code, lang_script, line_prefix='',
                            separator=''):
    data = []

    for text in content_collection:
        text = text.strip()

        if text and isinstance(text, str):
            if line_prefix and not text.startswith(line_prefix):
                continue

            if separator:
                text = text.split(separator['str'])[separator['idx']]

            source = 'unknown'
            url = 'unknown'

            if 'opus-all' in corpus_name:
                rgx = r"OPUS-(.*)_mono_gn.txt"
                res = re.search(rgx, corpus_file_name)
                try:
                    source = res.group(1)
                except Exception:
                    pass

            text_dict, num_words_split, num_words_punct_spacy, \
                num_words_no_punct_spacy, lang_score = process_text(
                    text,
                    corpus_name,
                    corpus_file_name,
                    source,
                    url,
                    lang_code,
                    lang_script
                )

            if text_dict['language'] != lang_code:
                continue
            if lang_score < MIN_LANG_SCORE:
                continue

            data.append(text_dict)
            report_dict['num_docs'] += 1
            report_dict['num_words_split'] += num_words_split
            report_dict['num_words_punct_spacy'] += num_words_punct_spacy
            report_dict['num_words_no_punct_spacy'] += num_words_no_punct_spacy
            report_dict['num_chars'] += len(text)
            report_dict['sum_lang_score'] += lang_score

    return data


def process_txt_corpus(file_path, output_dir_path, corpus_name, lang_code='grn',
                       lang_script='Latn', writing_mode='a', separator=None,
                       line_prefix=''):
    """
    Process a line-based text corpus.
    """
    print(f'Processing corpus: {"/".join(file_path.split("/")[-2:])}')
    report_dict = get_report_dict()
    f_lines = read_txt_corpus(file_path)
    corpus_file_name = file_path.split('/')[-1]

    data = process_text_collection(
        f_lines,
        report_dict,
        corpus_name,
        corpus_file_name,
        lang_code,
        lang_script,
        line_prefix,
        separator
    )

    save_processing(output_dir_path, corpus_name, data, writing_mode, report_dict)


def read_xml_corpus(file_path):
    """
    Parse an XML corpus and extract text segments.
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    text_list = []

    for element in root.findall('.//s'):
        text_list.append(element.text)

    return text_list


def process_xml_corpus(file_path, output_dir_path, corpus_name, lang_code='grn',
                       lang_script='Latn', writing_mode='a'):
    """
    Process and export an XML-based text corpus.
    """
    print(f'Processing corpus: {"/".join(file_path.split("/")[-2:])}')
    report_dict = get_report_dict()
    text_list = read_xml_corpus(file_path)
    corpus_file_name = file_path.split('/')[-1]

    data = process_text_collection(
        text_list,
        report_dict,
        corpus_name,
        corpus_file_name,
        lang_code,
        lang_script
    )

    save_processing(output_dir_path, corpus_name, data, writing_mode, report_dict)


def read_jsonl_corpus(file_path):
    """
    Read a JSONL corpus into memory.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        f_data = []
        for line in f:
            json_object = json.loads(line.strip())
            f_data.append(json_object)
    return f_data


def process_jsonl_corpus(file_path, output_dir_path, corpus_name, lang_code='grn',
                         lang_script='Latn', writing_mode='a'):
    """
    Process a corpus stored in JSON Lines format.
    """
    print(f'Processing corpus: {"/".join(file_path.split("/")[-2:])}')
    data = []
    report_dict = get_report_dict()
    f_data = read_jsonl_corpus(file_path)
    corpus_file_name = file_path.split('/')[-1]

    for line in f_data:
        for text in [line.get('flores_passage'), line.get('question'), line.get('trg')]:
            if isinstance(text, str):
                text_dict, num_words_split, num_words_punct_spacy, \
                    num_words_no_punct_spacy, lang_score = process_text(
                        text,
                        corpus_name,
                        corpus_file_name,
                        'unknown',
                        'unknown',
                        lang_code,
                        lang_script
                    )

                if text_dict['language'] != lang_code:
                    continue
                if lang_score < MIN_LANG_SCORE:
                    continue

                data.append(text_dict)
                report_dict['num_docs'] += 1
                report_dict['num_words_split'] += num_words_split
                report_dict['num_words_punct_spacy'] += num_words_punct_spacy
                report_dict['num_words_no_punct_spacy'] += num_words_no_punct_spacy
                report_dict['num_chars'] += len(text)
                report_dict['sum_lang_score'] += lang_score

    save_processing(output_dir_path, corpus_name, data, writing_mode, report_dict)


def read_json_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for obj in ijson.items(f, 'item'):
            yield obj


def process_json_corpus(file_path, output_dir_path, corpus_name, lang_code='grn',
                        lang_script='Latn', writing_mode='a'):
    """
    Process a corpus stored in JSON format.
    """
    print(f'Processing corpus: {"/".join(file_path.split("/")[-2:])}')
    data = []
    report_dict = get_report_dict()
    f_data = read_json_corpus(file_path)
    corpus_file_name = file_path.split('/')[-1]

    for line in f_data:
        if corpus_name == 'twitter_politic_bots':
            tweet_obj = line.get('tweet_obj', {})
            text = tweet_obj.get('full_text')

            if not isinstance(text, str) or not text.strip():
                continue

            source = 'unknown'
            url = 'unknown'

            text_dict, num_words_split, num_words_punct_spacy, \
                num_words_no_punct_spacy, lang_score = process_text(
                    text,
                    corpus_name,
                    corpus_file_name,
                    source,
                    url,
                    lang_code,
                    lang_script
                )

            if text_dict['language'] != lang_code:
                continue
            if lang_score < MIN_LANG_SCORE:
                continue

            data.append(text_dict)
            report_dict['num_docs'] += 1
            report_dict['num_words_split'] += num_words_split
            report_dict['num_words_punct_spacy'] += num_words_punct_spacy
            report_dict['num_words_no_punct_spacy'] += num_words_no_punct_spacy
            report_dict['num_chars'] += len(text)
            report_dict['sum_lang_score'] += lang_score

            continue

        lang = line['language']
        if lang != 'gn':
            continue

        source = line['source']

        if corpus_name == "apollomoedataset":
            conversations = line['conversations']
            text = ""
            for conv in conversations:
                text += conv['value']

        elif corpus_name == "apollomoebench":
            if isinstance(line['question'], str):
                text = line['question']
            else:
                continue

            if isinstance(line['options'], str):
                text += '\n\n' + line['options']

            if isinstance(line['answer'], str):
                text += '\n\n' + line['answer']

        text_dict, num_words_split, num_words_punct_spacy, \
            num_words_no_punct_spacy, lang_score = process_text(
                text,
                corpus_name,
                corpus_file_name,
                source,
                'unknown',
                lang_code,
                lang_script
            )

        if text_dict['language'] != lang_code:
            continue
        if lang_score < MIN_LANG_SCORE:
            continue

        data.append(text_dict)
        report_dict['num_docs'] += 1
        report_dict['num_words_split'] += num_words_split
        report_dict['num_words_punct_spacy'] += num_words_punct_spacy
        report_dict['num_words_no_punct_spacy'] += num_words_no_punct_spacy
        report_dict['num_chars'] += len(text)
        report_dict['sum_lang_score'] += lang_score

    save_processing(output_dir_path, corpus_name, data, writing_mode, report_dict)


def get_corpus_file_names(corpus_dir_path):
    """
    List all corpus files within a directory.
    """
    return os.listdir(corpus_dir_path)


def prepare_processing_csv_corpus(corpus_dir_path, corpus_dir_name, filename,
                                  processed_dir):
    """
    Identify corpus type and process a CSV-based corpus accordingly.
    """
    file_path = os.path.join(corpus_dir_path, filename)

    if 'jojajovai' in corpus_dir_name:
        text_col_name = 'gn'
        source_col_name = 'source'
        url_col_name = None
        corpus_name = 'jojajovai'

    elif corpus_dir_name == 'twitter_giossa_october':
        text_col_name = 'Tweet'
        source_col_name = ''
        url_col_name = ''
        corpus_name = 'twitter_giossa_october'

    elif corpus_dir_name == 'twitter_covid_es':
        text_col_name = 'text'
        source_col_name = 'source'
        url_col_name = ''
        corpus_name = 'twitter_covid_es'

    elif corpus_dir_name == 'twitter_covid_py':
        text_col_name = 'text'
        source_col_name = 'source'
        url_col_name = ''
        corpus_name = 'twitter_covid_py'

    elif 'culturax' in corpus_dir_name:
        text_col_name = 'text'
        source_col_name = 'source'
        url_col_name = 'url'
        corpus_name = 'culturax'

    elif 'fineweb-2' in corpus_dir_name:
        text_col_name = 'text'
        source_col_name = ''
        url_col_name = 'url'
        corpus_name = 'fineweb-2'

    elif 'multi-wiki-qa' in corpus_dir_name:
        text_col_name = ['context', 'question', 'answers']
        source_col_name = ''
        url_col_name = 'id'
        corpus_name = 'multi-wiki-qa'

    elif 'flores-200' in corpus_dir_name:
        text_col_name = 'sentence_grn_Latn'
        source_col_name = 'domain'
        url_col_name = 'URL'
        corpus_name = 'flores-200'

    elif 'moscar' in corpus_dir_name:
        text_col_name = 'text'
        source_col_name = ''
        url_col_name = 'metadata'
        corpus_name = 'moscar'

    elif 'glot500' in corpus_dir_name:
        text_col_name = 'text'
        source_col_name = ''
        url_col_name = ''
        corpus_name = 'glot500'

    elif 'Alpaca-gn-gpt4' in corpus_dir_name:
        text_col_name = 'instruction'
        source_col_name = ''
        url_col_name = ''
        corpus_name = 'Alpaca-gn-gpt4'

    elif 'Alpaca-gn-gpt3.5' in corpus_dir_name:
        text_col_name = 'instruction'
        source_col_name = ''
        url_col_name = ''
        corpus_name = 'Alpaca-gn-gpt3.5'

    elif 'FinePDF' in corpus_dir_name:
        text_col_name = 'text'
        source_col_name = ''
        url_col_name = 'url'
        corpus_name = 'FinePDF'

    elif 'gn-multi-affective-alpaca' in corpus_dir_name:
        text_col_name = 'instruction'
        source_col_name = ''
        url_col_name = ''
        corpus_name = 'gn-multi-affective-alpaca'

    elif 'mala-monolingual-split' in corpus_dir_name:
        text_col_name = 'text'
        source_col_name = ''
        url_col_name = ''
        corpus_name = 'mala-monolingual-split'

    elif 'smolsent__en_gn' in corpus_dir_name:
        text_col_name = 'trg'
        source_col_name = ''
        url_col_name = ''
        corpus_name = 'smolsent__en_gn'

    elif 'cc100_gn' in corpus_dir_name:
        text_col_name = 'text'
        source_col_name = ''
        url_col_name = ''
        corpus_name = 'cc100_gn'

    elif 'udhr-lid' in corpus_dir_name:
        text_col_name = 'sentence'
        source_col_name = ''
        url_col_name = ''
        corpus_name = 'udhr-lid'

    else:
        raise Exception(f'Unknown corpus in path {corpus_dir_path}')

    process_csv_corpus(
        file_path,
        processed_dir,
        corpus_name,
        text_col_name,
        source_col_name,
        url_col_name
    )


def prepare_processing_txt_corpus(corpus_dir_path, corpus_dir_name, filename,
                                  processed_dir):
    """
    Configure and process a text corpus file.
    """
    file_path = os.path.join(corpus_dir_path, filename)
    separator, line_prefix = None, ''

    if corpus_dir_name in ['joemo', 'joff+', 'jofun', 'josa']:
        separator = {'str': ' ||| ', 'idx': 0}
    elif corpus_dir_name == 'gua_spa':
        separator = {'str': ': ', 'idx': 1}
        line_prefix = '#'
    elif corpus_dir_name == 'grammar':
        separator = {'str': '.,', 'idx': 1}

    process_txt_corpus(
        file_path,
        processed_dir,
        corpus_dir_name,
        separator=separator,
        line_prefix=line_prefix
    )


def prepare_processing_tsv_corpus(corpus_dir_path, corpus_dir_name, filename,
                                  processed_dir):
    """
    Configure and process a TSV-based corpus.
    """
    file_path = os.path.join(corpus_dir_path, filename)
    names = None

    if corpus_dir_name == 'americasnlp2022':
        text_col_name = 'source_processed'
    elif corpus_dir_name == 'americasnli':
        text_col_name = 'hypothesis'
    elif corpus_dir_name == 'bible':
        text_col_name = 'col1'
        names = ['col1', 'col2']
    elif corpus_dir_name == 'ancora':
        text_col_name = 'col2'
        names = ['col1', 'col2']
    elif corpus_dir_name in ['americasnlp2024', 'tatoeba']:
        text_col_name = 'col3'
        names = ['col1', 'col2', 'col3']
    elif corpus_dir_name == 'commonvoice':
        text_col_name = 'sentence'
    else:
        raise Exception(f'Unknown corpus in path {corpus_dir_path}')

    sanitize_tsv_corpus(file_path)

    process_csv_corpus(
        file_path,
        processed_dir,
        corpus_dir_name,
        text_col_name,
        sep='\t',
        names=names
    )


def prepare_processing_xml_corpus(corpus_dir_path, corpus_dir_name, filename,
                                  processed_dir):
    """
    Prepare and process an XML corpus file.
    """
    file_path = os.path.join(corpus_dir_path, filename)
    process_xml_corpus(file_path, processed_dir, corpus_dir_name)


def prepare_processing_jsonl_corpus(corpus_dir_path, corpus_dir_name, filename,
                                    processed_dir):
    """
    Prepare and process a JSONL corpus file.
    """
    file_path = os.path.join(corpus_dir_path, filename)
    process_jsonl_corpus(file_path, processed_dir, corpus_dir_name)


def prepare_processing_json_corpus(corpus_dir_path, corpus_dir_name, filename,
                                   processed_dir):
    """
    Prepare and process a JSON corpus file.
    """
    file_path = os.path.join(corpus_dir_path, filename)
    process_json_corpus(file_path, processed_dir, corpus_dir_name)


def process_corpora(raw_corpora_dir_path, processed_corpora_dir, overwrite=False):
    """
    Process all supported corpus files in a directory tree.
    """
    os.makedirs(processed_corpora_dir, exist_ok=True)

    for corpus_dir_name in os.listdir(raw_corpora_dir_path):
        corpus_path = os.path.join(raw_corpora_dir_path, corpus_dir_name)
        processed_corpus_dir = os.path.join(processed_corpora_dir, corpus_dir_name)
        corpus_report_file_path = os.path.join(
            processed_corpus_dir,
            f'{corpus_dir_name}_report.json'
        )

        if os.path.exists(corpus_report_file_path):
            if not overwrite:
                continue

            os.remove(corpus_report_file_path)
            corpus_file_path = os.path.join(
                processed_corpus_dir,
                f'{corpus_dir_name}.jsonl'
            )
            os.remove(corpus_file_path)

        corpus_file_names = get_corpus_file_names(corpus_path)

        for filename in corpus_file_names:
            if filename.endswith('.csv'):
                prepare_processing_csv_corpus(
                    corpus_path,
                    corpus_dir_name,
                    filename,
                    processed_corpora_dir
                )
            elif filename.endswith('.gn') or filename.endswith('.txt'):
                prepare_processing_txt_corpus(
                    corpus_path,
                    corpus_dir_name,
                    filename,
                    processed_corpora_dir
                )
            elif filename.endswith('tsv'):
                prepare_processing_tsv_corpus(
                    corpus_path,
                    corpus_dir_name,
                    filename,
                    processed_corpora_dir
                )
            elif filename.endswith('xml'):
                prepare_processing_xml_corpus(
                    corpus_path,
                    corpus_dir_name,
                    filename,
                    processed_corpora_dir
                )
            elif filename.endswith('jsonl'):
                prepare_processing_jsonl_corpus(
                    corpus_path,
                    corpus_dir_name,
                    filename,
                    processed_corpora_dir
                )
            elif filename.endswith('.json'):
                prepare_processing_json_corpus(
                    corpus_path,
                    corpus_dir_name,
                    filename,
                    processed_corpora_dir
                )
            else:
                print(f'Extension of the file {filename} is not supported')


def compute_num_raw_records(raw_corpora_dir):
    """
    Compute the number of raw records across all corpora.
    """
    raw_records = {}

    for corpus_dir_name in os.listdir(raw_corpora_dir):
        raw_records[corpus_dir_name] = 0
        corpus_path = os.path.join(raw_corpora_dir, corpus_dir_name)
        corpus_file_names = get_corpus_file_names(corpus_path)

        for filename in corpus_file_names:
            file_path = os.path.join(corpus_path, filename)

            if filename.endswith('.csv'):
                df = read_csv_corpus(file_path)
                raw_records[corpus_dir_name] += df.shape[0]

            elif filename.endswith('.gn') or filename.endswith('.txt'):
                file_content = read_txt_corpus(file_path)

                if corpus_dir_name == 'gua_spa':
                    pattern = r"^#[A-Za-z0-9]+:"
                    text_lines = [
                        line for line in file_content
                        if re.match(pattern, line)
                    ]
                    raw_records[corpus_dir_name] += len(text_lines)
                else:
                    raw_records[corpus_dir_name] += len(file_content)

            elif filename.endswith('.tsv'):
                if corpus_dir_name == 'americasnli':
                    df = read_csv_corpus(file_path, '\t')
                    raw_records[corpus_dir_name] += (
                        df.shape[0] + len(df['premise'].unique())
                    )
                else:
                    names = None

                    if corpus_dir_name in ['americasnlp2024', 'tatoeba']:
                        names = ['col1', 'col2', 'col3']
                    elif corpus_dir_name in ['bible', 'ancora']:
                        names = ['col1', 'col2']

                    df = read_csv_corpus(file_path, '\t', names)
                    raw_records[corpus_dir_name] += df.shape[0]

            elif filename.endswith('xml'):
                raw_records[corpus_dir_name] += len(read_xml_corpus(file_path))

            elif filename.endswith('jsonl'):
                mult_factor = 1

                if corpus_dir_name == 'belele':
                    mult_factor = 2

                raw_records[corpus_dir_name] += (
                    len(read_jsonl_corpus(file_path)) * mult_factor
                )

            elif filename.endswith('.json'):
                if corpus_dir_name == 'twitter_politic_bots':
                    raw_records[corpus_dir_name] += sum(
                        1 for _ in read_json_corpus(file_path)
                    )

    return raw_records


def check_processed_corpora(processed_corpora_dir, raw_records):
    """
    Validate that processed corpora match the expected record counts.
    """
    for corpus_dir_name in os.listdir(processed_corpora_dir):
        print(f'Checking corpus {corpus_dir_name}...')
        corpus_path = os.path.join(processed_corpora_dir, corpus_dir_name)
        corpus_file_names = get_corpus_file_names(corpus_path)

        for filename in corpus_file_names:
            file_path = os.path.join(corpus_path, filename)
            raw_corpus_num_records = raw_records.get(corpus_dir_name, None)

            if not raw_corpus_num_records:
                continue

            if filename.endswith('json'):
                with open(file_path, 'r') as f:
                    processed_corpus_report = json.load(f)

                assert processed_corpus_report['num_docs'] == raw_corpus_num_records, \
                    f"Number of records reported for the copus {corpus_dir_name} is incorrect, " \
                    f"expected {raw_corpus_num_records}, reported {processed_corpus_report['num_docs']}"

            if filename.endswith('jsonl'):
                processed_corpus_num_records = len(read_jsonl_corpus(file_path))

                assert processed_corpus_num_records == raw_corpus_num_records, \
                    f"Number of processed records for the corpus {corpus_dir_name} is inconsistent, " \
                    f"expected {raw_corpus_num_records}, processed {processed_corpus_num_records}"

        print(f'Ok {corpus_dir_name}!')

    print('Everything is correct!, the number of proccessed records matchs expectation for each corpus')


def verify_processed_corpora(raw_corpora_dir, processed_corpora_dir):
    """
    Verify consistency between raw and processed corpora.
    """
    raw_records = compute_num_raw_records(raw_corpora_dir)
    check_processed_corpora(processed_corpora_dir, raw_records)