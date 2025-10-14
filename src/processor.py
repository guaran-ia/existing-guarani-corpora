import json
import os
import pandas as pd
import re
import xml.etree.ElementTree as ET

from utils import create_jsonl, word_count_spacy, word_count_split, \
    identify_language, save_to_json, sanitize_tsv_corpus


def process_parquet_files(dir_path):
    """
    Convert all `.parquet` files in a directory (recursively) to `.csv` format.

    Traverses the given directory tree, finds all Parquet files, and converts each 
    one into a corresponding CSV file using pandas with the PyArrow engine.
    If a corresponding CSV already exists, it will be skipped.

    Args:
        dir_path (str): Path to the root directory containing Parquet files.

    Returns:
        None
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

    Performs word counting using two methods (split and spaCy), identifies the 
    language, and compiles metadata describing the text for corpus integration.

    Args:
        text (str): The input text sample.
        corpus_name (str): Name of the corpus the text belongs to.
        corpus_file_name (str): Name of the file the text was extracted from.
        source (str): Source name or origin of the text.
        url (str): URL of the original text (if available).
        lang_code (str): Expected ISO 639-3 language code.
        lang_script (str): Script code (e.g., "Latn", "Cyrl").

    Returns:
        tuple: A tuple containing:
            - text_dict (dict): Dictionary with annotated text metadata.
            - num_words_split (int)
            - num_words_punct_spacy (int)
            - num_words_no_punct_spacy (int)
            - lang_score (float)
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

    Updates an existing report if it exists, aggregates statistics, and computes
    averages for word counts, characters, and language identification scores.

    Args:
        report_dict (dict): Dictionary containing corpus-level statistics.
        report_file_path (str): File path to save or update the JSON report.

    Returns:
        None
    """
    if os.path.exists(report_file_path):
        with open(report_file_path, 'r') as f:
            e_report_dict = json.load(f)
        # update dictionary if a previous report exist
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
        print(f'No documents found in file, skipping report generation.')
    
    save_to_json(report_dict, report_file_path)


def save_processing(output_dir_path, corpus_name, data, writing_mode, report_dict):
    """
    Save processed corpus data and associated report to disk.

    Creates a corpus-specific subdirectory (if missing), writes the processed
    samples to a JSONL file, and saves statistical summaries to a JSON report.

    Args:
        output_dir_path (str): Directory path for output corpora.
        corpus_name (str): Name of the corpus being processed.
        data (list[dict]): List of text metadata dictionaries.
        writing_mode (str): File write mode ('a' for append, 'w' for overwrite).
        report_dict (dict): Aggregated statistics for the corpus.

    Returns:
        None
    """
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

    Returns:
        dict: Dictionary with zero-initialized counters for word counts,
        character counts, and average statistics.
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


def read_csv_corpus(file_path, sep=',', names=None, ignore_bad_lines=True, 
                    drop_incomplete_records=True):
    """
    Read and clean a corpus from a CSV or TSV file.

    Args:
        file_path (str): Path to the CSV/TSV file.
        sep (str, optional): Column separator (default ',').
        names (list[str], optional): Custom column names.
        ignore_bad_lines (bool, optional): Skip malformed lines (default True).
        drop_incomplete_records (bool, optional): Drop rows with missing values (default True).

    Returns:
        pandas.DataFrame: Cleaned DataFrame containing corpus records.
    """
    if ignore_bad_lines:
        df = pd.read_csv(file_path, sep=sep, encoding='utf-8', names=names, 
                           on_bad_lines='skip')
    else:
        df = pd.read_csv(file_path, sep=sep, encoding='utf-8', names=names)
    if drop_incomplete_records:
        df = df.dropna()
    return df


def process_csv_corpus(file_path, output_dir_path, corpus_name, text_col_name, 
                       source_col_name='', url_col_name='', 
                       lang_code='grn', lang_script='Latn', writing_mode='a', 
                       sep=',', names=None):
    """
    Process and export a text corpus stored in CSV/TSV format.

    Reads text data, extracts metadata, performs linguistic analysis, and writes
    processed records to a JSONL file with a corresponding report.

    Args:
        file_path (str): Path to the CSV or TSV corpus file.
        output_dir_path (str): Directory for output corpus data.
        corpus_name (str): Name of the corpus.
        text_col_name (str): Column containing the text.
        source_col_name (str, optional): Column with source information.
        url_col_name (str, optional): Column with URLs.
        lang_code (str, optional): Default language code (default 'grn').
        lang_script (str, optional): Default script code (default 'Latn').
        writing_mode (str, optional): File mode for JSONL ('a' or 'w').
        sep (str, optional): Field separator (default ',').
        names (list[str], optional): Column names override.

    Returns:
        None
    """
    print(f'Processing corpus: {"/".join(file_path.split("/")[-2:])}')
    df = read_csv_corpus(file_path, sep, names)
    report_dict = get_report_dict()
    corpus_file_name = file_path.split('/')[-1]
    data = []
    for _, row in df.iterrows():
        text = row[text_col_name]
        if isinstance(text, str):
            source = row[source_col_name] if source_col_name in row else 'unknown'
            url = row[url_col_name] if url_col_name in row else 'unknown'
            text_dict, num_words_split, num_words_punct_spacy, num_words_no_punct_spacy, lang_score = \
                process_text(text, corpus_name, corpus_file_name, source, url, lang_code, lang_script)
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
        data.extend(process_text_collection(text_collection, report_dict, corpus_name,
                                            corpus_file_name, lang_code, lang_script))
    print(f'Finished processing {corpus_file_name}. From {df.shape[0]} lines, {report_dict["num_docs"]} were included')
    save_processing(output_dir_path, corpus_name, data, writing_mode, report_dict)


def read_txt_corpus(file_path):
    """
    Read a plain-text corpus file as a list of lines.

    Args:
        file_path (str): Path to the text file.

    Returns:
        list[str]: List of lines read from the file.
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
            text_dict, num_words_split, num_words_punct_spacy, \
                num_words_no_punct_spacy, lang_score = \
                process_text(text, corpus_name, corpus_file_name, 'unknown', 
                             'unknown', lang_code, lang_script)
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
    Process a line-based text corpus (e.g., `.txt`, `.gn` files).

    Supports line filtering, optional prefix-based selection, and splitting lines
    on custom separators. Each valid text line is analyzed and stored.

    Args:
        file_path (str): Path to the text file.
        output_dir_path (str): Output directory path.
        corpus_name (str): Name of the corpus.
        lang_code (str, optional): Default language code (default 'grn').
        lang_script (str, optional): Script code (default 'Latn').
        writing_mode (str, optional): Write mode for output file.
        separator (dict, optional): Dict specifying {'str': separator, 'idx': column index}.
        line_prefix (str, optional): Only process lines starting with this prefix.

    Returns:
        None
    """
    print(f'Processing corpus: {"/".join(file_path.split("/")[-2:])}')
    report_dict = get_report_dict()
    f_lines = read_txt_corpus(file_path)
    corpus_file_name = file_path.split('/')[-1]
    data = process_text_collection(f_lines, report_dict, corpus_name, 
                                   corpus_file_name, lang_code, lang_script, 
                                   line_prefix, separator)
    save_processing(output_dir_path, corpus_name, data, writing_mode, report_dict)


def read_xml_corpus(file_path):
    """
    Parse an XML corpus and extract text segments.

    Extracts text from all `<s>` elements within the XML document.

    Args:
        file_path (str): Path to the XML file.

    Returns:
        list[str]: List of text elements extracted from the XML structure.
    """
    tree = ET.parse(file_path)
    root = tree.getroot()  # Get the root element
    text_list = []
    for element in root.findall('.//s'):  # Find all 's' elements
        text_list.append(element.text)
    return text_list


def process_xml_corpus(file_path, output_dir_path, corpus_name, lang_code='grn', 
                       lang_script='Latn', writing_mode='a'):
    """
    Process and export an XML-based text corpus.

    Parses XML structure, extracts sentences, performs word counting and 
    language identification, and writes results to JSONL and report files.

    Args:
        file_path (str): Path to the XML file.
        output_dir_path (str): Output directory for processed corpus.
        corpus_name (str): Corpus identifier.
        lang_code (str, optional): Default language code.
        lang_script (str, optional): Script code.
        writing_mode (str, optional): Output write mode.

    Returns:
        None
    """
    print(f'Processing corpus: {"/".join(file_path.split("/")[-2:])}')
    report_dict = get_report_dict()
    text_list = read_xml_corpus(file_path)
    corpus_file_name = file_path.split('/')[-1]
    data = process_text_collection(text_list, report_dict, corpus_name,
                                   corpus_file_name, lang_code, lang_script)
    save_processing(output_dir_path, corpus_name, data, writing_mode, report_dict)


def read_jsonl_corpus(file_path):
    """
    Read a JSONL (JSON Lines) corpus into memory.

    Args:
        file_path (str): Path to the JSONL file.

    Returns:
        list[dict]: List of JSON objects representing corpus entries.
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
    Process a corpus stored in JSON Lines (JSONL) format.

    Iterates through JSON records, extracts relevant text fields (e.g., 
    'flores_passage' and 'question'), analyzes them, and saves results.

    Args:
        file_path (str): Path to the JSONL corpus file.
        output_dir_path (str): Directory for processed corpus.
        corpus_name (str): Corpus name.
        lang_code (str, optional): Language code.
        lang_script (str, optional): Script code.
        writing_mode (str, optional): File writing mode ('a' or 'w').

    Returns:
        None
    """
    print(f'Processing corpus: {"/".join(file_path.split("/")[-2:])}')
    data = []
    report_dict = get_report_dict()
    f_data = read_jsonl_corpus(file_path)
    corpus_file_name = file_path.split('/')[-1]
    for line in f_data:
        for text in [line['flores_passage'], line['question']]:
            if isinstance(text, str):
                text_dict, num_words_split, num_words_punct_spacy, num_words_no_punct_spacy, lang_score = \
                    process_text(text, corpus_name, corpus_file_name, 'unknown', 'unknown', lang_code, lang_script)
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

    Args:
        corpus_dir_path (str): Path to a corpus directory.

    Returns:
        list[str]: Filenames contained in the directory.
    """
    return os.listdir(corpus_dir_path)


def prepare_processing_cvs_corpus(corpus_dir_path, corpus_dir_name, filename, 
                                  processed_dir):
    """
    Identify corpus type and process a CSV-based corpus accordingly.

    Determines schema and processing parameters based on the corpus name,
    then delegates processing to `process_csv_corpus`.

    Args:
        corpus_dir_path (str): Directory containing corpus files.
        corpus_dir_name (str): Corpus name identifier.
        filename (str): File name to process.
        processed_dir (str): Output directory for processed results.

    Returns:
        None
    """
    file_path = os.path.join(corpus_dir_path, filename)
    if 'jojajovai' in corpus_dir_name:
        text_col_name = 'gn'
        source_col_name = 'source'
        url_col_name = None
        corpus_name = 'jojajovai'
    elif 'culturax' in corpus_dir_name:
        text_col_name = 'text'
        source_col_name = 'source'
        url_col_name = 'url'
        corpus_name = 'culturax'
    else:
        raise Exception(f'Unknown corpus in path {corpus_dir_path}')
    process_csv_corpus(file_path, processed_dir, corpus_name, text_col_name, 
                       source_col_name, url_col_name)


def prepare_processing_txt_corpus(corpus_dir_path, corpus_dir_name, filename,
                                  processed_dir):
    """
    Configure and process a text corpus file.

    Applies corpus-specific preprocessing logic (e.g., separators, line prefixes),
    then calls `process_txt_corpus`.

    Args:
        corpus_dir_path (str): Path to the corpus directory.
        corpus_dir_name (str): Name of the corpus.
        filename (str): Text file to process.
        processed_dir (str): Output directory for processed corpora.

    Returns:
        None
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
    process_txt_corpus(file_path, processed_dir, corpus_dir_name, 
                       separator=separator, line_prefix=line_prefix)


def prepare_processing_tsv_corpus(corpus_dir_path, corpus_dir_name, filename, 
                                  processed_dir):
    """
    Configure and process a TSV-based corpus.

    Handles specific corpus schemas (e.g., AmericasNLP variants, Tatoeba),
    and sanitizes TSV files if necessary before processing.

    Args:
        corpus_dir_path (str): Path to corpus directory.
        corpus_dir_name (str): Corpus name identifier.
        filename (str): TSV file to process.
        processed_dir (str): Output directory.

    Returns:
        None
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
    else:
        raise Exception(f'Unknown corpus in path {corpus_dir_path}')
    sanitize_tsv_corpus(file_path)
    process_csv_corpus(file_path, processed_dir, corpus_dir_name, text_col_name, 
                       sep='\t', names=names)


def prepare_processing_xml_corpus(corpus_dir_path, corpus_dir_name, filename, 
                                  processed_dir):
    """
    Prepare and process an XML corpus file.

    Args:
        corpus_dir_path (str): Directory containing the XML corpus.
        corpus_dir_name (str): Corpus name identifier.
        filename (str): XML file name.
        processed_dir (str): Directory to save processed corpus.

    Returns:
        None
    """
    file_path = os.path.join(corpus_dir_path, filename)
    process_xml_corpus(file_path, processed_dir, corpus_dir_name)


def prepare_processing_jsonl_corpus(corpus_dir_path, corpus_dir_name, filename, 
                                   processed_dir):
    """
    Prepare and process a JSONL corpus file.

    Args:
        corpus_dir_path (str): Path to corpus directory.
        corpus_dir_name (str): Name of corpus.
        filename (str): JSONL file name.
        processed_dir (str): Output directory.

    Returns:
        None
    """
    file_path = os.path.join(corpus_dir_path, filename)
    process_jsonl_corpus(file_path, processed_dir, corpus_dir_name)


def process_corpora(raw_corpora_dir_path, processed_corpora_dir, overwrite=False):
    """
    Process all supported corpus files in a directory tree.

    Detects corpus format by file extension and dispatches to the appropriate
    processing routine (CSV, TXT, TSV, XML, JSONL).

    Args:
        dir_path (str): Path to directory containing corpora.
        processed_corpora_dir (str): Directory to store processed results.

    Returns:
        None
    """
    os.makedirs(processed_corpora_dir, exist_ok=True)
    for corpus_dir_name in os.listdir(raw_corpora_dir_path):
        corpus_path = os.path.join(raw_corpora_dir_path, corpus_dir_name)
        processed_corpus_dir = os.path.join(processed_corpora_dir, corpus_dir_name)
        corpus_report_file_path = os.path.join(processed_corpus_dir, f'{corpus_dir_name}_report.json')
        if os.path.exists(corpus_report_file_path):
            if not overwrite:
                # if not overwrite and corpus report exists ignore
                continue
            else:
                # if overwrite and corpus report exists delete together with
                # the processed corpus
                os.remove(corpus_report_file_path)
                corpus_file_path = os.path.join(processed_corpus_dir, f'{corpus_dir_name}.jsonl')
                os.remove(corpus_file_path)
        corpus_file_names = get_corpus_file_names(corpus_path)
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
                prepare_processing_jsonl_corpus(
                    corpus_path, corpus_dir_name, filename, processed_corpora_dir
                )
            else:
                print(f'Extension of the file {filename} is not supported')


def compute_num_raw_records(raw_corpora_dir):
    """
    Compute the number of raw records across all corpora.

    Inspects each corpus file, counts raw text records, and accounts for 
    format-specific nuances (e.g., line filtering).

    Args:
        raw_corpora_dir (str): Directory containing unprocessed corpora.

    Returns:
        dict: Mapping of corpus name → number of raw records.
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
                    # for the corpus gua_spa only lines starting with the pattern 
                    # below should be included
                    pattern = r"^#[A-Za-z0-9]+:" # (e.g., `#dev15:`, `#test179:`)
                    text_lines = [line for line in file_content if re.match(pattern, line)]
                    raw_records[corpus_dir_name] += len(text_lines)
                else:
                    raw_records[corpus_dir_name] += len(file_content)
            elif filename.endswith('.tsv'):
                if corpus_dir_name == 'americasnli':
                    df = read_csv_corpus(file_path, '\t')
                    raw_records[corpus_dir_name] += df.shape[0] + len(df['premise'].unique())
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
                    # for each record in raw belele two lines are created in the 
                    # processed corpus, one line for the question
                    # and the other for the `flores` passage. therefore, the
                    # number of lines in the raw files is multipled by 2 to obtain
                    # the total number of lines in the processed corpus
                    mult_factor = 2
                raw_records[corpus_dir_name] += len(read_jsonl_corpus(file_path)) * mult_factor
    return raw_records


def check_processed_corpora(processed_corpora_dir, raw_records):
    """
    Validate that processed corpora match the expected record counts.

    Cross-checks processed JSONL and report files against counts computed
    from raw corpora. Raises assertion errors on mismatches.

    Args:
        processed_corpora_dir (str): Directory with processed corpora.
        raw_records (dict): Mapping of corpus name → expected record count.

    Returns:
        None
    """
    for corpus_dir_name in os.listdir(processed_corpora_dir):
        print(f'Checking corpus {corpus_dir_name}...')
        corpus_path = os.path.join(processed_corpora_dir, corpus_dir_name)
        corpus_file_names = get_corpus_file_names(corpus_path)
        for filename in corpus_file_names:
            file_path = os.path.join(corpus_path, filename)
            raw_corpus_num_records = raw_records[corpus_dir_name]
            if filename.endswith('json'):
                with open(file_path, 'r') as f:
                    processed_corpus_report = json.load(f)
                assert processed_corpus_report['num_docs'] == raw_corpus_num_records, \
                    f"Number of records reported for the copus {corpus_dir_name} is incorrect, "\
                    f"expected {raw_corpus_num_records}, reported {processed_corpus_report['num_docs']}"
            if filename.endswith('jsonl'):
                processed_corpus_num_records = len(read_jsonl_corpus(file_path))
                assert processed_corpus_num_records == raw_corpus_num_records, \
                    f"Number of processed records for the corpus {corpus_dir_name} is inconsistent, "\
                    f"expected {raw_corpus_num_records}, processed {processed_corpus_num_records}"
        print(f'Ok {corpus_dir_name}!')
    print('Everything is correct!, the number of proccessed records matchs expectation for each corpus')


def verify_processed_corpora(raw_corpora_dir, processed_corpora_dir):
    """
    Verify consistency between raw and processed corpora.

    Combines record counting (`compute_num_raw_records`) and validation
    (`check_processed_corpora`) into a full verification pipeline.

    Args:
        raw_corpora_dir (str): Directory with raw corpora.
        processed_corpora_dir (str): Directory with processed corpora.

    Returns:
        None
    """
    raw_records = compute_num_raw_records(raw_corpora_dir)
    check_processed_corpora(processed_corpora_dir, raw_records)
