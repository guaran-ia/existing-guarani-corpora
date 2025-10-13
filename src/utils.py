import json
import os
import re
import spacy

from corpus.src.pipeline.language_identifier.language_identifier import LanguageIdentifier


# set up spacy word segmentator
word_seg = spacy.blank("xx")
# set up guarani language identifier
identifier = LanguageIdentifier(glotlid=True, fasttext=True, openlid=True)
# define Guarani iso-6393 code
GN_CODE = 'grn'


def word_count_split(text):
    """
    Count the number of whitespace-separated tokens in a text string.

    This simple method splits the input string on whitespace characters
    and counts the resulting tokens. It does not perform any linguistic 
    normalization or punctuation filtering.

    Args:
        text (str): Input text to analyze.

    Returns:
        int: Number of tokens obtained by splitting the text on whitespace.
    """
    return (len(text.split()))


def word_count_spacy(text, include_punct=False):
    """
    Count the number of word tokens in a text using spaCy segmentation.

    Uses the `word_seg` tokenizer to segment the input text and counts
    tokens according to the specified punctuation inclusion rule.

    Args:
        text (str): Input text to be tokenized.
        include_punct (bool, optional): 
            If True, include punctuation tokens in the count.
            If False (default), exclude them.

    Returns:
        int: Number of tokens in the text after applying the inclusion rule.
    """
    words = []
    for t in word_seg(text):
      if not include_punct and t.is_punct:
        continue
      words.append(t.text)
    return len(words)


def identify_language(text):
    """
    Identify the primary language of a text using a language identification model.

    Utilizes a preconfigured `identifier` model to predict the most likely
    language of the input text. If the detected language is Guarani ('grn'),
    detailed metadata about the identification is returned; otherwise, None.

    Args:
        text (str): Input text whose language is to be identified.

    Returns:
        dict | None: 
            A dictionary containing language identification metadata if
            the detected language is 'grn', with the following keys:
                - 'lang' (str): Detected ISO 639-3 language code.
                - 'score' (float): Confidence score for the detection.
                - 'source_score' (float): Model-specific source score.
                - 'voting_method' (str): Method used for voting in language detection.
            Returns None if the language is not 'grn'.
    """
    result = identifier.identify_languages(text, k=1, raw_output=False)
    identified_lang = result['languages']
    if identified_lang[0] == GN_CODE:
        return {
            'lang': identified_lang[0], 
            'score': identified_lang[1], 
            'source_score': result['source'],
            'voting_method': result['voting']
        }
    return None


def create_jsonl(data, file_path, writing_mode='w'):
    """
    Creates a JSON Lines file from a list of dictionaries.

    Args:
        data (list): A list of dictionaries to be written to the file.
        file_path (str): The path to the output JSON Lines file.
    """
    try:
        with open(file_path, writing_mode, encoding='utf-8') as f:
            for item in data:
                  # Convert dictionary to JSON string
                json_string = json.dumps(item, ensure_ascii=False)
                  # Write JSON string followed by a newline character
                f.write(json_string + '\n')
        print(f'Successfully created JSON Lines file: {file_path}')
    except Exception as e:
        print(f'Error creating JSON Lines file: {e}')


def save_to_json(data, file_path):
    """
    Saves a Python dictionary to a JSON file.

    Args:
        data (dict): The dictionary to be saved.
        file_path (str): The path to the output JSON file.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            # Convert dictionary to JSON and write to file
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f'Successfully created JSON file: {file_path}')
    except Exception as e:
        print(f'Error creating JSON file: {e}')


def sanitize_tsv_corpus(file_path):
    """
    Clean and normalize a TSV (tab-separated values) corpus file in place.

    This function performs three cleaning operations to ensure TSV data 
    consistency and compatibility with downstream corpus processing:
      1. Removes all double-quote characters (`"`).
      2. Collapses consecutive tab characters into a single tab.
      3. Strips trailing/leading whitespace and ensures each line ends with a newline.

    The cleaned content replaces the original file contents.

    Args:
        file_path (str): Path to the TSV file to be sanitized.

    Returns:
        None
    """
    clean_tsv = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # 1. Remove all double quotes
            line = line.replace('"', '')
            # 2. Collapse multiple tabs into a single one
            line = re.sub(r'\t+', '\t', line.strip())
            # 3. Ensure each line ends properly
            clean_tsv.append(f'{line}\n')
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in clean_tsv:
            f.write(line)
            
        
def create_report(project_dir):
    data_dir = os.path.join(project_dir, 'data')
    file_path = os.path.join(data_dir, 'gn_corpora.json')
    with open(file_path, 'r', encoding='utf-8') as f:
        gn_corpora = json.load(f)
    report = "# Guarani Corpora \nPublicly available corpora that contain text in Guarani. \n---\n\n"
    report += "|Name|Multilingual|Parallel|Synthetic|License|Docs in Gn|Total Words|Total Chars|Avg. Words/Doc|Avg. Chars/Doc|Avg. Prop. of Gn| \n"
    report += "|:---|:---:|:---:|:---:|:---|---:|---:|---:|---:|---:|---:| \n"
    total_docs, total_words, total_chars = 0, 0, 0
    for corpus in gn_corpora:
        report += f"|[{corpus['name']}]({corpus['url']})|"
        if corpus['multilingual']:
            report += ":white_check_mark:|"
        else:
            report += " |"
        if corpus['parallel']:
            report += ":white_check_mark:|"
        else:
            report += " |"
        if corpus['synthetic']:
            report += ":white_check_mark:|"
        else:
            report += " |"
        report += f"{corpus['license']}|"
        corpus_report_file_path = os.path.join(data_dir, 'processed', corpus['name'], 
                                               f"{corpus['name']}_report.json")
        with open(corpus_report_file_path, 'r') as f:
            corpus_report_dict = json.load(f)
        report += f"{corpus_report_dict['num_docs']:,}|"
        total_docs += corpus_report_dict['num_docs']
        report += f"{corpus_report_dict['num_words_split']:,}|"
        total_words += corpus_report_dict['num_words_split']
        report += f"{corpus_report_dict['num_chars']:,}|"
        total_chars += corpus_report_dict['num_chars']
        report += f"{corpus_report_dict['avg_words_split']:.3f}|"
        report += f"{corpus_report_dict['avg_chars']:.3f}|"
        report += f"{corpus_report_dict['avg_language_score']:.3f}| \n"
    report += f"| Total |  |  |  | {total_docs:,} | {total_words:,} | {total_chars:,} |  |  |  | \n"
    report_file_path = os.path.join(project_dir, 'report.md')
    with open(report_file_path, 'w') as f:
        f.write(report)
