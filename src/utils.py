import json
import re
import spacy

from corpus.src.pipeline.language_identifier.language_identifier import LanguageIdentifier


# set up spacy word segmentator
word_seg = spacy.blank("xx")
# set up guarani language identifier
identifier = LanguageIdentifier(glotlid=True, fasttext=True, openlid=True)


def word_count_split(text):
  return (len(text.split()))


def word_count_spacy(text, include_punct=False):
    words = []
    for t in word_seg(text):
      if not include_punct and t.is_punct:
        continue
      words.append(t.text)
    return len(words)


def identify_language(text):
    result = identifier.identify_languages(text, k=1, raw_output=False)
    identified_lang = result['languages']
    if identified_lang[0] == 'grn':
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