# Existing Guarani Corpora

This repo contains the code used to explore the corpora containing Guarani text. 
In total, 11 corpora were analyzed including: `americasnlp`, `belele`, `culturax`,
`tatoeba`, `opus`, and `jojajovai`. Multilingual and parallel corpora were examined
processing only the Guarani text available in these datasets. The full list of 
corpora with their respective download urls can be accessed at `data/gn_corpora.json`.

## Processing Pipeline

The processing pipeline includes: i) download of raw corpora;
ii) extraction of corpora obtained in compressed format, like `tar.gz`;
iii) processing of corpora in multiple formats, including `txt`, `csv`, `tsv`, `xml`, 
and `json`; iv) transformation of the raw corpora format into a homogeneous scheme.

Processed corpus are stored in `data/processed` as `jsonl` files where `json` objects have the 
following structure:
```python
{
    'text': '',  # corpus text
    'corpus': '', # corpus name
    'corpus_file': '', # corpus file name
    'source': '', # text source (if available)
    'url': '', # text source url (if available)
    'language': 'gnr', # iso-6393 code for Guarani
    'language_score': 0.0, # proportion of Guarani in the text
    'language_script': 'Latn', # Guarani script
    'language_score_source': '', # source of Guarani score
    'language_identification_method': '', # method used to identify language
    'num_words_split': 0, # number of words in text based on white-space split
    'num_words_punct_spacy': 0, # number of words in text based on the Spacy generic segmentator
    'num_words_no_punct_spacy': 0, # number of words in text based on the Spacy generic segmentator (excluding punctuation)
    'num_chars': 0 # number of characters in the text
}
```

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/guaran-ia/existing-guarani-corpora.git
    cd existing-guarani-corpora
    ```
2.  Create a virtual environment (recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```
3.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Clone the `language identifier` repository:
    ```bash
    # clone without blobs and without checking out files
    git clone --filter=blob:none --no-checkout https://github.com/guaran-ia/corpus.git
    cd corpus
    # initialize sparse-checkout and enable the 'cone' mode (simpler patterns)
    git sparse-checkout init --cone
    # set the language identifier path
    git sparse-checkout set src/pipeline/language_identifier
    # check out the main branch
    git checkout main
    ```
5. Rename `.env.sample` to `.env`
6. Add your HuggingFace Access Token to `.env`

## Usage

The pipeline can be executed by running 

```bash
cd src
python main.py
```

## Dependencies
*   Python 3.12+
*   [Guarania language identifier](https://github.com/guaran-ia/corpus/tree/main/src/pipeline/language_identifier)

> The corpus [CulturaX](https://huggingface.co/datasets/uonlp/CulturaX) demands to 
previously accept a license of usage. In this sense, it is required that an 
autheticated HuggingFace user accepts this licenses before executing the code 
and adds a HuggingFace Access Token to the `env` file.

## Report

A report with information and statistics about the processed corpora can be found [here](https://github.com/guaran-ia/existing-guarani-corpora/blob/main/report.md).

## Contributors

[Jorge Saldivar](https://github.com/joausaga)
