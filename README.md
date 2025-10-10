# Existing Guarani Corpora

This repo contains the code used to explore the corpora containing Guarani text. 
In total, 11 corpora were analyzed including: `americasnlp`, `belele`, `culturax`,
`tatoeba`, `opus`, and `jojajovai`. The full list of corpora with their respective
download url can be accessed at `data/gn_corpora.json`.

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

## Corpora statistics

### Americas NLP [2021](https://github.com/AmericasNLP/americasnlp2021) and [2023](https://github.com/AmericasNLP/americasnlp2023)

* Number of docs: 27,027
* Number of words: 413,175
* Number of characters: 3,161,598
* Average number of words per document: 15.29
* Average number of characters per document: 116.98
* Average proportion of Guarani in documents: 0.98

### [Americas NLP 2022](https://github.com/AmericasNLP/americasnlp2022)

* Number of docs: 386
* Number of words: 1,666
* Number of characters: 11,941
* Average number of words per document: 4.32
* Average number of characters per document: 30.93
* Average proportion of Guarani in documents: 0.92

### [Americas NLP 2024](https://github.com/AmericasNLP/americasnlp2024)

* Number of docs: 109,719
* Number of words: 1,446,800
* Number of characters: 11,053,563
* Average number of words per document: 13.19
* Average number of characters per document: 100.74
* Average proportion of Guarani in documents: 0.95

### [Belele](https://huggingface.co/datasets/facebook/2M-Belebele)

* Number of docs: 1,800
* Number of words: 65,713
* Number of characters: 503,106
* Average number of words per document: 36.51
* Average number of characters per document: 279.50
* Average proportion of Guarani in documents: 0.99

### [CulturaX](https://huggingface.co/datasets/uonlp/CulturaX)

* Number of docs: 63
* Number of words: 2,333
* Number of characters: 17,437
* Average number of words per document: 37.03
* Average number of characters per document: 276.78
* Average proportion of Guarani in documents: 0.99

### [Gua-Spa](https://github.com/pln-fing-udelar/gua-spa-2023)

* Number of docs: 1,500
* Number of words: 23,855
* Number of characters: 150,185
* Average number of words per document: 15.90
* Average number of characters per document: 100.12
* Average proportion of Guarani in documents: 0.73

### [Joemo](https://huggingface.co/datasets/mmaguero/gn-emotion-recognition)

* Number of docs: 1,571
* Number of words: 10,693
* Number of characters: 78,034
* Average number of words per document: 6.81
* Average number of characters per document: 49.67
* Average proportion of Guarani in documents: 0.77

### [Joff+](https://huggingface.co/datasets/mmaguero/gn-offensive-language-identification)

* Number of docs: 2,170
* Number of words: 15,016
* Number of characters: 110,058
* Average number of words per document: 6.92
* Average number of characters per document: 50.72
* Average proportion of Guarani in documents: 0.78

### [Jofun](https://huggingface.co/datasets/mmaguero/gn-humor-detection)

* Number of docs: 1,842
* Number of words: 12,958
* Number of characters: 95,196
* Average number of words per document: 7.03
* Average number of characters per document: 51.68
* Average proportion of Guarani in documents: 0.79

### [Jojajovai](https://github.com/pln-fing-udelar/jojajovai)

* Number of docs: 30,855
* Number of words: 456,446
* Number of characters: 3,517,466
* Average number of words per document: 14.79
* Average number of characters per document: 113.99
* Average proportion of Guarani in documents: 0.96

### [Josa](https://huggingface.co/datasets/mmaguero/gn-jopara-sentiment-analysis)

* Number of docs: 3,491
* Number of words: 47,305
* Number of characters: 388,330
* Average number of words per document: 13.55
* Average number of characters per document: 111.24
* Average proportion of Guarani in documents: 0.81

### [Opus](https://opus.nlpl.eu/GNOME/es&gn/v1/GNOME)

* Number of docs: 267
* Number of words: 627
* Number of characters: 4,388
* Average number of words per document: 2.34
* Average number of characters per document: 16.43
* Average proportion of Guarani in documents: 0.34

### [Tatoeba](https://tatoeba.org/en/downloads)

* Number of docs: 3,367
* Number of words: 13,269
* Number of characters: 95,040
* Average number of words per document: 3.94
* Average number of characters per document: 28.23
* Average proportion of Guarani in documents: 0.94

### Total

* Number of docs: 191,058
* Number of words: 2,509,856
* Number of characters: 19,186,342

## Contributors

[Jorge Saldivar](https://github.com/joausaga)
