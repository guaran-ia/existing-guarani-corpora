import json
from pathlib import Path


def test_all_processed_jsonl_have_valid_structure():
    DATA_DIR = Path(__file__).parent.parent / "data" / "processed"

    REQUIRED_FIELDS = {
        "text": str,
        "corpus": str,
        "corpus_file": str,
        "source": str,
        "url": str,
        "language": str,
        "language_score": float,
        "language_script": str,
        "language_score_source": (str, type(None)),
        "language_identification_method": (str, type(None)),
        "num_words_split": int,
        "num_words_punct_spacy": int,
        "num_words_no_punct_spacy": int,
        "num_chars": int,
    }

    errors = []

    jsonl_files = list(DATA_DIR.rglob("*.jsonl"))
    assert jsonl_files, "No JSONL files found under data/processed/"

    for jsonl_path in jsonl_files:
        with jsonl_path.open(encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    errors.append(
                        f"{jsonl_path} line {line_num}: invalid JSON ({e})"
                    )
                    continue

                for field, expected_type in REQUIRED_FIELDS.items():
                    if field not in obj:
                        errors.append(
                            f"{jsonl_path} line {line_num}: missing field '{field}'"
                        )
                    elif not isinstance(obj[field], expected_type):
                        errors.append(
                            f"{jsonl_path} line {line_num}: field '{field}' "
                            f"expected {expected_type}, got {type(obj[field]).__name__}"
                        )

                # Extra simple validation
                if "text" in obj and not obj["text"].strip():
                    errors.append(
                        f"{jsonl_path} line {line_num}: empty text field"
                    )

    assert not errors, (
        "JSON structure validation failed:\n" + "\n".join(errors)
    )

