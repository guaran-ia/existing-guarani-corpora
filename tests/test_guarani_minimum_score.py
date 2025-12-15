import json
from pathlib import Path


DATA_DIR = Path("data/processed")
MIN_GUARANI_SCORE = 0.70
GUARANI_LANG_CODES = {"grn"}


def test_all_documents_have_minimum_guarani_score():
    errors = []

    for jsonl_path in DATA_DIR.rglob("*.jsonl"):
        with jsonl_path.open(encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                obj = json.loads(line)

                if obj["language"] not in GUARANI_LANG_CODES:
                    errors.append(
                        f"{jsonl_path} line {line_num}: invalid language "
                        f"{obj['language']}"
                    )
                    continue

                if obj["language_score"] < MIN_GUARANI_SCORE:
                    errors.append(
                        f"{jsonl_path} line {line_num}: "
                        f"language_score={obj['language_score']}"
                    )

    assert not errors, (
        "Documents below minimum Guarani threshold:\n" + "\n".join(errors)
    )
