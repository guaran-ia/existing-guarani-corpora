import json
from pathlib import Path
from collections import defaultdict


def test_all_documents_have_minimum_guarani_score():
    DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
    MIN_GUARANI_SCORE = 0.7
    GUARANI_LANG_CODES = {"grn", "gug", "gn"}

    stats = defaultdict(lambda: {"total": 0, "failed": 0})

    jsonl_files = list(DATA_DIR.rglob("*.jsonl"))
    assert jsonl_files, "No JSONL files found under data/processed/"

    for jsonl_path in jsonl_files:
        corpus_name = jsonl_path.parent.name
        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if obj["language"] in GUARANI_LANG_CODES:
                    stats[corpus_name]["total"] += 1
                    if obj["language_score"] < MIN_GUARANI_SCORE:
                        stats[corpus_name]["failed"] += 1

    report_lines = []
    has_failures = False

    for corpus, counts in sorted(stats.items()):
        if counts["total"] == 0:
            continue
        percentage = (counts["failed"] / counts["total"]) * 100
        report_lines.append(
            f"{corpus}: {counts['failed']} / {counts['total']} "
            f"documents below threshold ({percentage:.2f}%)"
        )
        if counts["failed"] > 0:
            has_failures = True

    assert not has_failures, (
        "Some corpora contain documents below the minimum Guarani threshold:\n"
        + "\n".join(report_lines)
    )



