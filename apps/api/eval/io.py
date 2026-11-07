import json
from pathlib import Path
from typing import Any


def load_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found: {source}")

    if source.suffix.lower() == ".jsonl":
        records = []
        for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number} of {source}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object on line {line_number} of {source}")
            records.append(record)
        return records

    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and isinstance(payload.get("records"), list):
        records = payload["records"]
    elif isinstance(payload, dict):
        records = [payload]
    else:
        raise ValueError("JSON dataset must be an object, an array, or an object with 'records'.")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("Every dataset record must be a JSON object.")
    return records


def write_report(report: dict[str, Any], path: str | Path | None) -> None:
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(f"{rendered}\n", encoding="utf-8")
    print(rendered)


def record_id(record: dict[str, Any], index: int) -> str:
    return str(record.get("id", f"sample-{index + 1}"))
