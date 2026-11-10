#!/usr/bin/env python3
"""Load the generated resume corpus into the running API container.

Run this inside the ``api`` container, where the application package and its
configuration are available::

    docker compose cp data/seed_resumes/. api:/tmp/seed_resumes
    docker compose cp scripts/seed_upload.py api:/tmp/seed_resumes/seed_upload.py
    docker compose exec -T api python /tmp/seed_resumes/seed_upload.py

The loader calls ``CandidateService.upload_resume`` directly, so files take the
same path as ``POST /api/v1/resumes``: text extraction (including OCR), contact
deduplication, SQLite persistence and Qdrant indexing. It bypasses HTTP only to
avoid the public upload rate limit while filling a local database.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from hr_assistant.container import ServiceContainer
from hr_assistant.core.config import get_settings


async def seed(corpus_dir: Path, manifest_path: Path, actor_id: str) -> list[dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    service = ServiceContainer(get_settings())
    await service.database.initialize()
    await service.vector_store.ensure_collection()

    seeded: list[dict] = []
    try:
        for item in manifest:
            path = corpus_dir / item["filename"]
            try:
                result = await service.candidates.upload_resume(
                    content=path.read_bytes(),
                    filename=item["filename"],
                    actor_id=actor_id,
                    candidate_id=None,
                    candidate_name=item["candidate_name"],
                    source=item["source"],
                    email=item["email"],
                    phone=item["phone"],
                )
            except Exception as error:  # keep going so one bad file does not stop the batch
                print(f"FAIL {item['filename']}: {type(error).__name__}: {error}")
                seeded.append({**item, "error": f"{type(error).__name__}: {error}"})
                continue

            document = result.sources[0] if result.sources else None
            seeded.append(
                {
                    **item,
                    "candidate_id": result.id,
                    "resume_id": document.id if document else None,
                    "indexing_status": document.indexing_status if document else None,
                    "extraction_method": document.extraction_method if document else None,
                }
            )
            print(
                f"OK   {item['filename']:32} {result.id} "
                f"status={document.indexing_status if document else '?'} "
                f"method={document.extraction_method if document else '?'}"
            )
    finally:
        await service.close()
    return seeded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-dir",
        default="/tmp/seed_resumes",
        help="Directory with the generated PDF/DOCX files and manifest.json",
    )
    parser.add_argument("--manifest", help="Defaults to <corpus-dir>/manifest.json")
    parser.add_argument("--actor-id", default="seed-script")
    parser.add_argument(
        "--output",
        help="Where to write seeded.json with real candidate IDs; defaults to <corpus-dir>/seeded.json",
    )
    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir)
    manifest_path = Path(args.manifest) if args.manifest else corpus_dir / "manifest.json"
    output_path = Path(args.output) if args.output else corpus_dir / "seeded.json"

    seeded = asyncio.run(seed(corpus_dir, manifest_path, args.actor_id))
    output_path.write_text(
        json.dumps(seeded, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    failed = [item for item in seeded if "error" in item]
    print(f"\nSeeded {len(seeded) - len(failed)}/{len(seeded)} files -> {output_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())