"""
ingest/add_ot_supplement.py — Add OT clinical supplement to the corpus.

Parses peripheral_nerve_ot_supplement.txt into 5 sections by heading,
late-chunks them, and upserts to existing ChromaDB.

Usage:
    PYTHONPATH=. python3 ingest/add_ot_supplement.py
"""

import json
import os
import re

import config
from ingest.late_chunker import late_chunk_section
from ingest.vector_store  import VectorStore

SUPPLEMENT_PATH  = os.path.join(config.RAW_DIR, "ot_specific",
                                "peripheral_nerve_ot_supplement.txt")
LATE_CHUNKS_PATH = os.path.join(config.CHUNKS_DIR,
                                f"late_chunks_{config.DOMAIN}.json")

# Section separator pattern: lines of 80 = signs
_SEP = re.compile(r"={40,}")


def parse_supplement(path: str) -> list[dict]:
    """Split supplement text on === headings into section dicts."""
    with open(path, encoding="utf-8") as f:
        raw = f.read()

    # Split on separator lines
    parts = _SEP.split(raw)

    sections = []
    i = 0
    while i < len(parts):
        block = parts[i].strip()
        # A heading block is short and all-caps
        if block and len(block.splitlines()) <= 3 and block.isupper():
            title = block.replace("\n", " ").strip()
            body  = parts[i + 1].strip() if i + 1 < len(parts) else ""
            if body and len(body.split()) >= 50:
                # slug for ID
                slug = re.sub(r"[^a-z0-9]+", "_", title.lower())[:50]
                sections.append({
                    "id":            f"{config.DOMAIN}_ot_supp_{slug}",
                    "section_id":    f"{config.DOMAIN}_ot_supp_{slug}",
                    "section_title": f"{title} — OT Supplement",
                    "parent_section": "Peripheral Nerve OT Supplement",
                    "level":         1,
                    "chapter":       "Peripheral Nerve OT Clinical Reference",
                    "chapter_num":   config.OT_NEUROLOGY_CHAPTER,
                    "section_num":   "",
                    "page_start":    0,
                    "page_end":      0,
                    "text":          body,
                    "raw_text":      body,
                    "source_pdf":    "peripheral_nerve_ot_supplement.txt",
                })
            i += 2
        else:
            i += 1

    return sections


def late_chunk_and_upsert(sections: list[dict]) -> None:
    vs = VectorStore(config.CHROMA_DIR, config.DOMAIN)
    existing_count = vs.chunks_col.count()
    print(f"Existing ChromaDB chunks: {existing_count}")

    new_chunks: list[dict] = []
    for section in sections:
        print(f"  Late-chunking: {section['section_title']}")
        try:
            chunks = late_chunk_section(section)
            print(f"    → {len(chunks)} chunks")
            new_chunks.extend(chunks)
        except Exception as exc:
            print(f"    ✗ FAILED: {exc}")

    if not new_chunks:
        print("No chunks produced.")
        return

    print(f"\nUpserting {len(new_chunks)} chunks…")
    for start in range(0, len(new_chunks), config.INGEST_BATCH_SIZE):
        batch = new_chunks[start : start + config.INGEST_BATCH_SIZE]
        vs.chunks_col.upsert(
            ids        = [c["id"]        for c in batch],
            embeddings = [c["embedding"] for c in batch],
            documents  = [c["text"]      for c in batch],
            metadatas  = [{
                "section_id":        c["section_id"],
                "section_title":     c["section_title"],
                "chapter_num":       str(c.get("chapter_num", config.OT_NEUROLOGY_CHAPTER)),
                "page_start":        str(c.get("page_start", "")),
                "tier":              c.get("tier", "late_chunk"),
                "chunk_index":       str(c.get("chunk_index", "0")),
                "full_section_text": c.get("full_section_text", c["text"]),
                "OT_priority":       "True",
            } for c in batch],
        )

    new_total = vs.chunks_col.count()
    print(f"ChromaDB: {existing_count} → {new_total} chunks (+{new_total - existing_count}) ✓")

    # Append to late_chunks JSON
    if os.path.exists(LATE_CHUNKS_PATH):
        with open(LATE_CHUNKS_PATH, encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = []
    combined = existing + new_chunks
    with open(LATE_CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False)
    print(f"late_chunks file: {len(existing)} → {len(combined)} ✓")


def main():
    print("=" * 60)
    print("Add OT Clinical Supplement to Corpus")
    print("=" * 60)

    print("\n── Step 1: Parse supplement ──────────────────────────────")
    sections = parse_supplement(SUPPLEMENT_PATH)
    print(f"  Extracted {len(sections)} sections:")
    for s in sections:
        print(f"    • {s['section_title']} ({len(s['text'].split())} words)")

    print("\n── Step 2: Late-chunk + upsert ───────────────────────────")
    late_chunk_and_upsert(sections)

    print("\n" + "=" * 60)
    print("Done. Re-run faithfulness:")
    print("  PYTHONPATH=. python3 evaluation/faithfulness.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
