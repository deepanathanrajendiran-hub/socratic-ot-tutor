"""
ingest/add_grays_nerves.py — Add Gray's Anatomy peripheral nerve sections to corpus.

Extracts brachial plexus + upper limb peripheral nerve pages from Gray's Anatomy
20e (public domain, 1918), late-chunks them with contextual embeddings, and
upserts to the existing ChromaDB — without re-processing all 2163 OpenStax chunks.

Sections extracted:
  1. Brachial Plexus              (PDF pages 930–933)
  2. Axillary & Musculocutaneous  (PDF pages 933–936)
  3. Median Nerve                 (PDF pages 937–940)
  4. Ulnar Nerve                  (PDF pages 940–943)
  5. Radial Nerve                 (PDF pages 943–945)
  6. Surface Anatomy — Upper Limb (PDF pages 1330–1336)

Usage:
    PYTHONPATH=. python3 ingest/add_grays_nerves.py
"""

import json
import os
import re
import sys

import fitz  # PyMuPDF

import config
from ingest.late_chunker import late_chunk_section
from ingest.vector_store import VectorStore

PDF_PATH   = os.path.join(config.RAW_DIR, "textbooks", "grays_anatomy_20e.pdf")
LATE_CHUNKS_PATH = os.path.join(config.CHUNKS_DIR, f"late_chunks_{config.DOMAIN}.json")

# ── Section definitions: (section_id_suffix, title, pdf_pages_0indexed) ─────
# PDF pages are 0-indexed. Book page N is at PDF index N-1 approximately.
# Verified by manual inspection of the PDF.
SECTION_DEFS = [
    (
        "grays_brachial_plexus",
        "Brachial Plexus — Gray's Anatomy 20e",
        list(range(929, 933)),   # PDF pages 930-933
    ),
    (
        "grays_axillary_musculocutaneous",
        "Axillary Nerve and Musculocutaneous Nerve — Gray's Anatomy 20e",
        list(range(932, 936)),   # PDF pages 933-936
    ),
    (
        "grays_median_nerve",
        "Median Nerve — Gray's Anatomy 20e",
        list(range(936, 940)),   # PDF pages 937-940
    ),
    (
        "grays_ulnar_nerve",
        "Ulnar Nerve — Gray's Anatomy 20e",
        list(range(939, 943)),   # PDF pages 940-943
    ),
    (
        "grays_radial_nerve",
        "Radial Nerve — Gray's Anatomy 20e",
        list(range(942, 945)),   # PDF pages 943-945
    ),
    (
        "grays_surface_anatomy_upper_limb",
        "Surface Anatomy Upper Extremity Nerves — Gray's Anatomy 20e",
        list(range(1329, 1336)), # PDF pages 1330-1336
    ),
]

# Running headers to strip (Gray's 1918 prints chapter title on alternating pages)
_HEADER_PATTERNS = [
    re.compile(r"^THE CERVICAL NERVES\s*\d*\s*$", re.MULTILINE),
    re.compile(r"^\d+\s*NEUROLOGY\s*$", re.MULTILINE),
    re.compile(r"^NEUROLOGY\s*\d*\s*$", re.MULTILINE),
    re.compile(r"^THE THORACIC NERVES\s*\d*\s*$", re.MULTILINE),
    re.compile(r"^SURFACE MARKINGS OF THE UPPER EXTREMITY\s*\d*\s*$", re.MULTILINE),
    re.compile(r"^\d+\s*SURFACE ANATOMY AND SURFACE MARKINGS\s*$", re.MULTILINE),
    re.compile(r"^\d+\s*$", re.MULTILINE),          # lone page numbers
]


def _clean(text: str) -> str:
    """Strip running headers and lone page numbers from extracted PDF text."""
    for pat in _HEADER_PATTERNS:
        text = pat.sub("", text)
    # Collapse 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_section(doc: fitz.Document, page_indices: list[int]) -> str:
    """Concatenate cleaned text from the given 0-indexed PDF pages."""
    parts = []
    for idx in page_indices:
        if idx < len(doc):
            parts.append(doc[idx].get_text())
    return _clean("\n".join(parts))


def _build_section_dict(suffix: str, title: str, text: str, page_start: int) -> dict:
    """Create a section dict matching the format expected by late_chunk_section()."""
    return {
        "id":               f"{config.DOMAIN}_{suffix}",
        "section_id":       f"{config.DOMAIN}_{suffix}",
        "section_title":    title,
        "parent_section":   title,
        "level":            1,
        "chapter":          "Peripheral Nervous System",
        "chapter_num":      config.OT_NEUROLOGY_CHAPTER,
        "section_num":      "",
        "page_start":       page_start,
        "page_end":         page_start + 4,
        "text":             text,
        "raw_text":         text,
        "source_pdf":       "grays_anatomy_20e.pdf",
    }


def parse_grays_sections(pdf_path: str) -> list[dict]:
    """Open Gray's Anatomy PDF and extract the nerve sections."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"Opening {os.path.basename(pdf_path)} ({os.path.getsize(pdf_path) // 1_000_000} MB)…")
    doc = fitz.open(pdf_path)
    print(f"  Pages: {len(doc)}")

    sections = []
    for suffix, title, page_indices in SECTION_DEFS:
        text = _extract_section(doc, page_indices)
        word_count = len(text.split())
        print(f"  {title}: {word_count} words from PDF pages {page_indices[0]+1}–{page_indices[-1]+1}")
        if word_count < 50:
            print(f"    ⚠  Very short — skipping (check page range)")
            continue
        sections.append(_build_section_dict(suffix, title, text, page_indices[0] + 1))

    doc.close()
    return sections


def late_chunk_and_upsert(sections: list[dict]) -> None:
    """Late-chunk new sections and upsert them into the existing ChromaDB."""
    vs = VectorStore(config.CHROMA_DIR, config.DOMAIN)
    existing_count = vs.chunks_col.count()
    print(f"\nExisting ChromaDB chunks: {existing_count}")

    new_chunks: list[dict] = []

    for section in sections:
        print(f"\n  Late-chunking: {section['section_title']}")
        try:
            chunks = late_chunk_section(section)
            print(f"    → {len(chunks)} chunks")
            new_chunks.extend(chunks)
        except Exception as exc:
            print(f"    ✗ FAILED: {exc}")

    if not new_chunks:
        print("\nNo new chunks produced — nothing to upsert.")
        return

    # ── Upsert directly to ChromaDB (bypass load_from_late_chunks count check) ──
    print(f"\nUpserting {len(new_chunks)} new chunks to ChromaDB…")
    for start in range(0, len(new_chunks), config.INGEST_BATCH_SIZE):
        batch = new_chunks[start : start + config.INGEST_BATCH_SIZE]
        vs.chunks_col.upsert(
            ids        = [c["id"]        for c in batch],
            embeddings = [c["embedding"] for c in batch],
            documents  = [c["text"]      for c in batch],
            metadatas  = [
                {
                    "section_id":        c["section_id"],
                    "section_title":     c["section_title"],
                    "chapter_num":       str(c.get("chapter_num", config.OT_NEUROLOGY_CHAPTER)),
                    "page_start":        str(c.get("page_start", "")),
                    "tier":              c.get("tier", "late_chunk"),
                    "chunk_index":       str(c.get("chunk_index", "0")),
                    "full_section_text": c.get("full_section_text", c["text"]),
                    "OT_priority":       str(c.get("OT_priority", "True")),
                }
                for c in batch
            ],
        )

    new_total = vs.chunks_col.count()
    added     = new_total - existing_count
    print(f"ChromaDB: {existing_count} → {new_total} chunks (+{added}) ✓")

    # ── Append new chunks to late_chunks JSON so the file stays in sync ─────────
    print(f"\nAppending {len(new_chunks)} chunks to {os.path.basename(LATE_CHUNKS_PATH)}…")
    if os.path.exists(LATE_CHUNKS_PATH):
        with open(LATE_CHUNKS_PATH, encoding="utf-8") as f:
            existing_chunks = json.load(f)
    else:
        existing_chunks = []

    combined = existing_chunks + new_chunks
    with open(LATE_CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False)
    print(f"  late_chunks file: {len(existing_chunks)} → {len(combined)} entries ✓")


def main() -> None:
    print("=" * 65)
    print("Add Gray's Anatomy Nerve Sections to OT Anatomy Corpus")
    print("=" * 65)

    # ── Step 1: Parse ─────────────────────────────────────────────────────────
    print("\n── Step 1: Parse Gray's Anatomy nerve pages ─────────────────")
    sections = parse_grays_sections(PDF_PATH)
    print(f"\n  Extracted {len(sections)} sections.")

    # ── Step 2: Late-chunk + upsert ───────────────────────────────────────────
    print("\n── Step 2: Late-chunk and upsert to ChromaDB ────────────────")
    late_chunk_and_upsert(sections)

    print("\n" + "=" * 65)
    print("Done. Re-run the faithfulness evaluation to see improvement:")
    print("  PYTHONPATH=. python3 evaluation/faithfulness.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
