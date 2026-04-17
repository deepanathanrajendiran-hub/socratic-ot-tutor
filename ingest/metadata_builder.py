"""
ingest/metadata_builder.py — two functions, no class.

Function 1: build_diagram_chunk_links
  Links each anatomical diagram in metadata.json to the most relevant
  textbook medium chunks via the retrieve() pipeline.

Function 2: tag_large_chunks_with_concepts
  Assigns a clean concept_label and OT_priority flag to every large chunk
  so the question bank builder (Step 10) knows what Socratic questions
  to generate.
"""

import json
import os
import re

import config
from ingest.retrieval_pipeline import retrieve

# ── Tag → natural-language query mapping ──────────────────────────────────────
TAG_TO_QUERY: dict[str, str] = {
    "ulnar_nerve":               "ulnar nerve anatomy innervation",
    "hand_intrinsics":           "intrinsic hand muscles lumbricals interossei",
    "claw_hand":                 "ulnar nerve lesion hand deformity",
    "median_nerve":              "median nerve anatomy carpal tunnel",
    "carpal_tunnel":             "carpal tunnel transverse ligament median nerve",
    "thenar_muscles":            "thenar muscles thumb opposition",
    "radial_nerve":              "radial nerve posterior forearm extensors",
    "wrist_drop":                "wrist drop extensor muscles",
    "extensor_muscles":          "forearm extensor muscles wrist extension",
    "brachial_plexus":           "brachial plexus roots trunks divisions cords",
    "upper_extremity_innervation": "upper extremity nerve supply innervation",
    "spinal_cord":               "spinal cord gray white matter tracts",
    "motor_tracts":              "corticospinal tract motor pathway",
    "sensory_tracts":            "spinothalamic tract dorsal columns sensation",
    "SCI":                       "spinal cord injury level function",
    "dermatomes":                "dermatomes skin sensation nerve roots",
    "sensory_mapping":           "sensory distribution nerve cutaneous",
    "upper_extremity":           "upper extremity anatomy arm forearm",
    "rotator_cuff":              "rotator cuff muscles shoulder SITS",
    "shoulder_muscles":          "shoulder muscles deltoid rotator cuff",
    "SITS_muscles":              "supraspinatus infraspinatus teres subscapularis",
    "motor_cortex":              "motor cortex primary voluntary movement",
    "homunculus":                "homunculus cortical representation body map",
    "stroke":                    "stroke cerebrovascular cortical damage",
    "neuroplasticity":           "neuroplasticity cortical reorganization recovery",
    "lumbricals":                "lumbrical muscles finger extension MCP",
    "interossei":                "interossei muscles finger abduction adduction",
    # carpal_tunnel diagram also uses this unmapped tag:
    "wrist_anatomy":             "wrist anatomy carpal bones flexor tendons",
}


# ── Function 1 ─────────────────────────────────────────────────────────────────

def build_diagram_chunk_links(
    diagram_metadata_path: str,
    output_path: str,
) -> dict:
    """
    For each diagram in metadata.json, retrieve the most relevant textbook
    chunks for every chunk_tag and build enriched diagram records with
    chunk_links and primary_concept.

    Args:
        diagram_metadata_path: path to data/raw/diagrams/grays/metadata.json
        output_path:           where to write the enriched JSON array

    Returns:
        {"diagrams_processed": int, "total_links": int}
    """
    with open(diagram_metadata_path, encoding="utf-8") as f:
        diagrams = json.load(f)

    enriched: list[dict] = []
    total_links = 0

    for diagram in diagrams:
        filename   = diagram["filename"]
        tags       = diagram.get("chunk_tags", [])

        # Collect all candidate links across all tags for this diagram
        # key = medium chunk id → best link record (highest rerank_score)
        best_by_id: dict[str, dict] = {}

        for tag in tags:
            query = TAG_TO_QUERY.get(tag, tag.replace("_", " "))
            try:
                reranked, _ = retrieve(query, weak_topics=None, top_k=3)
            except Exception as exc:
                print(f"  [WARN] retrieve failed for tag={tag!r}: {exc}")
                continue

            for result in reranked:
                chunk_id = result["id"]
                score    = result["rerank_score"]
                # Keep the record with the highest rerank_score for each chunk
                if chunk_id not in best_by_id or score > best_by_id[chunk_id]["rerank_score"]:
                    best_by_id[chunk_id] = {
                        "chunk_id":      chunk_id,
                        "parent_id":     result["parent_id"],
                        "section_title": result["section_title"],
                        "rerank_score":  round(float(score), 4),
                    }

        # Sort all candidates by score descending, keep top 5
        sorted_links = sorted(
            best_by_id.values(),
            key=lambda x: x["rerank_score"],
            reverse=True,
        )[:5]

        # primary_concept = section_title of the highest-scoring link
        primary_concept = sorted_links[0]["section_title"] if sorted_links else ""

        enriched_record = {
            "filename":       filename,
            "structure":      diagram.get("structure", ""),
            "region":         diagram.get("region", ""),
            "ot_relevance":   diagram.get("ot_relevance", ""),
            "blind_test":     diagram.get("blind_test", False),
            "chunk_links":    sorted_links,
            "primary_concept": primary_concept,
        }
        enriched.append(enriched_record)
        total_links += len(sorted_links)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2)

    avg_links = total_links / len(enriched) if enriched else 0
    print(
        f"Diagram metadata built: {len(enriched)} diagrams, "
        f"avg {avg_links:.1f} chunk links each"
    )
    return {"diagrams_processed": len(enriched), "total_links": total_links}


# ── Function 2 ─────────────────────────────────────────────────────────────────

# Chapters 9-16 are the OT-priority anatomical content
_OT_PRIORITY_CHAPTERS = set(range(9, 17))


def _clean_concept_label(section_title: str) -> str:
    """
    "11.5 The Muscular System — Muscles of the Hand"
      → "Muscles of the Hand"

    "9.1 Classification of Joints"
      → "Classification of Joints"

    Steps:
      1. Strip leading chapter.section prefix  ("11.5 ")
      2. If " — " (em-dash) present, take the part after it
      3. Strip whitespace
    """
    # Step 1: remove numeric prefix like "11.5 " or "28.3 "
    label = re.sub(r"^\d+\.\d+\s+", "", section_title)
    # Step 2: take after em-dash separator if present
    if " \u2014 " in label:
        label = label.split(" \u2014 ", 1)[1]
    return label.strip()


def tag_large_chunks_with_concepts(
    chunks_dir: str,
    output_path: str,
) -> dict:
    """
    Read every large chunk from large_chunks_OT_anatomy.json and assign:
      - concept_label  (cleaned section_title)
      - OT_priority    (True for chapters 9-16)

    Also reads chunk_tags from the ChromaDB large collection if available,
    falling back to [] for chunks with no tags stored.

    Args:
        chunks_dir:  e.g. "data/processed/chunks/"
        output_path: where to write concept_tags.json

    Returns:
        {"total": int, "OT_priority": int, "non_priority": int}
    """
    # Find the large chunks JSON file
    large_path = os.path.join(chunks_dir, f"large_chunks_{config.DOMAIN}.json")
    if not os.path.exists(large_path):
        raise FileNotFoundError(f"Large chunks file not found: {large_path}")

    with open(large_path, encoding="utf-8") as f:
        large_chunks: list[dict] = json.load(f)

    # Build id → chunk_tags lookup from ChromaDB (they're stored as JSON strings)
    chunk_tags_from_db: dict[str, list] = {}
    try:
        from ingest.vector_store import VectorStore
        vs = VectorStore(config.CHROMA_DIR, config.DOMAIN)
        db_result = vs._col_large.get(
            include=["metadatas"],
            limit=len(large_chunks) + 10,
        )
        for id_, meta in zip(db_result["ids"], db_result["metadatas"]):
            raw = meta.get("chunk_tags", "[]")
            try:
                chunk_tags_from_db[id_] = json.loads(raw) if isinstance(raw, str) else raw
            except (json.JSONDecodeError, TypeError):
                chunk_tags_from_db[id_] = []
    except Exception as exc:
        print(f"  [WARN] Could not load chunk_tags from ChromaDB: {exc}")
        print("         Defaulting to [] for all chunks.")

    records: list[dict] = []
    ot_count      = 0
    non_ot_count  = 0

    for chunk in large_chunks:
        chapter_num  = chunk["chapter_num"]           # int in JSON
        section_title = chunk["section_title"]
        concept_label = _clean_concept_label(section_title)
        ot_priority   = chapter_num in _OT_PRIORITY_CHAPTERS
        chunk_tags    = chunk_tags_from_db.get(chunk["id"], [])

        record = {
            "large_chunk_id": chunk["id"],
            "section_title":  section_title,
            "concept_label":  concept_label,
            "chapter_num":    chapter_num,
            "OT_priority":    ot_priority,
            "page_start":     chunk.get("page_start", 0),
            "chunk_tags":     chunk_tags,
        }
        records.append(record)

        if ot_priority:
            ot_count += 1
        else:
            non_ot_count += 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print("Concept tagging complete:")
    print(f"  Total large chunks tagged: {len(records)}")
    print(f"  OT priority chunks: {ot_count}  (chapters 9-16)")
    print(f"  Non-priority chunks: {non_ot_count}")

    return {
        "total":       len(records),
        "OT_priority": ot_count,
        "non_priority": non_ot_count,
    }
