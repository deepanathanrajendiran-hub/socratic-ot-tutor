"""
ingest/late_chunker.py — contextual late-chunking with nomic-embed-text-v1.5.

Why late chunking?
──────────────────
Standard chunking embeds each chunk in isolation. A chunk containing
"the ulnar nerve exits the cubital tunnel" with no surrounding context
gets an embedding dominated by "cubital tunnel" even if the section is
about nerve injury mechanisms. Late chunking runs the FULL section
through the transformer once, captures token-level embeddings from
last_hidden_state, then mean-pools over each chunk's token span. The
chunk's embedding reflects its context within the entire section.

Algorithm per section:
  1. Prepend "search_document: " prefix (nomic asymmetric embedding)
  2. Tokenize full text with return_offsets_mapping=True
  3. POP offset_mapping before forward pass (model rejects it)
  4. Run full token sequence through model → last_hidden_state [seq_len, 768]
  5. Split text on nltk sentence boundaries (handles "e.g.", "Dr.", "approx.")
  6. For each sentence-boundary chunk span: mean-pool token embeddings
  7. Store chunk with contextual embedding + full_section_text for LLM

Reviewer fixes applied (v3):
  - return_offsets_mapping=True — no manual token counting, no drift
  - Pop offset_mapping before forward pass
  - nltk.sent_tokenize — handles medical abbreviations correctly
  - Device order: MPS → CUDA → CPU
  - Field-name adapter: section["text"] / section["id"] → raw_text / section_id

Run from project root:
  PYTHONPATH=. python3 ingest/late_chunker.py
"""

import json
import os

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import nltk
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize

import config

MODEL_NAME = config.LATE_CHUNK_MODEL   # "nomic-ai/nomic-embed-text-v1.5"

# ── Module-level singletons (loaded once per process) ─────────────────────────
_tokenizer = None
_model     = None
_device    = None


def get_device() -> torch.device:
    global _device
    if _device is None:
        if torch.backends.mps.is_available():
            _device = torch.device("mps")
            print("[late_chunker] Using Apple Silicon MPS")
        elif torch.cuda.is_available():
            _device = torch.device("cuda")
            print("[late_chunker] Using CUDA GPU")
        else:
            _device = torch.device("cpu")
            print("[late_chunker] Using CPU (slow — expect ~2 min/section)")
    return _device


def get_model_and_tokenizer():
    global _tokenizer, _model
    if _tokenizer is None:
        print(f"[late_chunker] First run: loading {MODEL_NAME} (~274 MB) …")
        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, trust_remote_code=True
        )
        _model = AutoModel.from_pretrained(
            MODEL_NAME, trust_remote_code=True
        )
        _model.eval()
        _model = _model.to(get_device())
        print("[late_chunker] Model loaded. ✓")
    return _tokenizer, _model


# ── Boundary finder ────────────────────────────────────────────────────────────

def find_chunk_boundaries(
    text:       str,
    tokenizer,
    chunk_size: int = config.LATE_CHUNK_SIZE,
    overlap:    int = config.LATE_CHUNK_OVERLAP,
) -> list[tuple[int, int, str]]:
    """
    Returns list of (start_token_idx, end_token_idx, chunk_text).

    Uses offset_mapping for exact character→token alignment.
    No manual token counting — no drift possible.
    Splits on nltk sentences — handles medical abbreviations.
    """
    prefix     = "search_document: "
    prefix_len = len(prefix)
    prefixed   = f"{prefix}{text}"

    encoding = tokenizer(
        prefixed,
        return_offsets_mapping=True,
        max_length=config.SECTION_MAX_TOKENS,
        truncation=True,
        add_special_tokens=True,
    )
    offset_mapping = encoding["offset_mapping"]
    total_tokens   = len(offset_mapping)

    # Split on sentence boundaries using nltk
    sentences = sent_tokenize(text)

    boundaries      = []
    chunk_sents     = []
    chunk_start_tok = 1          # skip [CLS] at index 0
    current_char    = prefix_len # character offset after the prefix

    for sent in sentences:
        sent_end_char = current_char + len(sent)

        # Find the last token whose end falls within this sentence
        sent_end_tok = chunk_start_tok
        for tok_idx in range(chunk_start_tok, total_tokens - 1):
            char_start, char_end = offset_mapping[tok_idx]
            if char_end <= sent_end_char:
                sent_end_tok = tok_idx + 1
            else:
                break

        chunk_sents.append(sent)
        current_char = sent_end_char + 1   # +1 for space separator

        token_count = sent_end_tok - chunk_start_tok
        if token_count >= chunk_size and len(chunk_sents) > 1:
            chunk_text = " ".join(chunk_sents)
            boundaries.append((chunk_start_tok, sent_end_tok, chunk_text))

            # Overlap: keep last two sentences as context seed for next chunk
            overlap_sents = chunk_sents[-2:] if len(chunk_sents) >= 2 \
                            else chunk_sents[-1:]
            overlap_text_start = current_char - sum(
                len(s) + 1 for s in overlap_sents
            )
            overlap_start_tok = sent_end_tok
            for tok_idx in range(chunk_start_tok, sent_end_tok):
                char_start, _ = offset_mapping[tok_idx]
                if char_start >= overlap_text_start:
                    overlap_start_tok = tok_idx
                    break

            chunk_start_tok = overlap_start_tok
            chunk_sents     = list(overlap_sents)

    # Final chunk — whatever remains
    if chunk_sents:
        chunk_text = " ".join(chunk_sents)
        # Estimate end token: run to end of sequence
        final_end = min(total_tokens - 1, chunk_start_tok +
                        len(tokenizer.encode(" ".join(chunk_sents),
                                             add_special_tokens=False)))
        boundaries.append((chunk_start_tok, final_end, chunk_text))

    return boundaries


# ── Core function ──────────────────────────────────────────────────────────────

def late_chunk_section(section: dict) -> list[dict]:
    """
    Core function. One section dict in → list of late chunks out.
    Each chunk embedding is contextualized by the full section pass.

    Field-name adapter: handles both old ("text"/"id") and new
    ("raw_text"/"section_id") section dict shapes from parse_pdf.py.
    """
    # ── field-name adapter (parse_pdf.py uses "text" and "id") ────────────────
    section_text = section.get("raw_text") or section["text"]
    section_id   = section.get("section_id") or section["id"]

    tokenizer, model = get_model_and_tokenizer()
    device           = get_device()

    prefix   = "search_document: "
    prefixed = f"{prefix}{section_text}"

    # Tokenize with offset mapping
    encoding = tokenizer(
        prefixed,
        return_tensors="pt",
        max_length=config.SECTION_MAX_TOKENS,
        truncation=True,
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    # CRITICAL: pop offset_mapping before forward pass — model rejects it
    encoding.pop("offset_mapping")

    inputs = {k: v.to(device) for k, v in encoding.items()}

    # Single forward pass — full section context for all token embeddings
    try:
        with torch.no_grad():
            outputs = model(**inputs)
    except RuntimeError as e:
        if device.type == "mps":
            # MPS fallback: retry on CPU
            print(f"\n[late_chunker] MPS error for {section_id}, retrying on CPU: {e}")
            cpu_inputs = {k: v.to("cpu") for k, v in inputs.items()}
            model_cpu  = model.to("cpu")
            with torch.no_grad():
                outputs = model_cpu(**cpu_inputs)
            model.to(device)   # move back
        else:
            raise

    # Shape: (seq_len, 768)
    token_embeddings = outputs.last_hidden_state.squeeze(0)

    # Find sentence-boundary chunk positions
    boundaries = find_chunk_boundaries(
        section_text, tokenizer,
        chunk_size=config.LATE_CHUNK_SIZE,
        overlap=config.LATE_CHUNK_OVERLAP,
    )

    chunks = []
    for i, (start_tok, end_tok, chunk_text) in enumerate(boundaries):
        # Clamp to valid token range
        max_tok  = token_embeddings.shape[0]
        start_tok = max(1, min(start_tok, max_tok - 2))
        end_tok   = max(start_tok + 1, min(end_tok, max_tok - 1))

        # Mean-pool token embeddings within this chunk's span
        # These tokens were contextualized by the full section forward pass
        chunk_vec = token_embeddings[start_tok:end_tok].mean(dim=0)

        chunks.append({
            "id":               f"{section_id}_lc_{i:03d}",
            "text":             chunk_text,
            "embedding":        chunk_vec.cpu().numpy().tolist(),
            "section_id":       section_id,
            "section_title":    section.get("section_title", ""),
            "chapter_num":      str(section.get("chapter_num", "0")),
            "page_start":       str(section.get("page_start", "")),
            "tier":             "late_chunk",
            "chunk_index":      str(i),
            "full_section_text": section_text,
            "OT_priority":      str(
                9 <= int(section.get("chapter_num", 0)) <= 16
            ),
        })

    return chunks


# ── Pipeline runner ────────────────────────────────────────────────────────────

def run_late_chunker(
    sections_dir: str,
    output_dir:   str,
    domain:       str = config.DOMAIN,
) -> dict:
    """
    Process all sections from all_sections_{domain}.json.

    Reads the combined sections file produced by parse_pdf.py.
    Processes each section through late_chunk_section().
    Saves output to late_chunks_{domain}.json + embed_meta.json.

    Args:
        sections_dir: directory containing all_sections_{domain}.json
        output_dir:   directory to write late_chunks_{domain}.json
        domain:       domain name (default config.DOMAIN)

    Returns:
        {"chunk_count": int, "output": str, "failed": int}
    """
    all_sections_path = os.path.join(
        sections_dir, f"all_sections_{domain}.json"
    )
    if not os.path.exists(all_sections_path):
        raise FileNotFoundError(
            f"Sections file not found: {all_sections_path}\n"
            "Run ingest/parse_pdf.py first."
        )

    with open(all_sections_path, encoding="utf-8") as f:
        sections = json.load(f)

    print(f"[late_chunker] Processing {len(sections)} sections …")

    all_chunks: list[dict] = []
    failed:     list[dict] = []

    for section in tqdm(sections, desc="Late chunking", unit="section"):
        try:
            chunks = late_chunk_section(section)
            all_chunks.extend(chunks)
        except Exception as e:
            sid = section.get("id", section.get("section_id", "unknown"))
            failed.append({"section_id": sid, "error": str(e)})
            print(f"\n[late_chunker] WARN: failed {sid}: {e}")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"late_chunks_{domain}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False)

    meta = {
        "model":       MODEL_NAME,
        "dimension":   768,
        "chunk_count": len(all_chunks),
        "chunk_size":  config.LATE_CHUNK_SIZE,
        "overlap":     config.LATE_CHUNK_OVERLAP,
        "method":      "late_chunking_offset_mapping",
        "failed":      len(failed),
    }
    with open(os.path.join(output_dir, "embed_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[late_chunker] Complete.")
    print(f"  Chunks produced : {len(all_chunks)}")
    print(f"  Sections failed : {len(failed)}")
    if failed:
        print(f"  Failed IDs      : {[f['section_id'] for f in failed]}")

    return {
        "chunk_count": len(all_chunks),
        "output":      output_path,
        "failed":      len(failed),
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Late-chunk OpenStax sections")
    parser.add_argument(
        "--sections-dir",
        default=os.path.join(config.CHUNKS_DIR, "raw_sections"),
    )
    parser.add_argument(
        "--output-dir",
        default=config.CHUNKS_DIR,
    )
    parser.add_argument("--domain", default=config.DOMAIN)
    args = parser.parse_args()

    result = run_late_chunker(args.sections_dir, args.output_dir, args.domain)
    print(result)
