"""
ingest/vector_store.py — ChromaDB abstraction layer. v3

RULE: Nothing outside this file ever imports chromadb directly.
      All other modules call VectorStore methods only.

Architecture (v3)
─────────────────
One collection per domain, persisted at config.CHROMA_DIR:

  {domain}_chunks  hnsw:space=cosine  — queried at retrieval time
                   stores: late-chunk contextual embeddings (768-dim)
                   metadata includes full_section_text for LLM context

Retrieval path (v3):
  1. corrective_retrieve() in retrieval/crag.py calls vs.chunks_col.query()
  2. full_section_text is returned directly from chunk metadata

nomic asymmetric prefixes (unchanged from v1/v2):
  Stored documents   : "search_document: "  (applied in late_chunker.py)
  Query strings      : "search_query: "     (applied in retrieval/crag.py)
"""

import json
import os
from glob import glob

import chromadb
import requests
from tqdm import tqdm

import config

# ── Ollama constants (still used for query embedding via crag.py) ──────────────
_OLLAMA_BASE   = "http://localhost:11434"
_EMBED_MODEL   = config.EMBED_MODEL          # "nomic-embed-text"
_QUERY_PREFIX  = "search_query: "
_EMBED_TIMEOUT = 30

# ── ChromaDB collection space ──────────────────────────────────────────────────
_COSINE_META = {"hnsw:space": "cosine"}


# ═══════════════════════════════════════════════════════════════════════════════
class VectorStore:
    """
    Single interface to ChromaDB for one domain.

    v3: one collection {domain}_chunks replaces the old three-tier
    ({domain}_medium, {domain}_large, {domain}_small).

    Usage:
        vs = VectorStore(config.CHROMA_DIR, config.DOMAIN)
        vs.load_from_late_chunks("data/processed/chunks/late_chunks_OT_anatomy.json")
        # retrieval happens via retrieval/crag.py → vs.chunks_col.query()
        full_text = vs.get_full_section("OT_anatomy_ch12_sec0042")
    """

    # ── init ───────────────────────────────────────────────────────────────────

    def __init__(self, persist_dir: str, domain: str) -> None:
        """
        Create or reload a persistent ChromaDB client.
        Creates the single {domain}_chunks collection if it does not exist.

        Args:
            persist_dir: local directory to persist ChromaDB data (config.CHROMA_DIR)
            domain:      "OT_anatomy" or "physics" (config.DOMAIN)
        """
        self.persist_dir = persist_dir
        self.domain      = domain

        os.makedirs(persist_dir, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_dir)

        # v3: single collection replaces _medium/_large/_small
        self.chunks_col = self._client.get_or_create_collection(
            name=f"{domain}_chunks",
            metadata=_COSINE_META,
        )

        print(
            f"VectorStore loaded: 1 collection for domain {domain}  "
            f"(chunks={self.chunks_col.count()})"
        )

    # ── load late chunks (v3 primary loader) ──────────────────────────────────

    def load_from_late_chunks(self, json_path: str) -> dict:
        """
        Read late_chunks_{domain}.json produced by late_chunker.py and
        upsert all chunks into the {domain}_chunks collection.

        Each chunk must carry a pre-computed 768-dim embedding from the
        late-chunking contextual pass (nomic-embed-text-v1.5).

        Metadata stored per chunk:
          section_id        — parent section ID for get_full_section()
          section_title     — human-readable section title
          chapter_num       — str (ChromaDB metadata must be scalar)
          page_start        — str
          tier              — "late_chunk"
          chunk_index       — str
          full_section_text — full section text (used by corrective_retrieve()
                              to pass LLM context without a second DB lookup)
          OT_priority       — "True"/"False" (str)

        Processes in batches of 100. tqdm progress bar.
        Verifies collection.count() == expected after upsert.

        Args:
            json_path: path to late_chunks_{domain}.json

        Returns:
            {"status": "ok", "upserted": int}
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(
                f"Late chunks file not found: {json_path}\n"
                "Run ingest/late_chunker.py first."
            )

        print(f"\n[VectorStore] Loading late chunks from {os.path.basename(json_path)} …")
        with open(json_path, encoding="utf-8") as f:
            chunks = json.load(f)

        total      = len(chunks)
        batch_size = 100
        upserted   = 0

        for start in tqdm(range(0, total, batch_size),
                          desc="  Upserting late chunks", unit="batch"):
            batch = chunks[start : start + batch_size]

            ids        = [c["id"]        for c in batch]
            embeddings = [c["embedding"] for c in batch]
            documents  = [c["text"]      for c in batch]
            metadatas  = [
                {
                    "section_id":        c["section_id"],
                    "section_title":     c["section_title"],
                    "chapter_num":       str(c.get("chapter_num", "0")),
                    "page_start":        str(c.get("page_start", "")),
                    "tier":              c.get("tier", "late_chunk"),
                    "chunk_index":       str(c.get("chunk_index", "0")),
                    "full_section_text": c.get("full_section_text", c["text"]),
                    "OT_priority":       str(c.get("OT_priority", "False")),
                }
                for c in batch
            ]

            self.chunks_col.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            upserted += len(batch)

        # ── verify count ───────────────────────────────────────────────────────
        actual = self.chunks_col.count()
        if actual != total:
            raise RuntimeError(
                f"[VectorStore] Count mismatch after upsert: "
                f"expected {total}, got {actual}"
            )
        print(f"[VectorStore] chunks_col: {actual} late chunks verified. ✓")
        return {"status": "ok", "upserted": upserted}

    # ── get full section text (v3 replacement for get_large_chunk) ─────────────

    def get_full_section(self, section_id: str) -> str:
        """
        Fetch the full section text for a given section_id.
        Used by corrective_retrieve() after reranking to get LLM context.

        In v3 the full_section_text is stored in chunk metadata, so this
        is a simple metadata lookup — no embedding needed.

        Args:
            section_id: e.g. "OT_anatomy_ch12_sec0042"

        Returns:
            full section text as a string

        Raises:
            ValueError if no chunk with that section_id is found
        """
        results = self.chunks_col.get(
            where={"section_id": section_id},
            limit=1,
            include=["metadatas"],
        )
        if not results["metadatas"]:
            raise ValueError(f"[VectorStore] Section not found: {section_id}")
        return results["metadatas"][0].get("full_section_text", "")

    # ── query ──────────────────────────────────────────────────────────────────

    def query(
        self,
        query_text:   str,
        n_results:    int  = config.TOP_K_RETRIEVE,
        where_filter: dict = None,
    ) -> list[dict]:
        """
        Full retrieval: embed query → search {domain}_chunks → return results.

        Note: in v3 corrective_retrieve() in retrieval/crag.py directly
        queries self.chunks_col after embedding via ollama. This method
        is retained for backward compatibility and direct use.

        Args:
            query_text:   student's question or concept
            n_results:    how many chunks to return
            where_filter: optional ChromaDB metadata filter

        Returns:
            list of dicts: id, text, distance, section_id, section_title,
                           chapter_num, full_section_text
        """
        query_vec = self._embed_query(query_text)

        kwargs: dict = {
            "query_embeddings": [query_vec],
            "n_results":        n_results,
            "include":          ["documents", "metadatas", "distances"],
        }
        if where_filter:
            kwargs["where"] = where_filter

        raw   = self.chunks_col.query(**kwargs)
        ids   = raw["ids"][0]
        docs  = raw["documents"][0]
        dists = raw["distances"][0]
        metas = raw["metadatas"][0]

        return [
            {
                "id":               ids[i],
                "text":             docs[i],
                "distance":         dists[i],
                "section_id":       metas[i].get("section_id", ""),
                "section_title":    metas[i].get("section_title", ""),
                "chapter_num":      metas[i].get("chapter_num", ""),
                "full_section_text": metas[i].get("full_section_text", docs[i]),
            }
            for i in range(len(ids))
        ]

    # ── weak-topic boost + re-rank ─────────────────────────────────────────────

    def boost_and_rerank_by_weak_topics(
        self,
        results:     list[dict],
        weak_topics: list[str],
        boost:       float = config.WEAK_TOPIC_BOOST,
    ) -> list[dict]:
        """
        Re-rank query() results by boosting chunks whose section_title
        overlaps with the student's known weak topics.

        In v3 weak-topic matching is done on section_title (which is
        already in the result dict from query()) instead of chunk_tags
        from a separate large-chunk lookup.

        Lower distance = higher rank. Boost subtracts from distance so
        weak-topic chunks rise to the top.

        Args:
            results:     output of query() or crag._vector_search()
            weak_topics: list of concept strings from session memory
            boost:       distance reduction for weak-topic matches

        Returns:
            re-sorted results list (same dict shape as input)
        """
        if not weak_topics or not results:
            return results

        weak_set = {t.lower().strip() for t in weak_topics}

        boosted = []
        for r in results:
            r = dict(r)   # shallow copy
            title_lower = r.get("section_title", "").lower()
            if any(topic in title_lower for topic in weak_set):
                r["distance"] -= boost
                r["boosted"]   = True
            else:
                r["boosted"] = False
            boosted.append(r)

        boosted.sort(key=lambda x: x["distance"])
        return boosted

    # ── collection statistics ──────────────────────────────────────────────────

    def get_collection_stats(self) -> dict:
        """Return count for the single chunks collection."""
        embed_dim = 768
        meta_path = os.path.join(config.CHUNKS_DIR, "embed_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                embed_dim = json.load(f).get("dimension", 768)

        return {
            "domain":       self.domain,
            "chunks_count": self.chunks_col.count(),
            "embed_dim":    embed_dim,
        }

    # ── private: embed one query string via Ollama ─────────────────────────────

    def _embed_query(self, text: str) -> list[float]:
        """
        Embed a single query string using the Ollama /api/embed endpoint.
        Uses "search_query: " prefix (asymmetric from stored documents).
        """
        try:
            r = requests.post(
                f"{_OLLAMA_BASE}/api/embed",
                json={
                    "model": _EMBED_MODEL,
                    "input": [_QUERY_PREFIX + text],
                },
                timeout=_EMBED_TIMEOUT,
            )
            r.raise_for_status()
            return r.json()["embeddings"][0]
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "[VectorStore] Cannot reach Ollama for query embedding.\n"
                "  Start it: brew services start ollama"
            )
        except Exception as exc:
            raise RuntimeError(
                f"[VectorStore] Query embedding failed: {exc}"
            )
