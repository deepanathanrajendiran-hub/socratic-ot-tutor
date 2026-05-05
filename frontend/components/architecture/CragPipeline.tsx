// frontend/components/architecture/CragPipeline.tsx
"use client";
import { useState } from "react";
import { ChevronRight } from "lucide-react";

/* Vertical flow: query → expand → embed → cosine → rerank → eval → top-3.
 * Each step is clickable; clicking expands a detail panel underneath
 * with the production logs/numbers from the synapse trace we debugged. */

type Step = {
  id: string;
  title: string;
  oneLiner: string;
  detail: string;
  example?: string;
};

const STEPS: Step[] = [
  {
    id: "query",
    title: "1. Student query",
    oneLiner: "Raw natural-language question from the chat input",
    detail:
      "The query is whatever the student typed. The retrieval node uses build_turn_query() to optionally enrich it on follow-up turns — appending domain-specific facets like 'anatomy location structure' (turn 1) or 'function clinical significance' (turn 2+). Suffixes are domain-aware so physics queries don't get OT-specific noise.",
    example: '"What\'s the gap between neurons?"',
  },
  {
    id: "expand",
    title: "2. Synonym expansion",
    oneLiner: "OT lay-term → anatomical-term lookup (zero LLM cost)",
    detail:
      "ot_synonyms.OT_SYNONYMS is a curated dict mapping student vocabulary to textbook anatomical terms. 'funny bone' → 'ulnar nerve medial epicondyle'. 'wrist drop' → 'radial nerve extensor paralysis'. Pure substring match, runs in microseconds, handles ~80% of vocabulary mismatch without any API call.",
    example: 'no match → "What\'s the gap between neurons?"',
  },
  {
    id: "embed",
    title: "3. Embed query",
    oneLiner: "nomic-embed-text-v1.5 (768 dims, MTEB top-tier)",
    detail:
      "Loaded on first request via sentence-transformers, then cached in memory. EMBED_BACKEND=transformers in production; can fall back to ollama for local dev. The embedding goes against the ChromaDB collection for the active domain (OT_anatomy_chunks or physics_chunks).",
  },
  {
    id: "cosine",
    title: "4. Cosine top-15",
    oneLiner: "ChromaDB approximate-nearest-neighbor lookup",
    detail:
      "TOP_K_RETRIEVE = 15. Pulls the 15 closest chunks by cosine similarity. We widened from 10 → 15 so the cross-encoder gets enough headroom — the right chunk often sits at cosine rank 11-13 for vocabulary-mismatched queries (e.g., the 'synapse is the gap between nerve cells' chunk lives at cosine rank ~12 for the synapse query).",
  },
  {
    id: "rerank",
    title: "5. Cross-encoder rerank",
    oneLiner: "MS-MARCO MiniLM-L-6-v2 — semantic (query, chunk) scoring",
    detail:
      "The cosine stage is fast but lexical. The cross-encoder reads the full (query, chunk) pair and scores semantic relevance. For the synapse trace: cosine rank #12 chunk gets reranker score +5.504 while cosine rank #1 gets −8.309 — the cross-encoder cleanly elevates the textbook definition over generic 'nervous system' intro chunks.",
    example: "rank #12 cosine → rank #1 reranked (+5.504 vs −8.309)",
  },
  {
    id: "eval",
    title: "6. CRAG self-correction",
    oneLiner: "Self-correcting layer — LLM rewrites the query when retrieval is shaky",
    detail:
      "This is the step that distinguishes Corrective RAG from plain RAG. Haiku reads the (query, chunks) pair and returns a verdict + score. The high-value branch is AMBIGUOUS: the judge proposes a refined query, the system re-runs the whole retrieval against that, and if the second pass scores higher we ship those chunks instead. It's the only stage that can rescue a vocabulary-mismatched question the cross-encoder couldn't fix — without CRAG, mediocre retrieval would just ship as-is. CORRECT ships unchanged. INCORRECT we bypass at the chunk-pick step (the cross-encoder is more reliable as a chunk ranker than Haiku is as a judge), but the verdict is still logged to the dashboard as an observability signal.",
    example: 'AMBIGUOUS → refined query → re-search → swap in if score(new) > score(orig)',
  },
  {
    id: "deliver",
    title: "7. Top-3 chunks → teacher",
    oneLiner: "Full section text + section-id dedupe",
    detail:
      "The reranker's top-3 reranked chunks are deduped by section_id (multiple anchors in one section don't ship as separate chunks) and their full section text becomes the teacher node's grounding context. No raw chunk text reaches the student — only the teacher's Socratic phrasing.",
  },
];

export function CragPipeline() {
  const [openId, setOpenId] = useState<string | null>(null);

  return (
    <div className="space-y-2">
      {STEPS.map((step, i) => {
        const isOpen = openId === step.id;
        const isLast = i === STEPS.length - 1;
        return (
          <div key={step.id}>
            <button
              onClick={() => setOpenId(isOpen ? null : step.id)}
              className={`group flex w-full items-start gap-3 rounded-lg border px-4 py-3 text-left transition-all ${
                isOpen
                  ? "border-sky-300 bg-sky-50"
                  : "border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50"
              }`}
            >
              <div className="flex-shrink-0 pt-0.5">
                <ChevronRight
                  className={`h-4 w-4 transition-transform ${
                    isOpen ? "rotate-90 text-sky-700" : "text-slate-400"
                  }`}
                />
              </div>
              <div className="flex-1 min-w-0">
                <div
                  className={`text-sm font-semibold ${
                    isOpen ? "text-sky-900" : "text-slate-900"
                  }`}
                >
                  {step.title}
                </div>
                <div className="mt-0.5 text-xs text-slate-600">
                  {step.oneLiner}
                </div>
              </div>
            </button>
            {isOpen && (
              <div className="ml-7 mt-2 rounded-lg border border-sky-100 bg-sky-50/50 p-4 text-sm text-slate-700">
                <p className="leading-relaxed">{step.detail}</p>
                {step.example && (
                  <div className="mt-3 rounded-md bg-white px-3 py-2 font-mono text-xs text-slate-700 ring-1 ring-slate-200">
                    {step.example}
                  </div>
                )}
              </div>
            )}
            {!isLast && (
              <div className="ml-6 h-3 border-l-2 border-dashed border-slate-200" />
            )}
          </div>
        );
      })}
    </div>
  );
}
