// frontend/components/architecture/MultiDomainSwap.tsx
"use client";
import { Brain, Atom } from "lucide-react";

/* Side-by-side: same backbone, different ChromaDB collection +
 * DOMAIN_CONFIG entry. Demonstrates the architecture is not OT-specific. */

const COMMON_BACKBONE = [
  "rapport_node",
  "manager_agent",
  "retrieval_node (corrective RAG)",
  "response_classifier",
  "teacher_socratic / hint_error_node / teach_node",
  "step_advancer",
  "dean_node (quality gate)",
];

const OT_CONFIG = {
  collection: "OT_anatomy_chunks",
  textbook: "Anatomy & Physiology (OpenStax)",
  chunks: "2,246",
  exam: "NBCOT exam prep",
  exampleConcepts: ["synapse", "median nerve", "cerebellum", "reflex arc"],
  laySwap: '"funny bone" → "ulnar nerve medial epicondyle"',
};

const PHYSICS_CONFIG = {
  collection: "physics_chunks",
  textbook: "University Physics (OpenStax)",
  chunks: "1,141",
  exam: "physics midterm",
  exampleConcepts: ["Newton's third law", "kinetic energy", "torque", "angular momentum"],
  laySwap: '— (no lay synonyms — physics terms are formal)',
};

function DomainCard({
  icon: Icon,
  title,
  accentBg,
  accentBorder,
  accentText,
  config,
}: {
  icon: typeof Brain;
  title: string;
  accentBg: string;
  accentBorder: string;
  accentText: string;
  config: typeof OT_CONFIG;
}) {
  return (
    <div className={`rounded-lg border-2 ${accentBorder} ${accentBg} p-5`}>
      <div className="mb-4 flex items-center gap-3">
        <div className={`rounded-md bg-white p-2 ${accentText}`}>
          <Icon className="h-5 w-5" />
        </div>
        <div className={`text-lg font-semibold ${accentText}`}>{title}</div>
      </div>

      <dl className="space-y-3 text-sm">
        <div>
          <dt className="text-xs uppercase tracking-wide text-slate-500">
            ChromaDB collection
          </dt>
          <dd className="mt-0.5 font-mono text-[13px] text-slate-800">
            {config.collection}
          </dd>
        </div>
        <div>
          <dt className="text-xs uppercase tracking-wide text-slate-500">
            Textbook source
          </dt>
          <dd className="mt-0.5 text-slate-800">{config.textbook}</dd>
        </div>
        <div>
          <dt className="text-xs uppercase tracking-wide text-slate-500">
            Chunks ingested
          </dt>
          <dd className="mt-0.5 text-slate-800">{config.chunks}</dd>
        </div>
        <div>
          <dt className="text-xs uppercase tracking-wide text-slate-500">
            Target exam
          </dt>
          <dd className="mt-0.5 text-slate-800">{config.exam}</dd>
        </div>
        <div>
          <dt className="text-xs uppercase tracking-wide text-slate-500">
            Example concepts
          </dt>
          <dd className="mt-1 flex flex-wrap gap-1.5">
            {config.exampleConcepts.map((c) => (
              <span
                key={c}
                className="rounded-md bg-white px-2 py-0.5 font-mono text-[11px] text-slate-700 ring-1 ring-slate-200"
              >
                {c}
              </span>
            ))}
          </dd>
        </div>
        <div>
          <dt className="text-xs uppercase tracking-wide text-slate-500">
            Lay-term synonym
          </dt>
          <dd className="mt-0.5 font-mono text-[12px] text-slate-700">
            {config.laySwap}
          </dd>
        </div>
      </dl>
    </div>
  );
}

export function MultiDomainSwap() {
  return (
    <div className="space-y-6">
      {/* Shared backbone */}
      <div className="rounded-lg border border-slate-300 bg-slate-50 p-5">
        <div className="mb-3 text-sm font-semibold text-slate-900">
          Shared backbone (identical across domains)
        </div>
        <div className="flex flex-wrap items-center gap-2 text-xs">
          {COMMON_BACKBONE.map((n, i) => (
            <span key={n} className="flex items-center gap-2">
              <code className="rounded bg-white px-2 py-1 font-mono text-[11px] text-slate-800 ring-1 ring-slate-200">
                {n}
              </code>
              {i < COMMON_BACKBONE.length - 1 && (
                <span className="text-slate-400">→</span>
              )}
            </span>
          ))}
        </div>
        <p className="mt-3 text-xs leading-relaxed text-slate-600">
          Every node in the LangGraph above is domain-agnostic. The Socratic
          loop, hint ladder, classifier, and Dean gate all operate on the
          locked <code className="rounded bg-white px-1 font-mono text-[10px] ring-1 ring-slate-200">current_concept</code> string and the
          retrieved chunks — they don&apos;t care whether those chunks describe
          a synapse or a force vector.
        </p>
      </div>

      {/* Per-domain configuration */}
      <div className="grid gap-4 md:grid-cols-2">
        <DomainCard
          icon={Brain}
          title="OT_anatomy"
          accentBg="bg-rose-50"
          accentBorder="border-rose-200"
          accentText="text-rose-900"
          config={OT_CONFIG}
        />
        <DomainCard
          icon={Atom}
          title="physics"
          accentBg="bg-indigo-50"
          accentBorder="border-indigo-200"
          accentText="text-indigo-900"
          config={PHYSICS_CONFIG}
        />
      </div>

      {/* What it takes to add a domain */}
      <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-5">
        <div className="mb-3 text-sm font-semibold text-emerald-900">
          To add a new domain
        </div>
        <ol className="ml-5 list-decimal space-y-1.5 text-sm text-emerald-900">
          <li>
            Run <code className="rounded bg-white px-1.5 py-0.5 font-mono text-[11px] ring-1 ring-emerald-200">ingest/run_ingest_pipeline.py</code> on the new textbook
            PDF — produces a ChromaDB collection of section-anchored chunks.
          </li>
          <li>
            Add an entry to <code className="rounded bg-white px-1.5 py-0.5 font-mono text-[11px] ring-1 ring-emerald-200">DOMAIN_CONFIG</code> in <code className="rounded bg-white px-1.5 py-0.5 font-mono text-[11px] ring-1 ring-emerald-200">backend/config.py</code> — collection name,
            system context, target exam, example concepts.
          </li>
          <li>
            (Optional) Add lay-term synonyms to <code className="rounded bg-white px-1.5 py-0.5 font-mono text-[11px] ring-1 ring-emerald-200">ot_synonyms.py</code> if
            students use vocabulary that differs from the textbook.
          </li>
          <li>
            Add the domain string to the frontend domain dropdown.
          </li>
        </ol>
        <p className="mt-3 text-xs leading-relaxed text-emerald-800">
          No graph code changes. No prompt rewrites. The retrieval, classifier,
          and Dean prompts all read from the locked concept and chunk content;
          they are vocabulary-blind to the underlying subject.
        </p>
      </div>
    </div>
  );
}
