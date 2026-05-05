// frontend/components/architecture/ArchitectureView.tsx
"use client";
import Link from "next/link";
import { ArrowRight, Network, Search, Layers } from "lucide-react";
import { StateMachineDiagram } from "./StateMachineDiagram";
import { CragPipeline } from "./CragPipeline";
import { MultiDomainSwap } from "./MultiDomainSwap";

/* One-page scrollable narrative for an academic reviewer.
 * Three sections + intro + CTA. */

function SectionHeader({
  icon: Icon,
  num,
  title,
  blurb,
  id,
}: {
  icon: typeof Network;
  num: string;
  title: string;
  blurb: string;
  id: string;
}) {
  return (
    <div id={id} className="mb-6 scroll-mt-20">
      <div className="mb-2 flex items-baseline gap-3">
        <span className="text-xs font-semibold uppercase tracking-widest text-slate-500">
          {num}
        </span>
        <h2 className="flex items-center gap-2 text-2xl font-semibold text-slate-900">
          <Icon className="h-6 w-6 text-slate-700" />
          {title}
        </h2>
      </div>
      <p className="max-w-3xl text-sm leading-relaxed text-slate-600">{blurb}</p>
    </div>
  );
}

export function ArchitectureView() {
  return (
    <div className="space-y-16 pb-16">
      {/* ── Hero ─────────────────────────────────────────────────────── */}
      <header className="space-y-4">
        <h1 className="font-serif text-4xl tracking-tight text-slate-900">
          A Socratic teaching agent that doesn&apos;t leak the answer
        </h1>
        <p className="max-w-3xl text-base leading-relaxed text-slate-600">
          LLMs default to answering. A good tutor refuses to — they pull
          knowledge out of the student instead. This system enforces that
          discipline architecturally: a LangGraph state machine routes every
          turn through a hint ladder, a quality gate scrubs answer leaks
          before delivery, and a corrective retrieval pipeline keeps the
          tutor grounded in the textbook for any domain you point it at.
        </p>
        <nav className="flex flex-wrap gap-2 text-sm">
          {[
            { id: "state-machine", label: "1. State machine" },
            { id: "crag",          label: "2. Corrective RAG" },
            { id: "multi-domain",  label: "3. Multi-domain" },
          ].map(({ id, label }) => (
            <a
              key={id}
              href={`#${id}`}
              className="rounded-full border border-slate-200 bg-white px-3 py-1 text-slate-700 hover:border-slate-300 hover:bg-slate-50"
            >
              {label}
            </a>
          ))}
        </nav>
      </header>

      {/* ── § 1: State machine ────────────────────────────────────────── */}
      <section>
        <SectionHeader
          icon={Network}
          num="01"
          title="LangGraph state machine"
          blurb="Each turn flows through a directed graph of nodes. Routing decisions encode the pedagogy: a 'correct' answer goes to the mastery menu, an 'incorrect' attempt loops to a hint with progressively more specific scaffolding, and an 'idk' counter triggers reveal after two consecutive give-ups. Every generated draft is gated by Dean before the student sees it."
          id="state-machine"
        />
        <StateMachineDiagram />
        <p className="mt-3 text-xs italic text-slate-500">
          Hover or click any node above to see its source file and what it
          does. Edge colors map to classifier labels: green = correct, amber =
          incorrect/idk, red = reveal at SOCRATIC_TURN_GATE, blue = turn-0
          opener.
        </p>
      </section>

      {/* ── § 2: Corrective RAG ────────────────────────────────────────── */}
      <section>
        <SectionHeader
          icon={Search}
          num="02"
          title="Corrective Retrieval-Augmented Generation"
          blurb="Cosine similarity finds candidates by lexical overlap, but the chunk that literally answers the question often isn't ranked first — it sits at rank 11-13 because of vocabulary mismatch between student language and textbook prose. A cross-encoder reranks the top-15 by semantic (query, chunk) score and reliably elevates the textbook-defining chunk to the top. CRAG's LLM judge sits on top of all of it as a sanity check."
          id="crag"
        />
        <CragPipeline />
      </section>

      {/* ── § 3: Multi-domain ─────────────────────────────────────────── */}
      <section>
        <SectionHeader
          icon={Layers}
          num="03"
          title="Multi-domain — same backbone, swap the textbook"
          blurb="The Socratic loop, hint ladder, and quality gate are vocabulary-blind. They operate on a locked concept name and retrieved chunks — agnostic to whether those chunks describe a synapse or a force vector. Two domains are wired up today; adding a third is four small steps."
          id="multi-domain"
        />
        <MultiDomainSwap />
      </section>

      {/* ── CTA ────────────────────────────────────────────────────────── */}
      <section className="rounded-2xl border border-slate-300 bg-gradient-to-br from-slate-50 to-white p-8">
        <h3 className="text-xl font-semibold text-slate-900">
          Try it for yourself
        </h3>
        <p className="mt-2 max-w-2xl text-sm leading-relaxed text-slate-600">
          The live tutor runs on Cloud Run with persistent Postgres
          checkpointing. Pick a domain, ask a question — the system never
          gives you the answer until you&apos;ve worked toward it.
        </p>
        <div className="mt-5 flex flex-wrap gap-3">
          <Link
            href="/tutor"
            className="inline-flex items-center gap-2 rounded-full bg-slate-900 px-5 py-2.5 text-sm font-medium text-white hover:bg-slate-800"
          >
            Open the tutor
            <ArrowRight className="h-4 w-4" />
          </Link>
          <Link
            href="/architecture"
            className="inline-flex items-center gap-2 rounded-full border border-slate-300 bg-white px-5 py-2.5 text-sm font-medium text-slate-800 hover:bg-slate-50"
          >
            See a real-time pipeline trace
          </Link>
        </div>
      </section>
    </div>
  );
}
