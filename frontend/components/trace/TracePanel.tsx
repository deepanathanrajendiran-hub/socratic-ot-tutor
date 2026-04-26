// frontend/components/trace/TracePanel.tsx
"use client";
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import type { TraceStepEvent } from "@/lib/api-types";

const STEP_LABEL: Record<string, string> = {
  concept_extraction: "1 · Concept Extraction",
  retrieval:          "2 · Retrieval",
  classifier:         "3 · Response Classifier",
  generation:         "4 · Generation",
  dean:               "5 · Dean Quality Gate",
  study:              "Study mode answerer",
};

export function TracePanel({ event }: { event: TraceStepEvent }) {
  const [open, setOpen] = useState(true);
  const label = STEP_LABEL[event.step] ?? event.step;
  return (
    <motion.section
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className="overflow-hidden rounded-lg border border-slate-200 bg-white shadow-sm"
    >
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between gap-4 px-4 py-3 text-left hover:bg-slate-50"
      >
        <div className="flex items-baseline gap-3">
          <span className="font-medium">{label}</span>
          {event.model && <span className="text-xs text-slate-500">model: {event.model}</span>}
        </div>
        <span className="text-xs tabular-nums text-slate-500">{event.duration_ms} ms</span>
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0 }} animate={{ height: "auto" }} exit={{ height: 0 }}
            className="overflow-hidden border-t border-slate-100"
          >
            <div className="grid gap-4 p-4 md:grid-cols-2">
              <div>
                <div className="mb-1 text-xs uppercase tracking-wide text-slate-500">Input</div>
                <pre className="overflow-x-auto rounded bg-slate-50 p-2 text-xs">
                  {JSON.stringify(event.input, null, 2)}
                </pre>
              </div>
              <div>
                <div className="mb-1 text-xs uppercase tracking-wide text-slate-500">Output</div>
                <pre className="overflow-x-auto rounded bg-slate-50 p-2 text-xs">
                  {JSON.stringify(event.output, null, 2)}
                </pre>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.section>
  );
}
