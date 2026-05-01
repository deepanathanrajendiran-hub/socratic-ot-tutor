// frontend/components/chat/PipelineIndicator.tsx
//
// Live "what stage are we in" strip that renders above the typing
// bubble while the graph executes. Shows each pipeline stage as it
// fires (concept extraction → retrieval → classifier → thinking →
// generation → quality check), with the active stage pulsing in coral
// and completed stages collapsed to checkmarks.
//
// Educational value: makes the architecture visible to the user
// without forcing them to open /architecture.
"use client";
import { Fragment, useEffect, useState } from "react";
import { Check, X } from "lucide-react";
import type { PipelineStage } from "@/lib/useChatStream";

// Friendly labels keyed off backend step names. Falls through to the
// raw name if we don't have a mapping (so a new node added to the graph
// shows up immediately, just less prettily).
const LABEL: Record<string, string> = {
  concept_extraction: "Reading question",
  retrieval:          "Retrieving",
  classifier:         "Classifying",
  generation:         "Writing",
  thinking:           "Thinking",
  dean:               "Quality check",
  study:              "Answering",
  vlm:                "Vision",
};

/** Format an elapsed-duration in ms as a compact, stable string.
 *    < 1s  → "0.6s"
 *    < 60s → "12.3s"
 *    >= 60s → "1m12s"
 *  Floored to one decimal so the ticker doesn't jitter every frame. */
function fmtDuration(ms: number): string {
  if (ms < 0) ms = 0;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const mm = Math.floor(ms / 60_000);
  const ss = Math.floor((ms % 60_000) / 1000);
  return `${mm}m${ss.toString().padStart(2, "0")}s`;
}

export function PipelineIndicator({
  stages,
  visible,
}: {
  stages: PipelineStage[];
  /** When false, the strip fades out (kept mounted for ~200ms after the
   *  parent flips to !pending so it doesn't pop). */
  visible: boolean;
}) {
  // Live ticker: while any stage is active we re-render every 200 ms so
  // the elapsed-duration text on the pulsing pill counts up. Once every
  // stage is settled (done/error) the ticker stops to avoid wasted work.
  const hasActive = stages.some((s) => s.status === "active");
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!hasActive) return;
    const id = setInterval(() => setNow(Date.now()), 200);
    return () => clearInterval(id);
  }, [hasActive]);

  if (stages.length === 0) return null;
  return (
    <div
      className={`mb-2 transition-opacity duration-200 ${
        visible ? "opacity-100" : "opacity-0"
      }`}
      aria-hidden={!visible}
    >
      <div className="flex w-full flex-wrap items-center gap-1.5 text-[11px] text-ivory-500">
        {stages.map((s, i) => {
          const label = LABEL[s.name] ?? s.name;
          const isActive = s.status === "active";
          const isDone   = s.status === "done";
          const isError  = s.status === "error";
          // Frozen elapsed for settled stages, live count for the
          // active one. Hidden while < 100ms to avoid a "0.0s" flash
          // that flickers off the moment a fast stage completes.
          const endTs = s.completedAt ?? (isActive ? now : undefined);
          const elapsedMs =
            endTs !== undefined ? endTs - s.startedAt : undefined;
          const showDur = elapsedMs !== undefined && elapsedMs >= 100;
          return (
            <Fragment key={`${s.name}-${i}`}>
              <span
                className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 transition-colors ${
                  isActive
                    ? "border border-coral-200 bg-coral-50 text-coral-800"
                    : isError
                    ? "border border-rose-200 bg-rose-50 text-rose-800"
                    : isDone
                    ? "text-ivory-600"
                    : "text-ivory-400"
                }`}
                title={`${label} — ${s.status}${
                  showDur ? ` · ${fmtDuration(elapsedMs!)}` : ""
                }`}
              >
                {isActive && (
                  <span
                    aria-hidden
                    className="h-1.5 w-1.5 rounded-full bg-coral-500 animate-pulse"
                  />
                )}
                {isDone  && <Check size={11} className="shrink-0" />}
                {isError && <X size={11} className="shrink-0" />}
                <span>{label}</span>
                {showDur && (
                  <span
                    className={`tabular-nums ${
                      isActive
                        ? "text-coral-600/80"
                        : isError
                        ? "text-rose-600/80"
                        : "text-ivory-400"
                    }`}
                  >
                    · {fmtDuration(elapsedMs!)}
                  </span>
                )}
              </span>
              {i < stages.length - 1 && (
                <span className="text-ivory-300" aria-hidden>→</span>
              )}
            </Fragment>
          );
        })}
      </div>
    </div>
  );
}
