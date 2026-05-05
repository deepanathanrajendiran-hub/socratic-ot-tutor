// frontend/components/architecture/TraceView.tsx
"use client";
import { useState, useEffect } from "react";
import { useSession } from "@/lib/useSession";
import { useTraceStream } from "@/lib/useTraceStream";
import { api } from "@/lib/api";
import type { DemoTraceListItem, TraceEvent, TraceStepEvent } from "@/lib/api-types";
import { TracePanel } from "@/components/trace/TracePanel";
import { ChatInput } from "@/components/chat/ChatInput";

type RunMode = "live" | "replay";

/** The original /architecture page — live + replay trace visualization.
 *  Lifted into its own component so /architecture can host two views
 *  (this + a static architecture explainer) under a top-level dropdown. */
export function TraceView() {
  const { sessionId } = useSession();
  const trace = useTraceStream({ sessionId, mode: "socratic" });

  const [runMode, setRunMode] = useState<RunMode>("live");
  const [traces, setTraces] = useState<DemoTraceListItem[]>([]);
  const [replayId, setReplayId] = useState<string>("");
  const [replaySteps, setReplaySteps] = useState<TraceStepEvent[]>([]);
  const [replayResponse, setReplayResponse] = useState<string>("");

  useEffect(() => {
    api.listDemoTraces().then((r) => setTraces(r.traces)).catch(() => {});
  }, []);

  async function loadReplay(id: string) {
    setReplayId(id); setReplaySteps([]); setReplayResponse("");
    if (!id) return;
    const data = await api.getDemoTrace(id);
    const steps = data.events.filter((e): e is TraceStepEvent =>
      (e as TraceEvent).event === "trace");
    const resp = data.events.find((e): e is Extract<TraceEvent, { event: "response" }> =>
      (e as TraceEvent).event === "response");
    setReplaySteps(steps);
    setReplayResponse(resp?.response ?? "");
  }

  const steps    = runMode === "live" ? trace.steps         : replaySteps;
  const response = runMode === "live" ? trace.finalResponse : replayResponse;

  return (
    <div>
      <div className="mb-4 flex flex-wrap items-center gap-3">
        <div className="inline-flex rounded-md border border-slate-300 bg-white p-0.5">
          {(["live", "replay"] as const).map((m) => (
            <button key={m} onClick={() => setRunMode(m)}
              className={`rounded px-3 py-1 text-xs font-medium ${
                runMode === m ? "bg-slate-900 text-white"
                              : "text-slate-700 hover:bg-slate-100"
              }`}>
              {m === "live" ? "Run live" : "Replay canonical"}
            </button>
          ))}
        </div>
        {runMode === "replay" && (
          <select
            value={replayId}
            onChange={(e) => loadReplay(e.target.value)}
            className="rounded-md border border-slate-300 bg-white px-2 py-1 text-sm"
          >
            <option value="">— select —</option>
            {traces.map((t) => (
              <option key={t.id} value={t.id}>{t.label}</option>
            ))}
          </select>
        )}
      </div>

      {runMode === "live" && (
        <div className="mb-6">
          <ChatInput onSend={trace.run} disabled={trace.pending || !sessionId} />
          {trace.error && (
            <div className="mt-2 rounded-md border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-800">
              {trace.error}
            </div>
          )}
        </div>
      )}

      <div className="space-y-3">
        {steps.length === 0 && (
          <div className="rounded-lg border border-dashed border-slate-300 bg-white p-6 text-center text-sm text-slate-500">
            {runMode === "live"
              ? "Type a question above to see the pipeline run."
              : "Select a canonical trace to replay."}
          </div>
        )}
        {steps.map((s, i) => <TracePanel key={i} event={s} />)}
      </div>

      {response && (
        <div className="mt-6 rounded-lg border border-slate-200 bg-white p-4">
          <div className="mb-2 text-xs uppercase tracking-wide text-slate-500">Delivered response</div>
          <div className="text-sm whitespace-pre-wrap">{response}</div>
        </div>
      )}
    </div>
  );
}
