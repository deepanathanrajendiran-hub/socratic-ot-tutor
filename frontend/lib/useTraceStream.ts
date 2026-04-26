// frontend/lib/useTraceStream.ts
"use client";
import { useState, useCallback } from "react";
import { api } from "./api";
import type { Mode, TraceStepEvent } from "./api-types";

export interface UseTraceStreamReturn {
  steps: TraceStepEvent[];
  finalState: Record<string, unknown> | null;
  finalResponse: string;
  pending: boolean;
  error: string | null;
  run: (text: string) => Promise<void>;
  reset: () => void;
}

export function useTraceStream(opts: {
  sessionId: string | null;
  mode: Mode;
}): UseTraceStreamReturn {
  const [steps, setSteps]     = useState<TraceStepEvent[]>([]);
  const [finalState, setFs]   = useState<Record<string, unknown> | null>(null);
  const [finalResponse, setR] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError]     = useState<string | null>(null);

  function reset() { setSteps([]); setFs(null); setR(""); setError(null); }

  const run = useCallback(async (text: string) => {
    if (!opts.sessionId) { setError("no session"); return; }
    reset();
    setPending(true);
    try {
      for await (const ev of api.trace({
        messages: [{ role: "user", content: text }],
        session_id: opts.sessionId, mode: opts.mode,
      })) {
        if (ev.event === "trace") setSteps((p) => [...p, ev]);
        else if (ev.event === "state") setFs(ev.state);
        else if (ev.event === "response") setR(ev.response);
        else if (ev.event === "error") setError(ev.message);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "trace error");
    } finally {
      setPending(false);
    }
  }, [opts.sessionId, opts.mode]);

  return { steps, finalState, finalResponse, pending, error, run, reset };
}
