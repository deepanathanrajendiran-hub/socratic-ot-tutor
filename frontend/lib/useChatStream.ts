// frontend/lib/useChatStream.ts
"use client";
import { useState, useCallback } from "react";
import { api } from "./api";
import type { ChatMessage, ChatStepName, Mode } from "./api-types";

export type PipelineStageStatus = "active" | "done" | "error";

export interface PipelineStage {
  /** Backend step name, e.g. "concept_extraction", "retrieval", "thinking". */
  name: string;
  status: PipelineStageStatus;
  /** ms timestamp when the stage first started — used to compute duration. */
  startedAt: number;
  /** ms timestamp when the stage finished (done or error). Undefined while
   *  the stage is still active. UI freezes the displayed duration once
   *  this is set. */
  completedAt?: number;
}

export interface UseChatStreamReturn {
  messages: ChatMessage[];
  pending: boolean;
  error: string | null;
  /** Send a text message, optionally with a base64-encoded image (no
   *  data: prefix). When imageB64 is set the backend routes the turn
   *  through the multimodal vlm_node. */
  send: (text: string, imageB64?: string) => Promise<void>;
  setMessages: (m: ChatMessage[]) => void;
  turnCount: number;
  /** Last `step` event with status="start" still in flight, or null. Drives
   *  the typing-bubble status label. */
  currentStep: ChatStepName | string | null;
  /** Ordered list of pipeline stages as they unfold this turn. Reset on
   *  every send. UI uses this to render the "what stage are we in" strip. */
  pipelineStages: PipelineStage[];
}

export function useChatStream(opts: {
  sessionId: string | null;
  mode: Mode;
  /** Stable per-browser id from useUser. When set, the request includes
   *  it so the cross-session memory layer can attribute facts. */
  userId?: string | null;
  /** Domain selector — "OT_anatomy" | "physics". The backend reads this
   *  from the request and overrides config.DOMAIN per-request, so users
   *  can switch domain without redeploying. */
  domain?: string;
}): UseChatStreamReturn {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [pending, setPending]   = useState(false);
  const [error, setError]       = useState<string | null>(null);
  const [turnCount, setTurn]    = useState(0);
  const [currentStep, setStep]  = useState<ChatStepName | string | null>(null);
  const [pipelineStages, setStages] = useState<PipelineStage[]>([]);

  const send = useCallback(async (text: string, imageB64?: string) => {
    if (!opts.sessionId) { setError("no session"); return; }
    // Surface "[image attached]" in the visible thread so the user can
    // see they uploaded something, even if their text was just the
    // default "What is this structure?".
    const userContent = imageB64
      ? `${text}${text ? "\n" : ""}🖼️ [image attached]`
      : text;
    const baseline: ChatMessage[] = [
      ...messages,
      { role: "user", content: userContent },
    ];
    setMessages(baseline);
    setPending(true);
    setError(null);
    setStep(null);
    setStages([]);  // fresh pipeline for this turn

    // Local mutable view of the assistant bubble. We mirror it back into
    // React state on every change so the UI re-renders.
    let assistant = "";
    let assistantStarted = false;
    const commit = () => {
      if (!assistantStarted) {
        assistantStarted = true;
        setMessages([...baseline, { role: "assistant", content: assistant }]);
      } else {
        setMessages([...baseline, { role: "assistant", content: assistant }]);
      }
    };

    try {
      // The wire payload sends the raw text (no image marker) and the
      // base64 separately — backend reads image_b64 from the request, not
      // the message content. The bare text becomes the conversation log.
      const wireBaseline: ChatMessage[] = [
        ...messages,
        { role: "user", content: text },
      ];
      for await (const ev of api.chat({
        messages: wireBaseline,
        session_id: opts.sessionId,
        mode: opts.mode,
        image_b64: imageB64,
        user_id: opts.userId ?? undefined,
        domain: opts.domain,
      })) {
        // ── New streaming envelope ─────────────────────────────────────────
        if ("event" in ev && ev.event === "token") {
          assistant += ev.delta;
          // First token clears the step label so the bubble shows real text.
          if (currentStep !== null) setStep(null);
          commit();
          continue;
        }
        if ("event" in ev && ev.event === "step") {
          if (ev.status === "start") {
            setStep(ev.step);
            // Add (or update) this stage in the pipeline strip.
            setStages((prev) => {
              const existing = prev.findIndex((s) => s.name === ev.step);
              const stage: PipelineStage = {
                name: ev.step,
                status: "active",
                startedAt: Date.now(),
              };
              if (existing >= 0) {
                // Same step firing twice (e.g. Dean revision) — refresh
                // status to active.
                const next = [...prev];
                next[existing] = stage;
                return next;
              }
              return [...prev, stage];
            });
          } else if (ev.status === "done" || ev.status === "error") {
            // Only clear the inline label if this is the step we were showing.
            setStep(prev => (prev === ev.step ? null : prev));
            setStages((prev) => {
              const idx = prev.findIndex((s) => s.name === ev.step);
              if (idx < 0) return prev;
              const next = [...prev];
              next[idx] = {
                ...next[idx],
                status: ev.status === "error" ? "error" : "done",
                completedAt: Date.now(),
              };
              return next;
            });
          }
          continue;
        }
        if ("event" in ev && ev.event === "replace") {
          assistant = ev.response;
          commit();
          continue;
        }
        if ("event" in ev && ev.event === "done") {
          if (typeof ev.turn_count === "number") setTurn(ev.turn_count);
          continue;
        }
        if ("event" in ev && ev.event === "error") {
          setError(ev.error);
          continue;
        }

        // ── Legacy frames (older backends) ─────────────────────────────────
        if ("response" in ev && typeof ev.response === "string") {
          assistant = ev.response;
          commit();
        } else if ("done" in ev) {
          if (typeof ev.turn_count === "number") setTurn(ev.turn_count);
        } else if ("error" in ev) {
          setError(ev.error);
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "stream error");
    } finally {
      setPending(false);
      setStep(null);
      // Keep `pipelineStages` populated so the UI can briefly fade them
      // out after completion. They're cleared on the next send.
    }
  }, [messages, opts.sessionId, opts.mode, opts.userId, opts.domain, currentStep]);

  return {
    messages, pending, error, send, setMessages,
    turnCount, currentStep, pipelineStages,
  };
}
