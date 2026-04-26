// frontend/lib/useChatStream.ts
"use client";
import { useState, useCallback } from "react";
import { api } from "./api";
import type { ChatMessage, Mode } from "./api-types";

export interface UseChatStreamReturn {
  messages: ChatMessage[];
  pending: boolean;
  error: string | null;
  send: (text: string) => Promise<void>;
  setMessages: (m: ChatMessage[]) => void;
  turnCount: number;
}

export function useChatStream(opts: {
  sessionId: string | null;
  mode: Mode;
}): UseChatStreamReturn {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [pending, setPending]   = useState(false);
  const [error, setError]       = useState<string | null>(null);
  const [turnCount, setTurn]    = useState(0);

  const send = useCallback(async (text: string) => {
    if (!opts.sessionId) { setError("no session"); return; }
    const next: ChatMessage[] = [...messages, { role: "user", content: text }];
    setMessages(next);
    setPending(true);
    setError(null);
    try {
      for await (const ev of api.chat({
        messages: next, session_id: opts.sessionId, mode: opts.mode,
      })) {
        if ("response" in ev && typeof ev.response === "string") {
          setMessages([...next, { role: "assistant", content: ev.response }]);
        } else if ("error" in ev) {
          setError(ev.error);
        } else if ("done" in ev) {
          if (typeof ev.turn_count === "number") setTurn(ev.turn_count);
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "stream error");
    } finally {
      setPending(false);
    }
  }, [messages, opts.sessionId, opts.mode]);

  return { messages, pending, error, send, setMessages, turnCount };
}
