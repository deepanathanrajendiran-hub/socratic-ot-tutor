// frontend/lib/api.ts
import { BACKEND_URL } from "./env";
import type {
  ChatRequest, ChatEvent, TraceEvent,
  SessionState, DemoTraceListItem, DemoTracePayload,
} from "./api-types";
import { createParser } from "eventsource-parser";

async function jsonGet<T>(path: string): Promise<T> {
  const res = await fetch(`${BACKEND_URL}${path}`);
  if (!res.ok) throw new Error(`GET ${path} → ${res.status}`);
  return res.json() as Promise<T>;
}

async function jsonPost<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${BACKEND_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`POST ${path} → ${res.status}`);
  return res.json() as Promise<T>;
}

async function jsonRequest<T>(
  path: string, method: string, body?: unknown,
): Promise<T> {
  const res = await fetch(`${BACKEND_URL}${path}`, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`${method} ${path} → ${res.status}`);
  return res.json() as Promise<T>;
}

export interface SessionListItem {
  id:          string;
  title:       string | null;
  pinned:      boolean;
  created_at:  string;
  last_active: string;
}

/** Generic SSE consumer using eventsource-parser. Parses `data: {...}` lines
 *  into typed objects via the supplied decoder. */
async function* sseStream<T>(
  path: string, body: unknown, decode: (raw: string) => T,
): AsyncGenerator<T> {
  const res = await fetch(`${BACKEND_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok || !res.body) throw new Error(`SSE ${path} → ${res.status}`);

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  const queue: T[] = [];
  let resolveNext: ((v: T | undefined) => void) | null = null;
  let done = false;

  const parser = createParser({
    onEvent: (ev) => {
      try {
        const parsed = decode(ev.data);
        if (resolveNext) { resolveNext(parsed); resolveNext = null; }
        else queue.push(parsed);
      } catch { /* ignore malformed events */ }
    },
  });

  // Pump reader → parser
  (async () => {
    for (;;) {
      const { value, done: streamDone } = await reader.read();
      if (streamDone) {
        done = true;
        const r = resolveNext as ((v: T | undefined) => void) | null;
        if (r !== null) { resolveNext = null; r(undefined); }
        break;
      }
      parser.feed(value);
    }
  })();

  while (true) {
    if (queue.length) yield queue.shift()!;
    else if (done) return;
    else {
      const v = await new Promise<T | undefined>((r) => { resolveNext = r; });
      if (v === undefined) return;
      yield v;
    }
  }
}

// ── REST ────────────────────────────────────────────────────────────────────

export const api = {
  health:        () => jsonGet<{ status: string; version: string }>("/health"),
  createSession: () => jsonPost<{ session_id: string; created_at: string }>("/sessions"),
  getSession:    (id: string, userId?: string | null) =>
                   jsonGet<SessionState>(
                     `/sessions/${id}` +
                     (userId ? `?user_id=${encodeURIComponent(userId)}` : ""),
                   ),
  getUserWeakTopics: (userId: string) =>
                   jsonGet<{ user_id: string; weak_topics: string[] }>(
                     `/users/${encodeURIComponent(userId)}/weak_topics`,
                   ),
  truncateMessages: (sessionId: string, fromIndex: number) =>
                   jsonRequest<{ session_id: string; removed: number; remaining: number }>(
                     `/sessions/${encodeURIComponent(sessionId)}/messages/from/${fromIndex}`,
                     "DELETE",
                   ),
  listSessions:  () => jsonGet<{ sessions: SessionListItem[] }>("/sessions"),
  patchSession:  (id: string, patch: { title?: string; pinned?: boolean }) =>
                   jsonRequest<SessionListItem>(`/sessions/${id}`, "PATCH", patch),
  deleteSession: (id: string) =>
                   jsonRequest<{ session_id: string; deleted: boolean }>(
                     `/sessions/${id}`, "DELETE"),
  listDemoTraces: () => jsonGet<{ traces: DemoTraceListItem[] }>("/demo/traces"),
  getDemoTrace:  (id: string) => jsonGet<DemoTracePayload>(`/demo/traces/${id}`),

  chat: (req: ChatRequest) =>
    sseStream<ChatEvent>("/chat", req, (raw) => JSON.parse(raw) as ChatEvent),

  trace: (req: ChatRequest) =>
    sseStream<TraceEvent>("/chat/trace", req, (raw) => JSON.parse(raw) as TraceEvent),
};
