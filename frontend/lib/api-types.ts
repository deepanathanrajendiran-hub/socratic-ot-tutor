// frontend/lib/api-types.ts
//
// Mirrors the backend's Pydantic models. Keep this in sync with
// backend/api/main.py manually (low-traffic file; small surface).

export type Mode = "socratic" | "study";

export type ChatRole = "user" | "assistant";

export interface ChatMessage {
  role: ChatRole;
  content: string;
}

export interface ChatRequest {
  messages: ChatMessage[];
  session_id: string;
  mode: Mode;
  domain?: string;
  /** Optional base64-encoded anatomical image. Strip the "data:..;base64,"
   *  prefix before sending; the backend handles bare base64.
   *  When present, the graph routes through vlm_node (Sonnet vision)
   *  for identification + Socratic opener. */
  image_b64?: string;
  /** Stable per-browser user id (from localStorage via useUser). Used by
   *  the cross-session memory layer (mem0) to scope facts to a user
   *  across many sessions. Ignored when MEMORY_BACKEND=sqlite. */
  user_id?: string;
}

// SSE event payloads from POST /chat
//
// New token-streaming envelope (Apr 2026):
//   step    — node lifecycle, drives the typing-bubble status label
//   token   — append delta to the live assistant bubble
//   replace — overwrite the assistant bubble (used by length-retry,
//             Dean revisions, deterministic strips, fallback_scaffold)
//   done    — finalize; turn_count is authoritative
//   error   — graph-level failure
//
// The legacy {response} / {done: true} / {error} frames remain in this
// union so older backends or replayed traces still type-check.
export interface ChatTokenEvent {
  event: "token";
  delta: string;
}
export type ChatStepName =
  | "concept_extraction"
  | "retrieval"
  | "classifier"
  | "generation"
  | "dean"
  | "study";
export interface ChatStepEvent {
  event: "step";
  step: ChatStepName | string;
  status: "start" | "done" | "error";
}
export interface ChatReplaceEvent {
  event: "replace";
  response: string;
}
export interface ChatDoneEnvelopeEvent {
  event: "done";
  turn_count: number;
}
export interface ChatErrorEnvelopeEvent {
  event: "error";
  error: string;
}

// Legacy (pre-streaming) frames — kept so the union is backwards-compatible.
export interface ChatResponseEvent {
  response: string;
}
export interface ChatDoneEvent {
  done: true;
  turn_count: number;
}
export interface ChatErrorEvent {
  error: string;
}

export type ChatEvent =
  | ChatTokenEvent
  | ChatStepEvent
  | ChatReplaceEvent
  | ChatDoneEnvelopeEvent
  | ChatErrorEnvelopeEvent
  | ChatResponseEvent
  | ChatDoneEvent
  | ChatErrorEvent;

// SSE event payloads from POST /chat/trace
export interface TraceStepEvent {
  event: "trace";
  step: "concept_extraction" | "retrieval" | "classifier" |
        "generation" | "dean" | "study";
  input: Record<string, unknown>;
  output: Record<string, unknown>;
  duration_ms: number;
  model?: string;
}
export interface TraceStateEvent {
  event: "state";
  state: Record<string, unknown>;
}
export interface TraceResponseEvent {
  event: "response";
  response: string;
}
export interface TraceDoneEvent { event: "done" }
export interface TraceErrorEvent { event: "error"; message: string }
export type TraceEvent =
  | TraceStepEvent | TraceStateEvent | TraceResponseEvent
  | TraceDoneEvent | TraceErrorEvent;

// Session payload
export interface SessionState {
  session_id: string;
  mode: Mode;
  turn_count: number;
  current_concept: string;
  weak_topics: string[];
  student_phase: string;
  concept_mastered: boolean;
  mastery_level: string;
  // Debug fields — surfaced in the sidebar's Debug panel.
  idk_count?: number;
  student_attempted?: boolean;
  classifier_output?: string;
  crag_decision?: string;
  draft_source_node?: string;
  topic_choice?: string;
  mastery_choice?: string;
  dean_revisions?: number;
  messages: ChatMessage[];
}

export interface DemoTraceListItem {
  id: string;
  label: string;
  event_count: number;
}

export interface DemoTracePayload {
  id: string;
  label: string;
  events: TraceEvent[];
}
