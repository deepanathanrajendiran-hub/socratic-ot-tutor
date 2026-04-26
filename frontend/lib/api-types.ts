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
}

// SSE event payloads from POST /chat
export interface ChatResponseEvent {
  response: string;       // full assistant text (sync streaming)
}
export interface ChatDoneEvent {
  done: true;
  turn_count: number;
}
export interface ChatErrorEvent {
  error: string;
}
export type ChatEvent = ChatResponseEvent | ChatDoneEvent | ChatErrorEvent;

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
