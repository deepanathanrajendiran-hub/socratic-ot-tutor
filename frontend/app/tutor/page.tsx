// frontend/app/tutor/page.tsx
"use client";
import { useEffect, useRef, useState } from "react";
import { useSession } from "@/lib/useSession";
import { useUser } from "@/lib/useUser";
import { useChatStream } from "@/lib/useChatStream";
import { api } from "@/lib/api";
import type { Mode, SessionState } from "@/lib/api-types";
import { ChatThread } from "@/components/chat/ChatThread";
import { ChatInput } from "@/components/chat/ChatInput";
import { ModeToggle } from "@/components/chat/ModeToggle";
import { SessionsSidebar } from "@/components/chat/SessionsSidebar";
import { MarkSynapse } from "@/components/ui/Brand";

const SUGGESTIONS = [
  "What's the gap between neurons?",
  "Which nerve is compressed in carpal tunnel?",
  "How does a reflex arc work?",
  "What's the role of the cerebellum?",
  "What nerve causes the funny-bone sensation?",
];

export default function TutorPage() {
  const { sessionId, reset, switchTo } = useSession();
  const { userId } = useUser();
  const [mode, setMode] = useState<Mode>("socratic");
  // Domain selector — switching this on the client makes the next /chat
  // request use the new domain (backend reads `domain` per-request and
  // overrides config.DOMAIN). Persisted in localStorage so the choice
  // survives reload. Initialized lazily to avoid SSR/CSR hydration
  // mismatch (window is undefined during server render).
  const [domain, setDomain] = useState<string>("OT_anatomy");
  useEffect(() => {
    const saved = typeof window !== "undefined"
      ? window.localStorage.getItem("socratic_domain")
      : null;
    if (saved === "OT_anatomy" || saved === "physics") setDomain(saved);
  }, []);
  const {
    messages, pending, error, send, turnCount, currentStep,
    setMessages, pipelineStages,
  } = useChatStream({ sessionId, mode, userId, domain });

  function handleDomainChange(next: string) {
    if (next !== "OT_anatomy" && next !== "physics") return;
    setDomain(next);
    try { window.localStorage.setItem("socratic_domain", next); } catch {}
    // Switching domain mid-session would mix concepts across textbooks,
    // so start a fresh session. The new chat appears in the sidebar.
    handleNewChat();
  }
  const [sessionState, setSessionState] = useState<SessionState | null>(null);
  // User-scoped weak topics — independent of which chat is active. The
  // sidebar reads this so switching between old/new chats always shows
  // the same persistent list (single source of truth = user_weak_topics
  // table on the backend).
  const [userWeakTopics, setUserWeakTopics] = useState<string[]>([]);
  // Bumped whenever the recent-chats sidebar should refetch its list
  // (after new chat / send / rename / delete). Cheap React signal.
  const [sidebarRefresh, setSidebarRefresh] = useState(0);
  // Track the previous turnCount so we know when an actual turn just
  // completed (not a session-switch). Used to auto-title the chat
  // from the first user message.
  const prevTurnRef = useRef(0);

  async function handleNewChat(): Promise<string> {
    const id = await reset();
    setMessages([]);
    setSessionState(null);
    prevTurnRef.current = 0;
    setSidebarRefresh((n) => n + 1);
    return id;
  }

  async function handleSelectSession(id: string) {
    if (id === sessionId) return;
    switchTo(id);
    // Load history + mode + sidebar state from the server. New sessions
    // 404 here (no checkpoint yet) — that's fine, just blank thread.
    try {
      const s = await api.getSession(id, userId);
      setSessionState(s);
      setMessages(s.messages ?? []);
      if (s.mode === "socratic" || s.mode === "study") setMode(s.mode);
    } catch {
      setSessionState(null);
      setMessages([]);
    }
    prevTurnRef.current = 0;
  }

  // Pull session-scoped state (current_concept, mode, history) after
  // each turn — but NOT for the sidebar's weak-topics card. That card
  // reads userWeakTopics below.
  useEffect(() => {
    if (!sessionId) { setSessionState(null); return; }
    api.getSession(sessionId, userId)
      .then(setSessionState)
      .catch(() => setSessionState(null));
  }, [sessionId, turnCount, userId]);

  // Restore the message history when a session loads. handleSelectSession
  // already restores inline on sidebar clicks for snappier UI; this
  // effect catches the initial-mount path where sessionId is hydrated
  // from localStorage by useSession and no explicit switch event fires.
  // Without this, a page reload showed an empty thread even though the
  // chat history was persisted in Postgres — the server-side state was
  // fetched into sessionState but never copied into useChatStream's
  // messages array. Deps are sessionId only — restoration should run
  // once per session, not after every turn (which would race with the
  // streaming optimistic UI).
  useEffect(() => {
    if (!sessionId) return;
    let cancelled = false;
    api.getSession(sessionId, userId)
      .then((s) => {
        if (cancelled) return;
        setMessages(s.messages ?? []);
        if (s.mode === "socratic" || s.mode === "study") setMode(s.mode);
      })
      .catch(() => {});
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  // User-scoped weak topics — refresh when the user changes, after
  // each turn (a fail/master may have added/removed entries), and
  // after session switches (so the list shows immediately, not on
  // next turn).
  useEffect(() => {
    if (!userId) { setUserWeakTopics([]); return; }
    api.getUserWeakTopics(userId)
      .then((r) => setUserWeakTopics(r.weak_topics ?? []))
      .catch(() => setUserWeakTopics([]));
  }, [userId, turnCount, sessionId]);

  // After the first user message lands a real turn, set the chat title
  // from that message so the sidebar shows something meaningful.
  useEffect(() => {
    if (!sessionId) return;
    if (turnCount > prevTurnRef.current) {
      prevTurnRef.current = turnCount;
      setSidebarRefresh((n) => n + 1);
      // Look up the first user message and use it as the title (only
      // for the very first turn — later turns just refresh recency).
      if (turnCount === 1) {
        const firstUser = messages.find((m) => m.role === "user");
        if (firstUser?.content) {
          const title = firstUser.content.replace(/\s+/g, " ").trim().slice(0, 60);
          api.patchSession(sessionId, { title }).catch(() => {});
        }
      }
    }
  }, [turnCount, sessionId, messages]);

  async function handleRewind(msgIndex: number) {
    if (!sessionId || msgIndex < 0) return;
    try {
      await api.truncateMessages(sessionId, msgIndex);
      // Optimistic local trim — the next /sessions GET will confirm.
      setMessages(messages.slice(0, msgIndex));
      // Force a session-state refetch so turn_count / current_concept
      // pick up the truncation immediately.
      try {
        const s = await api.getSession(sessionId, userId);
        setSessionState(s);
        setMessages(s.messages ?? messages.slice(0, msgIndex));
      } catch { /* keep optimistic trim */ }
      setSidebarRefresh((n) => n + 1);
    } catch (e) {
      window.alert(
        `Couldn't rewind: ${e instanceof Error ? e.message : "unknown error"}`,
      );
    }
  }

  async function handleModeChange(m: Mode) {
    if (m === mode) return;
    if (messages.length > 0) {
      const ok = window.confirm(
        `Switch to ${m} mode? This starts a fresh conversation.`);
      if (!ok) return;
    }
    setMode(m);
    await handleNewChat();
  }

  const isEmpty = messages.length === 0;

  return (
    <div className="flex flex-col gap-6 lg:flex-row">
      {/* Left rail — recent chats + weak topics */}
      <SessionsSidebar
        activeId={sessionId}
        refreshKey={sidebarRefresh}
        onSelect={handleSelectSession}
        onNew={handleNewChat}
        weakTopics={userWeakTopics}
        debugInfo={sessionState}
      />

      {/* Main column — chat reading area */}
      <section className="flex min-h-[calc(100vh-8rem)] min-w-0 flex-1 flex-col">
        <div className="mb-4 flex items-center justify-between gap-3">
          <div className="flex items-baseline gap-3">
            <h1 className="font-serif text-2xl tracking-tight text-ink">
              Tutor
            </h1>
            <span className="text-xs text-ivory-500">
              {mode === "socratic"
                ? "I'll guide you with questions before giving answers."
                : "I'll explain concepts directly with textbook grounding."}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <select
              value={domain}
              onChange={(e) => handleDomainChange(e.target.value)}
              disabled={pending}
              aria-label="Domain"
              className="rounded-2xl border border-ivory-300 bg-white px-3 py-1.5 text-sm text-ink focus:outline-none focus:ring-2 focus:ring-blue-200 disabled:opacity-50"
            >
              <option value="OT_anatomy">OT anatomy</option>
              <option value="physics">Physics</option>
            </select>
            <ModeToggle mode={mode} onChange={handleModeChange} disabled={pending} />
          </div>
        </div>

        {error && (
          <div className="mb-3 rounded-2xl border border-rose-200 bg-rose-50 px-4 py-2.5 text-sm text-rose-800">
            {error}
          </div>
        )}

        {isEmpty ? (
          <EmptyState onPick={(text) => send(text)} disabled={pending || !sessionId} />
        ) : (
          <ChatThread
            messages={messages}
            pending={pending}
            currentStep={currentStep}
            pipelineStages={pipelineStages}
            onSend={send}
            onRewind={handleRewind}
          />
        )}

        <div className="mt-4">
          <ChatInput onSend={send} disabled={pending || !sessionId} />
        </div>
      </section>
    </div>
  );
}

function EmptyState({
  onPick,
  disabled,
}: {
  onPick: (text: string) => void;
  disabled: boolean;
}) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center px-4 py-10">
      <div className="mx-auto w-full max-w-reading text-center">
        <div className="mx-auto mb-6 grid h-14 w-14 place-items-center">
          <MarkSynapse size={56} label="" />
        </div>
        <h2 className="font-serif text-3xl font-medium tracking-tight text-ink md:text-4xl">
          What are you studying today?
        </h2>
        <p className="mt-3 text-sm text-ivory-600">
          Ask anything in OT anatomy or neuroscience — or start with one of these.
        </p>

        <div className="mt-7 flex flex-wrap justify-center gap-2">
          {SUGGESTIONS.map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => onPick(s)}
              disabled={disabled}
              className="rounded-full border border-ivory-200 bg-white px-4 py-2 text-sm text-ivory-700 shadow-sm transition hover:border-coral-300 hover:bg-coral-50 hover:text-coral-800 disabled:opacity-50 disabled:cursor-not-allowed ring-focus"
            >
              {s}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
