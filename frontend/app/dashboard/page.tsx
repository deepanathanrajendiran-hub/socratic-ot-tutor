// frontend/app/dashboard/page.tsx
"use client";
import { useEffect, useState } from "react";
import { useSession } from "@/lib/useSession";
import { api } from "@/lib/api";
import type { SessionState } from "@/lib/api-types";

export default function DashboardPage() {
  const { sessionId } = useSession();
  const [state, setState] = useState<SessionState | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!sessionId) return;
    api.getSession(sessionId)
       .then(setState)
       .catch((e) => setError(e instanceof Error ? e.message : "fetch error"));
  }, [sessionId]);

  if (!sessionId) return <div>Initializing session…</div>;
  if (error) return (
    <div className="rounded-md border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-800">
      {error} — try sending a message in /tutor first to seed the session.
    </div>
  );
  if (!state) return <div>Loading…</div>;

  return (
    <div>
      <h1 className="mb-4 text-xl font-semibold">Dashboard</h1>
      <div className="grid gap-4 md:grid-cols-2">
        <div className="rounded-lg border border-slate-200 bg-white p-4">
          <div className="mb-3 text-xs uppercase tracking-wide text-slate-500">Session</div>
          <dl className="space-y-1 text-sm">
            <div className="flex justify-between"><dt className="text-slate-500">id</dt><dd className="font-mono">{state.session_id.slice(0, 8)}…</dd></div>
            <div className="flex justify-between"><dt className="text-slate-500">mode</dt><dd>{state.mode}</dd></div>
            <div className="flex justify-between"><dt className="text-slate-500">turns</dt><dd>{state.turn_count}</dd></div>
            <div className="flex justify-between"><dt className="text-slate-500">phase</dt><dd>{state.student_phase}</dd></div>
            <div className="flex justify-between"><dt className="text-slate-500">mastery</dt><dd>{state.mastery_level || "—"}</dd></div>
          </dl>
        </div>
        <div className="rounded-lg border border-slate-200 bg-white p-4">
          <div className="mb-3 text-xs uppercase tracking-wide text-slate-500">Weak topics</div>
          {state.weak_topics.length === 0
            ? <div className="text-sm text-slate-400">none yet</div>
            : <ul className="space-y-1">
                {state.weak_topics.map((t) => (
                  <li key={t} className="rounded bg-amber-100 px-2 py-1 text-xs text-amber-900">{t}</li>
                ))}
              </ul>}
        </div>
      </div>
      <div className="mt-6 rounded-lg border border-slate-200 bg-white p-4">
        <div className="mb-3 text-xs uppercase tracking-wide text-slate-500">Messages</div>
        <div className="space-y-2 text-sm">
          {state.messages.map((m, i) => (
            <div key={i} className={m.role === "user" ? "text-slate-900" : "text-slate-600"}>
              <span className="mr-2 font-medium">{m.role === "user" ? "→" : "◇"}</span>
              {m.content}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
