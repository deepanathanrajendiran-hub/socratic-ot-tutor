// frontend/components/chat/WeakTopicsSidebar.tsx
"use client";
export function WeakTopicsSidebar(props: {
  weakTopics: string[]; concept: string; turnCount: number; mode: string;
}) {
  return (
    <aside className="w-full md:w-64 shrink-0 space-y-4 text-sm">
      <div className="rounded-lg border border-slate-200 bg-white p-3">
        <div className="mb-1 text-xs uppercase tracking-wide text-slate-500">Mode</div>
        <div className="font-medium capitalize">{props.mode}</div>
      </div>
      <div className="rounded-lg border border-slate-200 bg-white p-3">
        <div className="mb-1 text-xs uppercase tracking-wide text-slate-500">Concept this turn</div>
        <div className="font-medium">{props.concept || "—"}</div>
      </div>
      <div className="rounded-lg border border-slate-200 bg-white p-3">
        <div className="mb-1 text-xs uppercase tracking-wide text-slate-500">Turn</div>
        <div className="font-medium">{props.turnCount}</div>
      </div>
      <div className="rounded-lg border border-slate-200 bg-white p-3">
        <div className="mb-2 text-xs uppercase tracking-wide text-slate-500">Weak topics</div>
        {props.weakTopics.length === 0
          ? <div className="text-slate-400">none yet</div>
          : <ul className="space-y-1">
              {props.weakTopics.map((t) => (
                <li key={t} className="rounded bg-amber-100 px-2 py-1 text-xs text-amber-900">{t}</li>
              ))}
            </ul>}
      </div>
    </aside>
  );
}
