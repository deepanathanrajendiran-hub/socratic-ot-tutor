// frontend/components/chat/WeakTopicsSidebar.tsx
"use client";

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-2xl border border-ivory-200 bg-white p-4 shadow-sm">
      <div className="mb-1.5 text-[11px] font-medium uppercase tracking-[0.08em] text-ivory-500">
        {title}
      </div>
      <div className="text-sm text-ink">{children}</div>
    </div>
  );
}

export function WeakTopicsSidebar(props: {
  weakTopics: string[];
  concept: string;
  turnCount: number;
  mode: string;
}) {
  return (
    <aside className="w-full shrink-0 space-y-3 md:w-72">
      <Card title="Mode">
        <div className="flex items-center gap-2 capitalize">
          <span
            aria-hidden
            className="inline-block h-2 w-2 rounded-full bg-coral-500"
          />
          {props.mode}
          <span className="ml-auto text-[11px] text-ivory-500">
            turn {props.turnCount}
          </span>
        </div>
      </Card>

      <Card title="Now studying">
        {props.concept ? (
          <span className="font-serif text-[17px] leading-tight">
            {props.concept}
          </span>
        ) : (
          <span className="italic text-ivory-400">— pick a topic to begin</span>
        )}
      </Card>

      <Card title="Weak topics">
        {props.weakTopics.length === 0 ? (
          <span className="text-ivory-400">none yet</span>
        ) : (
          <ul className="flex flex-wrap gap-1.5">
            {props.weakTopics.map((t) => (
              <li
                key={t}
                className="rounded-full border border-coral-200 bg-coral-50 px-2.5 py-1 text-xs text-coral-800"
              >
                {t}
              </li>
            ))}
          </ul>
        )}
      </Card>
    </aside>
  );
}
