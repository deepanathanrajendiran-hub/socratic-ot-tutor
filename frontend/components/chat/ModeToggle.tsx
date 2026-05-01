// frontend/components/chat/ModeToggle.tsx
"use client";
import type { Mode } from "@/lib/api-types";

export function ModeToggle({ mode, onChange, disabled }: {
  mode: Mode; onChange: (m: Mode) => void; disabled?: boolean;
}) {
  const opts: { value: Mode; label: string; hint: string }[] = [
    { value: "socratic", label: "Socratic", hint: "Guided questioning" },
    { value: "study",    label: "Study",    hint: "Direct answers" },
  ];
  return (
    <div className="inline-flex items-center rounded-full border border-ivory-200 bg-white p-0.5 shadow-sm">
      {opts.map((o) => {
        const active = mode === o.value;
        return (
          <button
            key={o.value}
            type="button"
            onClick={() => !disabled && onChange(o.value)}
            disabled={disabled}
            title={o.hint}
            aria-pressed={active}
            className={`rounded-full px-3.5 py-1.5 text-xs font-medium transition-colors ring-focus ${
              active
                ? "bg-ink text-ivory-50 shadow-sm"
                : "text-ivory-600 hover:text-ink"
            } disabled:opacity-50 disabled:cursor-not-allowed`}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}
