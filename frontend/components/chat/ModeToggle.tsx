// frontend/components/chat/ModeToggle.tsx
"use client";
import type { Mode } from "@/lib/api-types";

export function ModeToggle({ mode, onChange, disabled }: {
  mode: Mode; onChange: (m: Mode) => void; disabled?: boolean;
}) {
  const opts: { value: Mode; label: string }[] = [
    { value: "socratic", label: "Socratic" },
    { value: "study",    label: "Study" },
  ];
  return (
    <div className="inline-flex rounded-md border border-slate-300 bg-white p-0.5">
      {opts.map((o) => (
        <button
          key={o.value}
          onClick={() => !disabled && onChange(o.value)}
          disabled={disabled}
          className={`rounded px-3 py-1 text-xs font-medium transition-colors ${
            mode === o.value ? "bg-slate-900 text-white"
                             : "text-slate-700 hover:bg-slate-100"
          } disabled:opacity-50`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}
