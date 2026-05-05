// frontend/app/architecture/page.tsx
"use client";
import { useEffect, useState } from "react";
import { TraceView } from "@/components/architecture/TraceView";
import { ArchitectureView } from "@/components/architecture/ArchitectureView";

type View = "architecture" | "trace";

const STORAGE_KEY = "socratic-ot.architecture_view";

export default function ArchitecturePage() {
  // Default to the explainer view — that's what a first-time visitor
  // (e.g. a professor) should see. The trace view is the deeper-dive
  // option behind the dropdown.
  const [view, setView] = useState<View>("architecture");

  // Persist the choice in localStorage so internal navigation between
  // /tutor and /architecture remembers what the user last looked at.
  useEffect(() => {
    if (typeof window === "undefined") return;
    const saved = window.localStorage.getItem(STORAGE_KEY);
    if (saved === "architecture" || saved === "trace") setView(saved);
  }, []);

  function handleViewChange(next: View) {
    setView(next);
    try {
      window.localStorage.setItem(STORAGE_KEY, next);
    } catch {
      // localStorage unavailable (private mode, etc.) — fail silently;
      // the in-memory state still works for this session.
    }
  }

  return (
    <div>
      {/* ── View switcher ──────────────────────────────────────────── */}
      <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
        <h1 className="font-serif text-2xl tracking-tight text-slate-900">
          Architecture
        </h1>
        <div className="flex items-center gap-2">
          <label
            htmlFor="arch-view-select"
            className="text-xs uppercase tracking-wide text-slate-500"
          >
            View
          </label>
          <select
            id="arch-view-select"
            value={view}
            onChange={(e) => handleViewChange(e.target.value as View)}
            className="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-slate-300"
          >
            <option value="architecture">Architecture overview</option>
            <option value="trace">Real-time pipeline trace</option>
          </select>
        </div>
      </div>

      {view === "architecture" ? <ArchitectureView /> : <TraceView />}
    </div>
  );
}
