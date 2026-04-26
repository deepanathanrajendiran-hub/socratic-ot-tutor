// frontend/app/compare/page.tsx
"use client";
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useSession } from "@/lib/useSession";
import { api } from "@/lib/api";
import { ChatInput } from "@/components/chat/ChatInput";

interface AnswerCard {
  mode: "socratic" | "study";
  text: string;
}

async function fetchOnce(
  sessionId: string,
  mode: "socratic" | "study",
  text: string,
): Promise<string> {
  let out = "";
  for await (const ev of api.chat({
    messages: [{ role: "user", content: text }],
    session_id: sessionId,
    mode,
  })) {
    if ("response" in ev && typeof ev.response === "string") out = ev.response;
  }
  return out;
}

export default function ComparePage() {
  const { sessionId, reset } = useSession();
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [cards, setCards] = useState<AnswerCard[]>([]);

  async function run(text: string) {
    if (!sessionId) return;
    setPending(true);
    setError(null);
    setCards([]);
    try {
      // Use a fresh session for each mode so neither response sees the other's history.
      await reset();
      const sidA = (typeof window !== "undefined")
        ? window.localStorage.getItem("socratic-ot.session_id")
        : null;
      if (!sidA) throw new Error("no session id after reset");
      const socraticText = await fetchOnce(sidA, "socratic", text);
      setCards([{ mode: "socratic", text: socraticText }]);

      // 800ms breath before the Study card slides in (sequential reveal, not split-pane)
      await new Promise((r) => setTimeout(r, 800));

      await reset();
      const sidB = (typeof window !== "undefined")
        ? window.localStorage.getItem("socratic-ot.session_id")
        : null;
      if (!sidB) throw new Error("no session id after reset");
      const studyText = await fetchOnce(sidB, "study", text);
      setCards((c) => [...c, { mode: "study", text: studyText }]);
    } catch (e) {
      setError(e instanceof Error ? e.message : "compare error");
    } finally {
      setPending(false);
    }
  }

  return (
    <div>
      <h1 className="mb-3 text-xl font-semibold">Compare</h1>
      <p className="mb-4 max-w-2xl text-sm text-slate-600">
        Same question, two modes. Socratic guides you toward the answer; Study mode
        explains directly. The Socratic answer appears first; Study slides in after.
      </p>
      <ChatInput onSend={run} disabled={pending || !sessionId} />
      {error && (
        <div className="mt-3 rounded-md border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-800">
          {error}
        </div>
      )}
      <div className="mt-6 space-y-4">
        <AnimatePresence>
          {cards.map((c) => (
            <motion.div
              key={c.mode}
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.45 }}
              className="rounded-lg border border-slate-200 bg-white p-4"
            >
              <div className="mb-2 text-xs uppercase tracking-wide text-slate-500">
                {c.mode === "socratic" ? "Socratic mode" : "Study mode"}
              </div>
              <div className="whitespace-pre-wrap text-sm">{c.text}</div>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </div>
  );
}
