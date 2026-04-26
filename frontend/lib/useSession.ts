// frontend/lib/useSession.ts
"use client";
import { useEffect, useState } from "react";
import { api } from "./api";

const KEY = "socratic-ot.session_id";

/** Pin a session id in localStorage. Creates one on first call. */
export function useSession(): { sessionId: string | null; reset: () => Promise<void> } {
  const [sessionId, setSessionId] = useState<string | null>(null);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const id = window.localStorage.getItem(KEY);
    if (id) { setSessionId(id); return; }
    api.createSession()
       .then((r) => { window.localStorage.setItem(KEY, r.session_id);
                       setSessionId(r.session_id); })
       .catch(() => setSessionId(null));
  }, []);

  async function reset() {
    const r = await api.createSession();
    if (typeof window !== "undefined") window.localStorage.setItem(KEY, r.session_id);
    setSessionId(r.session_id);
  }

  return { sessionId, reset };
}
