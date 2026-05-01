// frontend/lib/useSession.ts
"use client";
import { useEffect, useState } from "react";
import { api } from "./api";

const KEY = "socratic-ot.session_id";

/** Pin a session id in localStorage. Creates one on first call.
 *  - `reset()`     mints a new id and replaces the stored one (New chat).
 *  - `switchTo(id)` adopts an existing id from the sidebar.
 */
export function useSession(): {
  sessionId: string | null;
  reset:    () => Promise<string>;
  switchTo: (id: string) => void;
} {
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

  async function reset(): Promise<string> {
    const r = await api.createSession();
    if (typeof window !== "undefined") window.localStorage.setItem(KEY, r.session_id);
    setSessionId(r.session_id);
    return r.session_id;
  }

  function switchTo(id: string): void {
    if (!id || id === sessionId) return;
    if (typeof window !== "undefined") window.localStorage.setItem(KEY, id);
    setSessionId(id);
  }

  return { sessionId, reset, switchTo };
}
