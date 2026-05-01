// frontend/lib/useUser.ts
//
// Stable per-browser user identifier. Persists in localStorage under
// `socratic-ot.user_id`. Generated once on first visit; reused forever
// after.
//
// Why we need it:
//   `session_id` resets every time the user clicks "New chat", so the
//   server-side cross-session memory layer (mem0) needs a separate id
//   that survives chat-reset. localStorage is the right scope — it
//   shares across tabs but doesn't follow the user across browsers
//   (which is fine for the demo; production would swap this for real
//   auth).
//
// Privacy note:
//   The id is a random UUID — no PII. The user can clear it any time
//   from devtools (`localStorage.removeItem(...)`) and a fresh id
//   will be minted on the next visit.
"use client";
import { useEffect, useState } from "react";

const KEY = "socratic-ot.user_id";


function _generateId(): string {
  // crypto.randomUUID is available in all modern browsers (Chrome 92+,
  // Safari 15.4+, Firefox 95+). Fallback to a Math.random-based id only
  // for ancient browsers — collision risk is acceptable since this is a
  // demo and the value is not a security boundary.
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return "u_" + Math.random().toString(36).slice(2) + Date.now().toString(36);
}

export function useUser(): { userId: string | null } {
  const [userId, setUserId] = useState<string | null>(null);

  useEffect(() => {
    if (typeof window === "undefined") return;
    let id = window.localStorage.getItem(KEY);
    if (!id) {
      id = _generateId();
      window.localStorage.setItem(KEY, id);
    }
    setUserId(id);
  }, []);

  return { userId };
}
