// frontend/app/page.tsx
"use client";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";

export default function Home() {
  const [status, setStatus] = useState<string>("…");
  useEffect(() => {
    api.health().then((h) => setStatus(JSON.stringify(h)))
              .catch((e) => setStatus(`error: ${e.message}`));
  }, []);
  return (
    <main className="p-8 font-mono">
      <h1 className="text-2xl">Socratic-OT (frontend bootstrap)</h1>
      <p className="mt-4">/health → {status}</p>
    </main>
  );
}
