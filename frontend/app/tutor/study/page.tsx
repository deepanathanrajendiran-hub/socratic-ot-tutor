// frontend/app/tutor/study/page.tsx
"use client";
import { useEffect, useState } from "react";
import { useSession } from "@/lib/useSession";
import { useChatStream } from "@/lib/useChatStream";
import type { Mode } from "@/lib/api-types";
import { ChatThread } from "@/components/chat/ChatThread";
import { ChatInput } from "@/components/chat/ChatInput";
import { ModeToggle } from "@/components/chat/ModeToggle";

export default function StudyPage() {
  const { sessionId, reset } = useSession();
  const [mode, setMode] = useState<Mode>("study");
  const chat = useChatStream({ sessionId, mode });

  // Force a fresh session on first load so we don't pick up Socratic history
  useEffect(() => {
    reset();
    // We intentionally run this once on mount; reset is a stable callback.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="flex flex-col gap-6 lg:flex-row">
      <section className="flex min-h-[calc(100vh-8rem)] flex-1 flex-col">
        <div className="mb-4 flex items-center justify-between gap-3">
          <div className="flex items-baseline gap-3">
            <h1 className="font-serif text-2xl tracking-tight text-ink">Study mode</h1>
            <span className="text-xs text-ivory-500">
              Direct, textbook-grounded answers — no Socratic gating.
            </span>
          </div>
          <ModeToggle
            mode={mode}
            onChange={(m) => { setMode(m); reset(); }}
            disabled={chat.pending}
          />
        </div>
        {chat.error && (
          <div className="mb-3 rounded-2xl border border-rose-200 bg-rose-50 px-4 py-2.5 text-sm text-rose-800">
            {chat.error}
          </div>
        )}
        <ChatThread
          messages={chat.messages}
          pending={chat.pending}
          currentStep={chat.currentStep}
          pipelineStages={chat.pipelineStages}
          onSend={chat.send}
        />
        <div className="mt-4">
          <ChatInput onSend={chat.send} disabled={chat.pending || !sessionId} />
        </div>
      </section>
    </div>
  );
}
