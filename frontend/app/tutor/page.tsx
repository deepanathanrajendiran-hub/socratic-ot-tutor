// frontend/app/tutor/page.tsx
"use client";
import { useState } from "react";
import { useSession } from "@/lib/useSession";
import { useChatStream } from "@/lib/useChatStream";
import type { Mode } from "@/lib/api-types";
import { ChatThread } from "@/components/chat/ChatThread";
import { ChatInput } from "@/components/chat/ChatInput";
import { ModeToggle } from "@/components/chat/ModeToggle";
import { WeakTopicsSidebar } from "@/components/chat/WeakTopicsSidebar";

export default function TutorPage() {
  const { sessionId, reset } = useSession();
  const [mode, setMode] = useState<Mode>("socratic");
  const { messages, pending, error, send, turnCount } =
    useChatStream({ sessionId, mode });

  async function handleModeChange(m: Mode) {
    if (m === mode) return;
    if (messages.length > 0) {
      const ok = window.confirm(
        `Switch to ${m} mode? This starts a fresh conversation.`);
      if (!ok) return;
    }
    setMode(m);
    await reset();
  }

  return (
    <div className="flex flex-col gap-4 md:flex-row">
      <div className="flex-1">
        <div className="mb-3 flex items-center justify-between">
          <h1 className="text-xl font-semibold">Tutor</h1>
          <ModeToggle mode={mode} onChange={handleModeChange} disabled={pending} />
        </div>
        {error && (
          <div className="mb-3 rounded-md border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-800">
            Error: {error}
          </div>
        )}
        <ChatThread messages={messages} pending={pending} />
        <ChatInput onSend={send} disabled={pending || !sessionId} />
      </div>
      <WeakTopicsSidebar
        weakTopics={[]}
        concept=""
        turnCount={turnCount}
        mode={mode}
      />
    </div>
  );
}
