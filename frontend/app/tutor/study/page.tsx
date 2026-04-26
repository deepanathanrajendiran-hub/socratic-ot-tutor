// frontend/app/tutor/study/page.tsx
"use client";
import { useEffect, useState } from "react";
import { useSession } from "@/lib/useSession";
import { useChatStream } from "@/lib/useChatStream";
import type { Mode } from "@/lib/api-types";
import { ChatThread } from "@/components/chat/ChatThread";
import { ChatInput } from "@/components/chat/ChatInput";
import { ModeToggle } from "@/components/chat/ModeToggle";
import { WeakTopicsSidebar } from "@/components/chat/WeakTopicsSidebar";

export default function StudyPage() {
  const { sessionId, reset } = useSession();
  const [mode, setMode] = useState<Mode>("study");
  const chat = useChatStream({ sessionId, mode });

  // Force a fresh session on first load so we don't pick up Socratic history
  useEffect(() => { reset(); /* eslint-disable-next-line react-hooks/exhaustive-deps */ }, []);

  return (
    <div className="flex flex-col gap-4 md:flex-row">
      <div className="flex-1">
        <div className="mb-3 flex items-center justify-between">
          <h1 className="text-xl font-semibold">Study mode</h1>
          <ModeToggle mode={mode} onChange={(m) => { setMode(m); reset(); }} disabled={chat.pending} />
        </div>
        {chat.error && (
          <div className="mb-3 rounded-md border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-800">
            Error: {chat.error}
          </div>
        )}
        <ChatThread messages={chat.messages} pending={chat.pending} />
        <ChatInput onSend={chat.send} disabled={chat.pending || !sessionId} />
      </div>
      <WeakTopicsSidebar
        weakTopics={[]} concept="" turnCount={chat.turnCount} mode={mode} />
    </div>
  );
}
