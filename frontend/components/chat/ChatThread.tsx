// frontend/components/chat/ChatThread.tsx
"use client";
import { useEffect, useRef } from "react";
import type { ChatMessage } from "@/lib/api-types";
import { ChatBubble } from "./ChatBubble";

export function ChatThread({ messages, pending }: {
  messages: ChatMessage[];
  pending: boolean;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: 1e9, behavior: "smooth" });
  }, [messages.length, pending]);

  return (
    <div
      ref={scrollRef}
      className="flex flex-col gap-3 overflow-y-auto rounded-lg bg-slate-100 p-4"
      style={{ maxHeight: "60vh" }}
    >
      {messages.map((m, i) => <ChatBubble key={i} msg={m} />)}
      {pending && (
        <div className="flex justify-start">
          <div className="rounded-2xl bg-white border border-slate-200 px-4 py-2.5 text-sm text-slate-500">
            <span className="inline-flex gap-1">
              <span className="animate-pulse">●</span>
              <span className="animate-pulse [animation-delay:0.2s]">●</span>
              <span className="animate-pulse [animation-delay:0.4s]">●</span>
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
