// frontend/components/chat/ChatInput.tsx
"use client";
import { useState, KeyboardEvent } from "react";

export function ChatInput({ onSend, disabled }: {
  onSend: (text: string) => void; disabled: boolean;
}) {
  const [text, setText] = useState("");
  function submit() {
    const t = text.trim();
    if (!t || disabled) return;
    onSend(t);
    setText("");
  }
  return (
    <div className="mt-4 flex gap-2">
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e: KeyboardEvent<HTMLTextAreaElement>) => {
          if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); }
        }}
        placeholder="Ask about an OT anatomy concept…"
        disabled={disabled}
        rows={2}
        className="flex-1 resize-none rounded-md border border-slate-300 bg-white px-3 py-2 text-sm focus:border-slate-900 focus:outline-none disabled:bg-slate-100"
      />
      <button
        onClick={submit}
        disabled={disabled || !text.trim()}
        className="self-end rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white disabled:bg-slate-400"
      >
        Send
      </button>
    </div>
  );
}
