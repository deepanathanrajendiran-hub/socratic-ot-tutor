// frontend/components/chat/ChatBubble.tsx
"use client";
import { motion } from "framer-motion";
import type { ChatMessage } from "@/lib/api-types";

export function ChatBubble({ msg }: { msg: ChatMessage }) {
  const isUser = msg.role === "user";
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className={`flex w-full ${isUser ? "justify-end" : "justify-start"}`}
    >
      <div
        className={`max-w-[78%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
          isUser
            ? "bg-slate-900 text-white"
            : "bg-white text-slate-900 border border-slate-200"
        }`}
      >
        {msg.content || (
          <span className="italic text-slate-400">(empty response)</span>
        )}
      </div>
    </motion.div>
  );
}
