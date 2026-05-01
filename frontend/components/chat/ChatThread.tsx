// frontend/components/chat/ChatThread.tsx
"use client";
import { useEffect, useRef } from "react";
import type { ChatMessage } from "@/lib/api-types";
import type { PipelineStage } from "@/lib/useChatStream";
import { ChatBubble } from "./ChatBubble";
import { PipelineIndicator } from "./PipelineIndicator";
import { MarkSynapse } from "@/components/ui/Brand";

const STEP_LABELS: Record<string, string> = {
  concept_extraction: "Reading your question…",
  retrieval:          "Pulling textbook context…",
  classifier:         "Reading your answer…",
  generation:         "Writing response…",
  thinking:           "Thinking through your question…",
  dean:               "Checking quality…",
  study:              "Preparing answer…",
};

export function ChatThread({
  messages, pending, currentStep, pipelineStages, onSend, onRewind,
}: {
  messages: ChatMessage[];
  pending: boolean;
  /** Active node from the last `step` start event, or null. When set,
   *  the typing bubble shows a status label instead of dots. Once the
   *  first response token arrives the parent should clear this. */
  currentStep?: string | null;
  /** Ordered pipeline stages this turn (active + done). When non-empty
   *  and `pending` is true, renders a horizontal progress strip above
   *  the in-flight assistant message. */
  pipelineStages?: PipelineStage[];
  /** Sender function — passed to ChatBubble so the A/B/C menu cards
   *  can fire onSend("A" | "B" | "C") when clicked. */
  onSend?: (text: string) => void;
  /** Rewind handler — when set, user-message bubbles render a small ×
   *  button on hover. Click → truncate this turn (and everything
   *  after) from the checkpoint, restoring the thread to before the
   *  user typed it. Pass undefined to disable. */
  onRewind?: (msgIndex: number) => void;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: 1e9, behavior: "smooth" });
  }, [messages.length, pending, currentStep, pipelineStages?.length]);

  const last = messages[messages.length - 1];

  // Show the typing-bubble (3 dots / step label) only before any
  // assistant content has streamed in. Once tokens start arriving we
  // switch to inline blinking cursor on the live assistant message.
  const showTypingBubble =
    pending && (!last || last.role !== "assistant" || !last.content);
  const isStreamingLastAssistant =
    pending && !!last && last.role === "assistant" && !!last.content;

  const stepLabel = currentStep ? STEP_LABELS[currentStep] ?? "Working…" : null;
  const stages = pipelineStages ?? [];

  // Find the index of the last assistant message so ChoiceButtons render
  // only there. Walking backwards is O(1) for typical thread sizes.
  let lastAssistantIdx = -1;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "assistant") { lastAssistantIdx = i; break; }
  }

  return (
    <div
      ref={scrollRef}
      className="relative flex-1 overflow-y-auto"
      style={{ minHeight: "60vh", maxHeight: "calc(100vh - 16rem)" }}
    >
      <div className="mx-auto flex max-w-reading flex-col gap-7 px-1 py-6">
        {messages.map((m, i) => (
          <ChatBubble
            key={i}
            msg={m}
            streaming={isStreamingLastAssistant && i === messages.length - 1}
            isLastAssistant={i === lastAssistantIdx}
            pending={pending}
            onSend={onSend}
            onRewind={onRewind ? () => onRewind(i) : undefined}
          />
        ))}

        {/* Pipeline strip — visible whenever stages exist for this turn.
            Sits above the current assistant message: while the bubble
            is still empty (showTypingBubble) it appears below the user
            message; once tokens stream in it sits above the live
            assistant text and fades when done. */}
        {pending && stages.length > 0 && (
          <div className="flex w-full gap-3">
            <div className="mt-0.5 h-7 w-7 shrink-0" aria-hidden />
            <div className="min-w-0 flex-1">
              <PipelineIndicator stages={stages} visible={pending} />
            </div>
          </div>
        )}

        {showTypingBubble && (
          <div className="flex w-full gap-3">
            <div
              aria-hidden
              className="mt-0.5 grid h-7 w-7 shrink-0 place-items-center rounded-full bg-ivory-100 ring-1 ring-ivory-200"
            >
              <MarkSynapse size={20} label="" />
            </div>
            <div className="flex items-center text-[14px] text-ivory-500">
              {stepLabel ? (
                <span className="inline-flex items-center gap-2">
                  <span
                    aria-hidden
                    className="inline-block h-1.5 w-1.5 rounded-full bg-coral-500 animate-pulse"
                  />
                  {stepLabel}
                </span>
              ) : (
                <span className="inline-flex gap-1">
                  <span className="h-1.5 w-1.5 rounded-full bg-ivory-400 animate-pulse" />
                  <span className="h-1.5 w-1.5 rounded-full bg-ivory-400 animate-pulse [animation-delay:0.18s]" />
                  <span className="h-1.5 w-1.5 rounded-full bg-ivory-400 animate-pulse [animation-delay:0.36s]" />
                </span>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
