// frontend/components/chat/ChatBubble.tsx
"use client";
import { motion } from "framer-motion";
import { Volume2, VolumeX, X } from "lucide-react";
import type { ChatMessage } from "@/lib/api-types";
import { MarkSynapse } from "@/components/ui/Brand";
import { useSpeechSynthesis } from "@/lib/useSpeechSynthesis";
import { ChoiceButtons, parseChoiceMenu, stripChoiceMenu } from "./ChoiceButtons";
import { Fragment } from "react";

/** Render a small subset of inline markdown — `**bold**` and `*italic*`
 *  — into JSX. Claude's outputs use these naturally; rendering them
 *  raw shows asterisks to the user.
 *
 *  Intentionally narrow: no links, no code blocks, no lists. The chat
 *  bubble already preserves newlines via `whitespace-pre-wrap`, so
 *  paragraphs and line breaks render correctly without a full markdown
 *  parser. If we ever need code blocks or tables we'll swap in
 *  react-markdown — until then this 30-liner avoids the dependency. */
function renderInlineMarkdown(text: string): React.ReactNode {
  // Match bold first (longer prefix), then italic. Lazy `+?` so a stray
  // `*` deep in the message can't gobble the rest. We DO allow newlines
  // inside the emphasis body — Claude often line-wraps long italics
  // and the previous strict-single-line rule left their closing `*`
  // unmatched (visible asterisks bug).
  const PATTERN = /(\*\*[^*]+?\*\*|\*[^*]+?\*)/g;
  const parts: React.ReactNode[] = [];
  let lastIdx = 0;
  let key = 0;
  for (const match of text.matchAll(PATTERN)) {
    const idx = match.index ?? 0;
    if (idx > lastIdx) parts.push(text.slice(lastIdx, idx));
    const m = match[0];
    if (m.startsWith("**")) {
      parts.push(
        <strong key={key++} className="font-semibold">
          {m.slice(2, -2)}
        </strong>,
      );
    } else {
      parts.push(<em key={key++}>{m.slice(1, -1)}</em>);
    }
    lastIdx = idx + m.length;
  }
  if (lastIdx < text.length) parts.push(text.slice(lastIdx));
  return parts.map((p, i) => <Fragment key={i}>{p}</Fragment>);
}

/** A single message row.
 *
 *  user      → small ivory pill, right-aligned
 *  assistant → no bubble; coral "S" avatar on the left, prose text on the
 *              ivory background. When `streaming` is true a blinking
 *              caret is appended to the visible text. A 🔊 button below
 *              the prose reads the message aloud via Web Speech API.
 */
export function ChatBubble({
  msg,
  streaming = false,
  isLastAssistant = false,
  pending = false,
  onSend,
  onRewind,
}: {
  msg: ChatMessage;
  streaming?: boolean;
  /** True when this is the most recent assistant message in the thread.
   *  Only the latest A/B/C menu should render clickable buttons; older
   *  ones are stale and the student already responded to them. */
  isLastAssistant?: boolean;
  /** True while the next turn is in flight — used to disable buttons so
   *  the student can't double-click. */
  pending?: boolean;
  /** Bubble doesn't need to know the wider chat plumbing — just give
   *  it a function to send text as the user. Click on a Choice card
   *  calls onSend("A" | "B" | "C"). */
  onSend?: (text: string) => void;
  /** When set on a USER message, renders a small × button on hover.
   *  Click → confirm → truncate this turn (and everything after) so
   *  the student can re-type. Disabled while pending. */
  onRewind?: () => void;
}) {
  const isUser = msg.role === "user";

  if (isUser) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.18, ease: "easeOut" }}
        className="group relative flex w-full justify-end"
      >
        {onRewind && !pending && (
          <button
            type="button"
            onClick={() => {
              const preview = msg.content.slice(0, 40);
              const ok = window.confirm(
                `Delete "${preview}${
                  msg.content.length > 40 ? "…" : ""
                }" and everything after it? This rewinds the chat.`,
              );
              if (ok) onRewind();
            }}
            aria-label="Delete this message and everything after"
            title="Delete this message"
            className="mr-1 mt-1.5 grid h-6 w-6 shrink-0 place-items-center rounded-full border border-ivory-200 bg-white text-ivory-400 opacity-0 shadow-sm transition group-hover:opacity-100 hover:border-rose-300 hover:text-rose-600 ring-focus"
          >
            <X size={12} />
          </button>
        )}
        <div className="max-w-[85%] whitespace-pre-wrap rounded-2xl border border-ivory-200 bg-ivory-100 px-4 py-2.5 text-[15px] leading-relaxed text-ink">
          {msg.content || (
            <span className="italic text-ivory-400">(empty message)</span>
          )}
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.18, ease: "easeOut" }}
      className="flex w-full gap-3"
    >
      <div
        aria-hidden
        className="mt-0.5 grid h-7 w-7 shrink-0 place-items-center rounded-full bg-ivory-100 ring-1 ring-ivory-200"
      >
        <MarkSynapse size={20} label="" />
      </div>
      {(() => {
        // Detect the A/B/C menu once. If buttons will render below the
        // prose, strip the menu lines from the visible text so the
        // user doesn't see the same options twice (text + buttons).
        // The original msg.content is unchanged in state — the
        // mastery_choice_classifier on the backend still sees the
        // full text on the next turn.
        const choices =
          !streaming && msg.content && isLastAssistant && onSend
            ? parseChoiceMenu(msg.content)
            : null;
        const visibleText = choices
          ? stripChoiceMenu(msg.content)
          : msg.content;

        return (
          <div className="min-w-0 flex-1">
            <div className="whitespace-pre-wrap text-[15px] leading-[1.7] text-ink">
              {visibleText ? (
                renderInlineMarkdown(visibleText)
              ) : (
                <span className="italic text-ivory-400">(empty response)</span>
              )}
              {streaming && (
                <span
                  aria-hidden
                  className="ml-0.5 inline-block h-[1.05em] w-[2px] translate-y-[2px] bg-coral-500 align-middle animate-blink"
                />
              )}
            </div>
            {choices && onSend && (
              <ChoiceButtons
                choices={choices}
                disabled={pending}
                onPick={(letter) => onSend(letter)}
              />
            )}

            {/* Hide the speaker button while streaming so the user
                doesn't try to read partial text aloud. */}
            {!streaming && msg.content && (
              <SpeakButton text={msg.content} />
            )}
          </div>
        );
      })()}
    </motion.div>
  );
}

function SpeakButton({ text }: { text: string }) {
  const { supported, speakingId, speak, cancel } = useSpeechSynthesis();
  const isPlaying = speakingId === "tts";

  if (!supported) return null;

  return (
    <button
      type="button"
      onClick={() => (isPlaying ? cancel() : speak("tts", text))}
      aria-label={isPlaying ? "Stop reading aloud" : "Read aloud"}
      title={isPlaying ? "Stop" : "Read aloud"}
      className={`mt-2 inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] transition-colors ring-focus ${
        isPlaying
          ? "border-coral-300 bg-coral-50 text-coral-700"
          : "border-ivory-200 bg-white text-ivory-500 hover:border-ivory-300 hover:text-ink"
      }`}
    >
      {isPlaying ? <VolumeX size={13} /> : <Volume2 size={13} />}
      <span>{isPlaying ? "Stop" : "Listen"}</span>
    </button>
  );
}
