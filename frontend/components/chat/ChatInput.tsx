// frontend/components/chat/ChatInput.tsx
"use client";
import { useEffect, useRef, useState, KeyboardEvent, ChangeEvent } from "react";
import { Paperclip, ArrowUp, X, Mic, MicOff } from "lucide-react";
import { useSpeechRecognition } from "@/lib/useSpeechRecognition";

const MAX_BYTES = 5 * 1024 * 1024; // 5 MB upper bound — Sonnet limit is higher
                                   // but anatomy diagrams should be way smaller

function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error);
    reader.onload = () => {
      const result = reader.result as string;
      // result is "data:image/jpeg;base64,<b64>" — strip the prefix
      const idx = result.indexOf(",");
      resolve(idx === -1 ? result : result.slice(idx + 1));
    };
    reader.readAsDataURL(file);
  });
}

export function ChatInput({ onSend, disabled }: {
  onSend: (text: string, imageB64?: string) => void;
  disabled: boolean;
}) {
  const [text, setText] = useState("");
  const [pendingImage, setPendingImage] = useState<{
    name: string;
    b64: string;
    previewUrl: string;
  } | null>(null);
  const [imageError, setImageError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // STT via Web Speech API. Captures the snapshot of `text` at start
  // time so partials append to whatever the user already typed without
  // racing the textarea state.
  const stt = useSpeechRecognition();
  const sttBaseRef = useRef("");
  function toggleMic() {
    if (disabled) return;
    if (stt.listening) { stt.stop(); return; }
    sttBaseRef.current = text.trim() ? text.trim() + " " : "";
    stt.start((transcript) => {
      // Replace strategy: each call gives the FULL utterance so far,
      // so we set text = base + latest. Final utterance keeps the same
      // shape; user can keep typing afterward.
      setText(sttBaseRef.current + transcript);
    });
  }

  // Auto-grow the textarea up to ~6 lines, then scroll inside.
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 192)}px`;
  }, [text]);

  async function handleFileChange(e: ChangeEvent<HTMLInputElement>) {
    setImageError(null);
    const file = e.target.files?.[0];
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setImageError("Please pick an image file (JPEG, PNG, WEBP).");
      return;
    }
    if (file.size > MAX_BYTES) {
      setImageError(
        `Image is ${(file.size / 1024 / 1024).toFixed(1)} MB; max is 5 MB.`);
      return;
    }
    try {
      const b64 = await fileToBase64(file);
      const previewUrl = URL.createObjectURL(file);
      setPendingImage({ name: file.name, b64, previewUrl });
    } catch (err) {
      setImageError("Couldn't read that image — try a different file.");
    }
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  function clearImage() {
    if (pendingImage?.previewUrl) URL.revokeObjectURL(pendingImage.previewUrl);
    setPendingImage(null);
  }

  function submit() {
    const t = text.trim();
    if (disabled) return;
    if (!t && !pendingImage) return;
    const finalText = t || (pendingImage ? "What is this structure?" : "");
    onSend(finalText, pendingImage?.b64);
    setText("");
    clearImage();
  }

  const canSubmit = !disabled && (text.trim().length > 0 || !!pendingImage);

  return (
    <div className="mx-auto w-full max-w-reading">
      {imageError && (
        <div className="mb-2 rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
          {imageError}
        </div>
      )}

      <div
        className={`group relative flex flex-col rounded-3xl border bg-white shadow-sm transition-all
          ${disabled ? "opacity-70" : ""}
          border-ivory-200 focus-within:border-coral-300 focus-within:shadow-md`}
      >
        {pendingImage && (
          <div className="flex items-center gap-3 border-b border-ivory-100 px-4 pt-3 pb-2">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={pendingImage.previewUrl}
              alt={pendingImage.name}
              className="h-10 w-10 rounded-lg object-cover ring-1 ring-ivory-200"
            />
            <div className="min-w-0 flex-1 truncate text-sm text-ivory-600" title={pendingImage.name}>
              {pendingImage.name}
            </div>
            <button
              type="button"
              onClick={clearImage}
              disabled={disabled}
              className="rounded-full p-1 text-ivory-500 transition hover:bg-ivory-100 hover:text-ink disabled:opacity-40"
              title="Remove image"
              aria-label="Remove attached image"
            >
              <X size={16} />
            </button>
          </div>
        )}

        <div className="flex items-end gap-1 px-2.5 py-2.5">
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            disabled={disabled}
            className="hidden"
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={disabled}
            title="Attach an anatomy diagram"
            aria-label="Attach an image"
            className="grid h-9 w-9 shrink-0 place-items-center rounded-full text-ivory-500 transition hover:bg-ivory-100 hover:text-ink disabled:opacity-40 ring-focus"
          >
            <Paperclip size={18} />
          </button>

          {stt.supported && (
            <button
              type="button"
              onClick={toggleMic}
              disabled={disabled}
              title={stt.listening ? "Stop dictation" : "Dictate (Web Speech)"}
              aria-label={stt.listening ? "Stop dictation" : "Start dictation"}
              aria-pressed={stt.listening}
              className={`grid h-9 w-9 shrink-0 place-items-center rounded-full transition disabled:opacity-40 ring-focus ${
                stt.listening
                  ? "bg-coral-500 text-white shadow-sm animate-pulse"
                  : "text-ivory-500 hover:bg-ivory-100 hover:text-ink"
              }`}
            >
              {stt.listening ? <MicOff size={18} /> : <Mic size={18} />}
            </button>
          )}

          <textarea
            ref={textareaRef}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e: KeyboardEvent<HTMLTextAreaElement>) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            placeholder={
              pendingImage
                ? "Add a question (or leave blank for 'What is this structure?')…"
                : "Ask about anything in OT anatomy…"
            }
            disabled={disabled}
            rows={1}
            className="min-h-[2.25rem] max-h-48 flex-1 resize-none border-0 bg-transparent px-2 py-1.5 text-[15px] leading-relaxed text-ink placeholder:text-ivory-400 focus:outline-none disabled:opacity-60"
          />

          <button
            type="button"
            onClick={submit}
            disabled={!canSubmit}
            aria-label="Send message"
            title="Send (Enter)"
            className={`grid h-9 w-9 shrink-0 place-items-center rounded-full text-white transition ring-focus
              ${canSubmit ? "bg-ink hover:bg-ivory-700" : "bg-ivory-300 cursor-not-allowed"}`}
          >
            <ArrowUp size={18} />
          </button>
        </div>
      </div>

      <p className="mt-2 text-center text-[11px] text-ivory-500">
        Press <kbd className="rounded border border-ivory-200 bg-white px-1 py-0.5 text-[10px] font-mono">Enter</kbd> to send,
        {" "}<kbd className="rounded border border-ivory-200 bg-white px-1 py-0.5 text-[10px] font-mono">Shift</kbd>+<kbd className="rounded border border-ivory-200 bg-white px-1 py-0.5 text-[10px] font-mono">Enter</kbd> for a new line.
      </p>
    </div>
  );
}
