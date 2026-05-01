// frontend/lib/useSpeechSynthesis.ts
//
// Thin wrapper around the browser's Web Speech API for TTS. The page
// can call `speak(text)` to read a string aloud, `cancel()` to stop,
// and watch `speakingId` to know which message is currently playing.
//
// Why a hook + a single global queue:
//   speechSynthesis is a singleton owned by the browser. If two
//   ChatBubbles race their `speak()` calls the second one wins and
//   the first leaks a dangling utterance. We expose `speak(id, text)`
//   so callers identify themselves; starting a new speak cancels any
//   prior one and updates `speakingId` to the new one. Bubble UIs
//   compare their own id to `speakingId` to render play vs stop.
"use client";
import { useCallback, useEffect, useState } from "react";

export interface UseSpeechSynthesis {
  /** True when the browser supports SpeechSynthesis. False on rare
   *  outdated UAs — callers should hide TTS buttons when false. */
  supported: boolean;
  /** ID of the utterance currently being spoken, or null if silent.
   *  Use any stable string (e.g. message index "msg-3"). */
  speakingId: string | null;
  /** Speak `text` aloud, tagging the utterance with `id`. Cancels any
   *  prior in-flight utterance first. */
  speak: (id: string, text: string) => void;
  /** Stop any in-flight speech immediately. */
  cancel: () => void;
}

export function useSpeechSynthesis(): UseSpeechSynthesis {
  const [supported, setSupported] = useState(false);
  const [speakingId, setSpeakingId] = useState<string | null>(null);

  useEffect(() => {
    if (typeof window === "undefined") return;
    setSupported(typeof window.speechSynthesis !== "undefined");
  }, []);

  // Always cancel on unmount so navigating between routes doesn't leak
  // a still-speaking utterance.
  useEffect(() => {
    return () => {
      if (typeof window !== "undefined" && window.speechSynthesis) {
        window.speechSynthesis.cancel();
      }
    };
  }, []);

  const cancel = useCallback(() => {
    if (typeof window === "undefined" || !window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    setSpeakingId(null);
  }, []);

  const speak = useCallback((id: string, text: string) => {
    if (typeof window === "undefined" || !window.speechSynthesis) return;
    if (!text || !text.trim()) return;
    // Always cancel any prior utterance before starting a new one —
    // queueing two utterances back-to-back means the user hears stale
    // content from a previous message they probably already moved past.
    window.speechSynthesis.cancel();

    const u = new SpeechSynthesisUtterance(text);
    // Slightly slower than default (1.0) — anatomy terms benefit from
    // a touch of breathing room.
    u.rate = 0.97;
    u.pitch = 1.0;
    u.onend   = () => setSpeakingId((cur) => (cur === id ? null : cur));
    u.onerror = () => setSpeakingId((cur) => (cur === id ? null : cur));

    setSpeakingId(id);
    window.speechSynthesis.speak(u);
  }, []);

  return { supported, speakingId, speak, cancel };
}
