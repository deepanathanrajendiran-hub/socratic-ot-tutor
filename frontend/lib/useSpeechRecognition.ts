// frontend/lib/useSpeechRecognition.ts
//
// Browser Web Speech API wrapper for STT. Uses the prefixed
// webkitSpeechRecognition where the unprefixed name isn't yet
// declared (Chrome / Edge / Safari support this; Firefox does not).
//
// Behavior:
//   - `start()` requests mic permission and begins streaming partial
//     transcripts to the `transcript` state in real time.
//   - `stop()` finalizes the current sentence and the recognizer
//     auto-stops after a short silence too.
//   - Each new result REPLACES `transcript` (not append) — the caller
//     decides whether to merge into existing text via `onResult`.
//
// SSR safety: `supported` is false on the server and during the
// first render. The actual SpeechRecognition class is only touched
// inside `start()`, which is a user click → always runs in browser.
"use client";
import { useCallback, useEffect, useRef, useState } from "react";

// Minimal type declarations — TS DOM lib doesn't ship complete
// types for the prefixed API in some builds.
interface SpeechRecognitionResultLike {
  isFinal: boolean;
  readonly length: number;
  [index: number]: { transcript: string };
}
interface SpeechRecognitionResultListLike {
  readonly length: number;
  [index: number]: SpeechRecognitionResultLike;
}
interface SpeechRecognitionEventLike extends Event {
  results:    SpeechRecognitionResultListLike;
  resultIndex: number;
}
interface SpeechRecognitionLike {
  continuous:     boolean;
  interimResults: boolean;
  lang:           string;
  onresult:       ((e: SpeechRecognitionEventLike) => void) | null;
  onerror:        ((e: Event) => void) | null;
  onend:          (() => void) | null;
  start:          () => void;
  stop:           () => void;
  abort:          () => void;
}
type SpeechRecognitionCtor = new () => SpeechRecognitionLike;

declare global {
  interface Window {
    SpeechRecognition?:       SpeechRecognitionCtor;
    webkitSpeechRecognition?: SpeechRecognitionCtor;
  }
}

export interface UseSpeechRecognition {
  /** True when the browser exposes a SpeechRecognition class. */
  supported: boolean;
  /** True while the mic is open and streaming partial transcripts. */
  listening: boolean;
  /** Latest transcript (interim or final). Resets on each `start()`. */
  transcript: string;
  /** Last error message (e.g. "not-allowed", "no-speech"), or null. */
  error: string | null;
  /** Begin listening. Optionally pass an onResult callback that
   *  receives interim + final transcripts as they arrive — the
   *  caller decides how to merge into form state. */
  start: (onResult?: (transcript: string, isFinal: boolean) => void) => void;
  /** Stop listening. Auto-fires after a short silence too. */
  stop: () => void;
}

export function useSpeechRecognition(): UseSpeechRecognition {
  const [supported,  setSupported]  = useState(false);
  const [listening,  setListening]  = useState(false);
  const [transcript, setTranscript] = useState("");
  const [error,      setError]      = useState<string | null>(null);
  const ref = useRef<SpeechRecognitionLike | null>(null);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const Ctor = window.SpeechRecognition ?? window.webkitSpeechRecognition;
    setSupported(!!Ctor);
  }, []);

  // Tear down on unmount so route changes don't leak an open mic.
  useEffect(() => {
    return () => {
      try { ref.current?.abort(); } catch { /* noop */ }
    };
  }, []);

  const start = useCallback((
    onResult?: (transcript: string, isFinal: boolean) => void,
  ) => {
    if (typeof window === "undefined") return;
    const Ctor = window.SpeechRecognition ?? window.webkitSpeechRecognition;
    if (!Ctor) {
      setError("SpeechRecognition is not supported in this browser.");
      return;
    }
    // If a previous session is still running, abort it cleanly.
    try { ref.current?.abort(); } catch { /* noop */ }

    const rec = new Ctor();
    rec.continuous     = false;   // stop after a pause; tap mic again to dictate more
    rec.interimResults = true;    // surface partials so the textarea reacts live
    rec.lang           = "en-US"; // could be wired to user setting later

    rec.onresult = (e: SpeechRecognitionEventLike) => {
      let txt = "";
      let isFinal = false;
      // Concatenate every result (final + interim) so the UI shows
      // the full in-progress utterance, not just the latest fragment.
      for (let i = 0; i < e.results.length; i++) {
        const r = e.results[i];
        txt += r[0].transcript;
        if (r.isFinal) isFinal = true;
      }
      setTranscript(txt);
      if (onResult) onResult(txt, isFinal);
    };
    rec.onerror = (e: Event) => {
      const err = (e as Event & { error?: string }).error ?? "stt error";
      // "no-speech" fires when the user opens the mic and stays silent —
      // not a real error, just inform the caller via state.
      setError(err);
      setListening(false);
    };
    rec.onend = () => setListening(false);

    setError(null);
    setTranscript("");
    setListening(true);
    try {
      rec.start();
      ref.current = rec;
    } catch (exc) {
      // start() throws if called while another recognition is in-flight
      // (different from the abort path above when the prior session
      // hadn't fully torn down yet).
      setError(exc instanceof Error ? exc.message : "couldn't start mic");
      setListening(false);
    }
  }, []);

  const stop = useCallback(() => {
    try { ref.current?.stop(); } catch { /* noop */ }
    setListening(false);
  }, []);

  return { supported, listening, transcript, error, start, stop };
}
