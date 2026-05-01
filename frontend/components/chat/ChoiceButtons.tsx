// frontend/components/chat/ChoiceButtons.tsx
//
// Claude-style suggestion pills. When the tutor produces an A/B/C
// mastery menu in its prose, render the three options as clean
// rounded-rectangle buttons below the message. Click → sends the
// letter as the next user turn.
//
// Visual: matches the suggestion-chip pattern in claude.ai —
//   - thin border, white background
//   - just the option text, no "CHOICE A" label
//   - subtle hover: ivory tint
//   - one-line per option, wraps to a column on narrow viewports
//
// Pattern matched (loose so the LLM can reword slightly):
//
//   A) Try a clinical application question for this concept
//   B) Move on to the next topic
//   C) Stop here for now
//
// Edge cases handled:
//   - Bold markers like **A)** — the *s are stripped before parsing.
//   - Mixed punctuation: "A)", "A.", "A:" all match.
//   - Returns null when fewer than 3 options are found — hides the
//     buttons rather than showing a broken half-menu.
"use client";

type Choice = { letter: "A" | "B" | "C"; text: string };


/** Trim the A/B/C lines out of a message so the visible prose doesn't
 *  duplicate the clickable buttons. Returns the raw text unchanged
 *  when no menu was found (so non-menu messages render unaffected).
 *
 *  Strips:
 *    - Every consecutive line that starts with "A)" / "B)" / "C)"
 *      (or "A." / "A:" — same forms parseChoiceMenu accepts)
 *    - Any leading blank lines that become orphaned by the cut
 *
 *  Keeps:
 *    - The "What would you like to do next?" question above the menu —
 *      it's a useful lead-in to the buttons. Drop it manually if it
 *      ever feels redundant. */
export function stripChoiceMenu(raw: string): string {
  if (!raw) return raw;
  // Strip markdown bold so the regex matches "**A)**" too.
  const cleaned = raw.replace(/\*\*([A-C])\*\*/g, "$1");
  const lines = cleaned.split("\n");
  // Find the FIRST menu line. If we don't find a full A/B/C run we
  // bail — we never want to half-truncate a non-menu message.
  let firstIdx = -1;
  for (let i = 0; i < lines.length; i++) {
    if (/^\s*[A-C][.):]\s/.test(lines[i])) { firstIdx = i; break; }
  }
  if (firstIdx === -1) return raw;
  // Walk forward from firstIdx as long as we see menu / blank lines.
  let lastIdx = firstIdx;
  for (let i = firstIdx; i < lines.length; i++) {
    if (/^\s*[A-C][.):]\s/.test(lines[i]) || lines[i].trim() === "") {
      lastIdx = i;
    } else {
      // First non-menu, non-blank line — stop. We don't want to eat
      // unrelated content that happens to follow the menu.
      break;
    }
  }
  // Drop blank lines immediately preceding the menu so we don't leave
  // an awkward trailing newline above where the buttons render.
  let cutFrom = firstIdx;
  while (cutFrom > 0 && lines[cutFrom - 1].trim() === "") cutFrom--;
  const before = lines.slice(0, cutFrom);
  const after = lines.slice(lastIdx + 1);
  return [...before, ...after].join("\n").trimEnd();
}


export function parseChoiceMenu(raw: string): Choice[] | null {
  if (!raw) return null;
  // Strip markdown bold around the letter so "**A)**" works the same as "A)".
  const cleaned = raw.replace(/\*\*([A-C])\*\*/g, "$1");
  // Greedy line-anchored match — "A)" / "A." / "A:" at the start of a
  // (possibly indented) line, followed by the option text up to the
  // line end.
  const re = /^\s*([A-C])[.):]\s*(.+?)\s*$/gm;
  const found = new Map<Choice["letter"], string>();
  for (const m of cleaned.matchAll(re)) {
    const letter = m[1] as Choice["letter"];
    if (!found.has(letter)) found.set(letter, m[2]);
  }
  if (!found.has("A") || !found.has("B") || !found.has("C")) return null;
  return [
    { letter: "A", text: found.get("A")! },
    { letter: "B", text: found.get("B")! },
    { letter: "C", text: found.get("C")! },
  ];
}


export function ChoiceButtons({
  choices,
  onPick,
  disabled,
}: {
  choices: Choice[];
  onPick: (letter: Choice["letter"]) => void;
  disabled?: boolean;
}) {
  return (
    <div
      className="mt-3 flex flex-wrap gap-1.5"
      role="group"
      aria-label="Choose what to do next"
    >
      {choices.map((c) => (
        <button
          key={c.letter}
          type="button"
          onClick={() => !disabled && onPick(c.letter)}
          disabled={disabled}
          aria-label={`Choice ${c.letter}: ${c.text}`}
          className="rounded-xl border border-ivory-200 bg-white px-3.5 py-1.5 text-[13px] text-ink shadow-sm transition hover:border-coral-300 hover:bg-ivory-50 disabled:opacity-50 disabled:cursor-not-allowed ring-focus"
        >
          {c.text}
        </button>
      ))}
    </div>
  );
}
